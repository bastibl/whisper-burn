use crate::audio::{max_waveform_samples, prep_audio};
use crate::beam;
use crate::model::*;
use crate::token::{self, *};
use burn::{
    module::Module,
    prelude::*,
    tensor::{ElementConversion, activation::log_softmax, backend::Backend},
};
use std::iter;
use std::ops::Div;

pub fn waveform_to_text<B: Backend>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer,
    lang: Language,
    waveform: Vec<f32>,
    sample_rate: usize,
) -> token::Result<(String, Vec<usize>)> {
    let device = whisper.devices()[0].clone();

    let n_ctx_max_encoder = whisper.encoder_ctx_size();
    let padding = 300;
    let n_waveform_samples_per_window = max_waveform_samples(n_ctx_max_encoder - padding) * 3 / 4;

    let n_mels = whisper.encoder_mel_size();
    let mel_iter = waveform_to_mel_tensor(
        waveform,
        sample_rate,
        n_waveform_samples_per_window,
        &device,
        n_mels,
    );

    let mut text = String::new();
    let mut tokens: Vec<usize> = Vec::new();
    for mel in mel_iter {
        let (_new_text, new_tokens) = mels_to_text(whisper, bpe, lang, mel, padding)?;

        println!("tokens {tokens:?}");
        println!("new tokens {new_tokens:?}");

        if let Some((prev_index, curr_index)) =
            find_chunk_overlap(&tokens[..], &new_tokens[..], 20, 3)
            && prev_index > 0
            && curr_index > 0
        {
            println!("prev index {prev_index}     curr_index {curr_index}");
            tokens.truncate(prev_index);
            tokens.extend(&new_tokens[curr_index..]);
        } else {
            text += &*bpe.decode(&tokens[..], true).unwrap();
            println!("{text}");
            tokens = new_tokens;
        }
    }
    text += &*bpe.decode(&tokens[..], true).unwrap();
    println!("{text}");

    Ok((text, tokens))
}

fn waveform_to_mel_tensor<B: Backend>(
    waveform: Vec<f32>,
    sample_rate: usize,
    window_length_samples: usize,
    device: &B::Device,
    n_mels: usize,
) -> impl Iterator<Item = Tensor<B, 3>> + use<'_, B> {
    let chunk_overlap = sample_rate * 3;
    let n_samples_per_tensor = window_length_samples;
    let shift = n_samples_per_tensor.saturating_sub(chunk_overlap).max(1);
    let iter_len = waveform.len().saturating_sub(1).div(shift) + 1;

    (0..iter_len).map(move |i| {
        let start = i * shift;
        let end = (start + n_samples_per_tensor).min(waveform.len());

        let slice = &waveform[start..end];

        let waveform: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(slice.to_vec(), [slice.len()]), device);
        prep_audio(waveform.unsqueeze(), sample_rate as f64, n_mels)
    })
}

#[derive(Clone, Debug)]
struct BeamSearchToken(usize);

pub fn mels_to_text<B: Backend>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer,
    lang: Language,
    mels: Tensor<B, 3>,
    padding: usize,
) -> token::Result<(String, Vec<usize>)> {
    let device = mels.device();

    // 1500
    let n_ctx_max_encoder = whisper.encoder_ctx_size();

    // [1, 80, 900]
    let [_n_channel, n_mel, n_ctx] = mels.dims();
    if n_ctx + padding > n_ctx_max_encoder {
        println!(
            "Audio has length of {} which exceeds maximum length {}. It will be clipped.",
            n_ctx + padding,
            n_ctx_max_encoder
        );
    }

    // the zero padding helps whisper determine end of text
    let mels = Tensor::cat(
        vec![
            mels.slice([0..1, 0..n_mel, 0..n_ctx.min(n_ctx_max_encoder - padding)]),
            Tensor::zeros([1, n_mel, padding], &device),
        ],
        2,
    );
    // [1, 600 (max), 384]
    let encoder_output = whisper.forward_encoder(mels);

    let start_token = bpe.special_token(SpecialToken::StartofTranscript).unwrap();
    let transcription_token = bpe.special_token(SpecialToken::Transcribe).unwrap();
    let lang_token = bpe.special_token(SpecialToken::Language(lang)).unwrap();
    let end_token = bpe.special_token(SpecialToken::EndofText).unwrap();
    let notimestamp = bpe.special_token(SpecialToken::NoTimeStamps).unwrap();

    let initial_tokens = vec![start_token, lang_token, transcription_token, notimestamp];

    type BeamNode = beam::BeamNode<BeamSearchToken>;

    let initial_tokens = BeamNode {
        seq: initial_tokens.into_iter().map(BeamSearchToken).collect(),
        log_prob: 0.0,
    };

    let neg_infty = -f32::INFINITY;

    let vocab_size = bpe.vocab_size();
    let special_tokens_maskout: Vec<f32> = (0..vocab_size)
        .map(|token| {
            if bpe.is_special(token) || (50364..=51864).contains(&token) {
                neg_infty
            } else {
                0.0
            }
        })
        .collect();

    let special_tokens_maskout: Tensor<B, 1> = Tensor::from_data(
        TensorData::new(special_tokens_maskout, [vocab_size]),
        &device,
    );

    let time_token_maskout: Vec<f32> = (0..vocab_size)
        .map(|token| {
            if (50364..=51864).contains(&token) {
                neg_infty
            } else {
                0.0
            }
        })
        .collect();

    let time_token_maskout: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(time_token_maskout, [vocab_size]), &device);

    let beamsearch_next = |beams: &[BeamNode]| {
        // convert tokens into tensor
        let max_seq_len = beams.iter().map(|beam| beam.seq.len()).max().unwrap_or(0);
        let flattened_tokens: Vec<_> = beams
            .iter()
            .flat_map(|beam| {
                let additional_tokens = max_seq_len - beam.seq.len();
                beam.seq
                    .iter()
                    .map(|btok| btok.0)
                    .chain(iter::once(0).cycle().take(additional_tokens))
            })
            .collect();

        let vecu32: Vec<u32> = flattened_tokens.into_iter().map(|x| x as u32).collect();
        let vecu32_len = vecu32.len();
        let token_tensor: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(vecu32, [vecu32_len]), &device);
        let token_tensor = token_tensor.reshape([beams.len(), max_seq_len]);

        let logits =
            whisper.forward_decoder(token_tensor, encoder_output.clone().repeat(&[beams.len()]));

        // Safety: make sure axis 2 is vocab
        let [_, _, v] = logits.dims();
        assert_eq!(v, vocab_size, "decoder output not [B,S,V]");

        // Generated-length accounting (don’t use global max_seq_len threshold)
        let initial_len = 4; // SOT, <|lang|>, <|transcribe|>, <|notimestamps|>
        let min_text_tokens: usize = 4; // keep EOT blocked for first N generated tokens

        // For each beam, score only *its* current step and mask *that* vector
        let beam_log_probs = beams.iter().enumerate().map(|(i, beam)| {
            // current step for this beam
            let token_index = beam.seq.len().saturating_sub(1);

            // slice per-beam, per-step logits -> [V]
            let mut step_logits = logits
                .clone()
                .slice([i..i + 1, token_index..token_index + 1, 0..vocab_size]) // [1,1,V]
                .squeeze::<2>(0) // [1,V] -> [V]
                .squeeze::<1>(0); // [V]

            // Decide which mask to apply for this beam at this step
            let decoded_so_far = beam.seq.len().saturating_sub(initial_len);
            // Always suppress timestamps; and for the first N generated tokens, also suppress specials/EOT.
            let mask_vec = if decoded_so_far < min_text_tokens {
                special_tokens_maskout.clone()
            } else {
                time_token_maskout.clone()
            };

            // Apply mask to the *exact* vector we will sample from
            step_logits = step_logits + mask_vec;

            // (Optional) sanity check: timestamps must be ~-inf here
            // let ts_max = step_logits.clone().slice([50364..51865]).max().into_scalar();
            // println!("beam {i} @pos {token_index}: max TS logit = {ts_max}");

            // Now compute log-probs for this step along vocab axis
            let step_log_probs = log_softmax(step_logits, 0); // [V]

            // Return host data for enumeration (same as your previous code expected)
            step_log_probs.into_data()
        });

        beam_log_probs
            .zip(beams)
            .map(|(log_probs, beam)| {
                log_probs
                    .iter()
                    .map(|log_prob: f32| log_prob.elem::<f64>())
                    .enumerate()
                    .map(|(token_id, log_prob)| {
                        (BeamSearchToken(token_id), beam.log_prob + log_prob)
                    })
                    .collect()
            })
            .collect()
    };

    let beamsearch_is_finished = |toks: &[BeamSearchToken]| {
        if let Some(btok) = toks.last() {
            btok.0 == end_token
        } else {
            false
        }
    };

    let beam_size = 5;
    let max_depth = 30;
    let tokens: Vec<_> = beam::beam_search(
        vec![initial_tokens],
        beamsearch_next,
        beamsearch_is_finished,
        beam_size,
        max_depth,
    )
    .into_iter()
    .map(|btok| btok.0)
    .collect();

    // println!("Generated tokens: {:?}", tokens);
    // for (i, &token) in tokens.iter().enumerate() {
    //     if let Ok(text_part) = bpe.decode(&[token], false) {
    //         println!("Token {}: {} -> '{}'", i, token, text_part);
    //     }
    // }

    let text = bpe.decode(&tokens[..], false)?;
    Ok((text, tokens))
}

pub fn find_chunk_overlap(
    prev_tokens: &[usize],
    curr_tokens: &[usize],
    max_n_offsets: usize,
    min_n_overlaps: usize,
) -> Option<(usize, usize)> {
    let mut max_overlap = 0;
    let mut max_overlap_indices = (0, 0);
    let n_offsets = prev_tokens.len().min(curr_tokens.len()).min(max_n_offsets);

    for offset in 0..n_offsets {
        let prev_start_index = prev_tokens.len() - 1 - offset;
        let mut overlap_iter = prev_tokens
            .iter()
            .skip(prev_start_index)
            .zip(curr_tokens.iter())
            .enumerate()
            .filter(|&(_, (&old, &new))| old == new);

        let n_overlap = overlap_iter.clone().count();
        if n_overlap > max_overlap {
            max_overlap = n_overlap;
            let curr_overlap_index = overlap_iter.next().unwrap().0;
            let prev_overlap_index = prev_start_index + curr_overlap_index;
            max_overlap_indices = (prev_overlap_index, curr_overlap_index)
        }
    }
    if max_overlap >= min_n_overlaps {
        Some(max_overlap_indices)
    } else {
        None
    }
}
