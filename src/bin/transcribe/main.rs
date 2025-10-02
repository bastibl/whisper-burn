#![recursion_limit = "512"]
use burn::{config::Config, module::Module, tensor::backend::Backend};
use hound::{self, SampleFormat};
use strum::IntoEnumIterator;
use whisper_burn::model::*;
use whisper_burn::token::Language;
use whisper_burn::transcribe::waveform_to_text;

fn load_audio_waveform(filename: &str) -> hound::Result<(Vec<f32>, usize)> {
    let reader = hound::WavReader::open(filename)?;
    let spec = reader.spec();

    let _duration = reader.duration() as usize;
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as usize;
    let _bits_per_sample = spec.bits_per_sample;
    let sample_format = spec.sample_format;

    assert_eq!(sample_rate, 16000, "The audio sample rate must be 16k.");
    assert_eq!(channels, 1, "The audio must be single-channel.");

    let max_int_val = 2_u32.pow(spec.bits_per_sample as u32 - 1) - 1;

    let floats = match sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().collect::<hound::Result<_>>()?,
        SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|s| s as f32 / max_int_val as f32))
            .collect::<hound::Result<_>>()?,
    };

    Ok((floats, sample_rate))
}

use burn::record::{DefaultRecorder, Recorder, RecorderError};
use whisper_burn::token::Gpt2Tokenizer;

fn load_whisper_model_file<B: Backend>(
    config: &WhisperConfig,
    filename: &str,
    device: &B::Device,
) -> Result<Whisper<B>, RecorderError> {
    DefaultRecorder::new()
        .load(filename.into(), device)
        .map(|record| config.init(device).load_record(record))
}

use std::{env, fs, process};

fn main() {
    // type Backend = burn::backend::Vulkan;
    type Backend = burn::backend::WebGpu;
    let device = Default::default();

    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        eprintln!(
            "Usage: {} <model name> <audio file> <lang> <transcription file>",
            args[0]
        );
        process::exit(1);
    }

    let wav_file = &args[2];
    let text_file = &args[4];

    let lang_str = &args[3];
    let lang = match Language::iter().find(|lang| lang.as_str() == lang_str) {
        Some(lang) => lang,
        None => {
            eprintln!("Invalid language abbreviation: {lang_str}");
            process::exit(1);
        }
    };

    let model_name = &args[1];

    println!("Loading waveform...");
    let (waveform, sample_rate) = match load_audio_waveform(wav_file) {
        Ok((w, sr)) => (w, sr),
        Err(e) => {
            eprintln!("Failed to load audio file: {e}");
            process::exit(1);
        }
    };

    let bpe = match Gpt2Tokenizer::new(model_name) {
        Ok(bpe) => bpe,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {e}");
            process::exit(1);
        }
    };

    let whisper_config = match WhisperConfig::load(format!("{model_name}/{model_name}.cfg")) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load whisper config: {e}");
            process::exit(1);
        }
    };

    println!("Loading model...");
    let whisper: Whisper<Backend> = match load_whisper_model_file(
        &whisper_config,
        &format!("{model_name}/{model_name}"),
        &device,
    ) {
        Ok(whisper_model) => whisper_model,
        Err(e) => {
            eprintln!("Failed to load whisper model file: {e}");
            process::exit(1);
        }
    };

    let whisper = whisper.to_device(&device);
    println!("Loading model... done");

    let (text, _tokens) = match waveform_to_text(&whisper, &bpe, lang, waveform, sample_rate) {
        Ok((text, tokens)) => (text, tokens),
        Err(e) => {
            eprintln!("Error during transcription: {e}");
            process::exit(1);
        }
    };

    fs::write(text_file, text).unwrap_or_else(|e| {
        eprintln!("Error writing transcription file: {e}");
        process::exit(1);
    });

    println!("Transcription finished.");
}
