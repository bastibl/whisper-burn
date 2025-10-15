use super::*;
use burn::tensor::Shape;
use burn::{
    module::Module,
    nn::{
        self, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig, Conv1dRecord},
    },
    prelude::*,
    tensor::{Tensor, backend::Backend},
};
use npyz::NpyFile;
use std::error::Error;
use std::io::Read;

fn numpy_to_tensor<B: Backend, R: Read, const D: usize>(
    numpy_data: NpyFile<R>,
    device: &B::Device,
) -> Tensor<B, D> {
    let v: Vec<f32> = numpy_data.into_vec().unwrap();
    let shape: Shape = v[0..D]
        .iter()
        .map(|&v| v as usize)
        .collect::<Vec<_>>()
        .into();
    // let end = std::cmp::min(v.len(), D + 10);
    // println!("first {:?}   shape {:?}   next {:?}", &v[0..D], shape, &v[D..end]);
    Tensor::<B, 1>::from_floats(&v[D..], device).reshape(shape)
}

fn load_tensor<B: Backend, const D: usize>(
    name: &str,
    path: &str,
    device: &B::Device,
) -> Result<Tensor<B, D>, Box<dyn Error>> {
    let tensor_path = format!("{path}/{name}.npy");

    let buf = std::fs::read(tensor_path)?;

    let tensor_numpy = NpyFile::new(&buf[..])?;

    let tensor = numpy_to_tensor(tensor_numpy, device);

    Ok(tensor)
}

fn load_f32<B: Backend>(name: &str, path: &str, device: &B::Device) -> Result<f32, Box<dyn Error>> {
    load_tensor::<B, 1>(name, path, device).map(|t| t.into_scalar().elem::<f32>())
}

fn load_usize<B: Backend>(
    name: &str,
    path: &str,
    device: &B::Device,
) -> Result<usize, Box<dyn Error>> {
    load_tensor::<B, 1>(name, path, device).map(|t| t.into_scalar().elem::<f32>() as usize)
}

fn load_linear<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<nn::Linear<B>, Box<dyn Error>> {
    let weight = load_tensor::<B, 2>("weight", path, device)?;
    let bias = load_tensor::<B, 1>("bias", path, device).ok();

    let record = nn::LinearRecord {
        weight: Param::from_tensor(weight),
        bias: bias.map(|v| Param::from_tensor(v)),
    };

    let linear: nn::Linear<B> = nn::LinearConfig::new(3, 3).init(device).load_record(record);
    Ok(linear)
}

fn load_layer_norm<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<nn::LayerNorm<B>, Box<dyn Error>> {
    let weight = load_tensor::<B, 1>("weight", path, device)?;
    let bias = load_tensor::<B, 1>("bias", path, device)?;
    let eps = load_f32::<B>("eps", path, device)? as f64;

    let [n_state] = weight.dims();

    let record = nn::LayerNormRecord {
        gamma: Param::from_tensor(weight),
        beta: Param::from_tensor(bias),
        epsilon: <f64 as Module<B>>::into_record(eps),
    };

    let layer_norm: nn::LayerNorm<B> = nn::LayerNormConfig::new(n_state)
        .init(device)
        .load_record(record);

    Ok(layer_norm)
}

fn load_multihead_self_attention<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<MultiHeadSelfAttention<B>, Box<dyn Error>> {
    let query = load_linear(&format!("{path}/query"), device)?;
    let key = load_linear(&format!("{path}/key"), device)?;
    let value = load_linear(&format!("{path}/value"), device)?;
    let out = load_linear(&format!("{path}/out"), device)?;

    let n_head: usize = load_usize::<B>("n_head", path, device)?;

    // Initializing attention block
    let attention_block = MultiHeadSelfAttention {
        n_head,
        query,
        key,
        value,
        out,
    };

    Ok(attention_block)
}

fn load_multihead_cross_attention<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<MultiHeadCrossAttention<B>, Box<dyn Error>> {
    let query = load_linear(&format!("{path}/query"), device)?;
    let key = load_linear(&format!("{path}/key"), device)?;
    let value = load_linear(&format!("{path}/value"), device)?;
    let out = load_linear(&format!("{path}/out"), device)?;

    let n_head: usize = load_usize::<B>("n_head", path, device)?;

    // Initializing attention block
    let attention_block = MultiHeadCrossAttention {
        n_head,
        query,
        key,
        value,
        out,
    };

    Ok(attention_block)
}

fn load_mlp<B: Backend>(path: &str, device: &B::Device) -> Result<MLP<B>, Box<dyn Error>> {
    let lin1 = load_linear(&format!("{path}/mlp1"), device)?;
    let lin2 = load_linear(&format!("{path}/mlp2"), device)?;

    let gelu = nn::Gelu::new();

    let mlp = MLP { lin1, lin2, gelu };

    Ok(mlp)
}

fn load_conv1d<B: Backend>(
    path: &str,
    config: Conv1dConfig,
    device: &B::Device,
) -> Result<Conv1d<B>, Box<dyn Error>> {
    let weight = load_tensor::<B, 3>("weight", path, device)?;
    let bias = load_tensor::<B, 1>("bias", path, device)?;

    let record = Conv1dRecord {
        weight: Param::from_tensor(weight),
        bias: Some(Param::from_tensor(bias)),
        stride: <usize as Module<B>>::into_record(1),
        kernel_size: <usize as Module<B>>::into_record(1),
        dilation: <usize as Module<B>>::into_record(1),
        groups: <usize as Module<B>>::into_record(1),
        padding: <usize as Module<B>>::into_record(10),
    };

    let conv1d: Conv1d<B> = config.init(device).load_record(record);
    Ok(conv1d)
}

fn load_residual_encoder_attention_block<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<ResidualEncoderAttentionBlock<B>, Box<dyn Error>> {
    let attn = load_multihead_self_attention(&format!("{path}/attn"), device)?;
    let attn_ln = load_layer_norm(&format!("{path}/attn_ln"), device)?;
    let mlp = load_mlp(&format!("{path}/mlp"), device)?;
    let mlp_ln = load_layer_norm(&format!("{path}/mlp_ln"), device)?;

    let residual_block = ResidualEncoderAttentionBlock {
        attn,
        attn_ln,
        mlp,
        mlp_ln,
    };

    Ok(residual_block)
}

fn load_residual_decoder_attention_block<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<ResidualDecoderAttentionBlock<B>, Box<dyn Error>> {
    let attn = load_multihead_self_attention(&format!("{path}/attn"), device)?;
    let attn_ln = load_layer_norm(&format!("{path}/attn_ln"), device)?;
    let cross_attn = load_multihead_cross_attention(&format!("{path}/cross_attn"), device)?;
    let cross_attn_ln = load_layer_norm(&format!("{path}/cross_attn_ln"), device)?;
    let mlp = load_mlp(&format!("{path}/mlp"), device)?;
    let mlp_ln = load_layer_norm(&format!("{path}/mlp_ln"), device)?;

    let residual_block = ResidualDecoderAttentionBlock {
        attn,
        attn_ln,
        cross_attn,
        cross_attn_ln,
        mlp,
        mlp_ln,
    };

    Ok(residual_block)
}

fn load_audio_encoder<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<(AudioEncoder<B>, AudioEncoderConfig), Box<dyn Error>> {
    let n_mels = load_usize::<B>("n_mels", path, device)?;
    let n_audio_state = load_usize::<B>("n_audio_state", path, device)?;

    let conv1_config =
        Conv1dConfig::new(n_mels, n_audio_state, 3).with_padding(PaddingConfig1d::Explicit(1));
    let conv2_config = Conv1dConfig::new(n_audio_state, n_audio_state, 3)
        .with_padding(PaddingConfig1d::Explicit(1))
        .with_stride(2);

    let conv1 = load_conv1d(&format!("{path}/conv1"), conv1_config, device)?;
    let conv2 = load_conv1d(&format!("{path}/conv2"), conv2_config, device)?;

    let n_layer = load_usize::<B>("n_layer", path, device)?;

    let blocks: Vec<ResidualEncoderAttentionBlock<B>> = (0..n_layer)
        .map(|i| load_residual_encoder_attention_block(&format!("{path}/block_{i}"), device))
        .collect::<Result<_, _>>()?;

    let ln_post = load_layer_norm(&format!("{path}/ln_post"), device)?;
    let positional_embedding = load_tensor::<B, 2>("positional_embedding", path, device)?;

    let [n_audio_ctx, _] = positional_embedding.dims();

    let n_head = blocks[0].attn.n_head;

    let audio_encoder = AudioEncoder {
        conv1,
        gelu1: nn::Gelu::new(),
        conv2,
        gelu2: nn::Gelu::new(),
        blocks,
        ln_post,
        positional_embedding: Param::from_tensor(positional_embedding),
        n_audio_ctx,
        n_mels,
    };

    let config = AudioEncoderConfig {
        n_mels,
        n_audio_ctx,
        n_audio_state,
        n_audio_head: n_head,
        n_audio_layer: n_layer,
    };

    Ok((audio_encoder, config))
}

fn load_text_decoder<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<(TextDecoder<B>, TextDecoderConfig), Box<dyn Error>> {
    let token_embedding = load_tensor::<B, 2>("token_embedding/weight", path, device)?;
    let positional_embedding = load_tensor::<B, 2>("positional_embedding", path, device)?;

    let n_layer = load_usize::<B>("n_layer", path, device)?;
    println!("n_layer {n_layer}");
    let blocks: Vec<ResidualDecoderAttentionBlock<B>> = (0..n_layer)
        .map(|i| load_residual_decoder_attention_block(&format!("{path}/block_{i}"), device))
        .collect::<Result<_, _>>()?;

    let n_text_head = blocks[0].attn.n_head;

    println!("load layer norm");
    let ln = load_layer_norm(&format!("{path}/ln"), device)?;

    let [n_text_ctx, n_text_state] = positional_embedding.dims();
    let mask = attn_decoder_mask(n_text_ctx, device);

    let [n_vocab, _] = token_embedding.dims();

    let text_decoder = TextDecoder {
        token_embedding: Param::from_tensor(token_embedding),
        positional_embedding: Param::from_tensor(positional_embedding),
        blocks,
        ln,
        mask: Param::from_tensor(mask),
        n_text_ctx,
        n_vocab,
    };

    let config = TextDecoderConfig {
        n_vocab,
        n_text_ctx,
        n_text_state,
        n_text_head,
        n_text_layer: n_layer,
    };

    Ok((text_decoder, config))
}

pub fn load_whisper<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<(Whisper<B>, WhisperConfig), Box<dyn Error>> {
    let (encoder, encoder_config) = load_audio_encoder(&format!("{path}/encoder"), device)?;
    let (decoder, decoder_config) = load_text_decoder(&format!("{path}/decoder"), device)?;

    let whisper = Whisper { encoder, decoder };

    let config = WhisperConfig {
        audio_encoder_config: encoder_config,
        text_decoder_config: decoder_config,
    };

    Ok((whisper, config))
}
