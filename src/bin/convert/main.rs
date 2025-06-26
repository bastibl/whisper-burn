#![recursion_limit = "512"]
use burn::{module::Module, tensor::backend::Backend};
use whisper::model::{load::*, *};

use burn::config::Config;
use burn::record::{self, DefaultRecorder, Recorder};

fn save_whisper<B: Backend>(whisper: Whisper<B>, name: &str) -> Result<(), record::RecorderError> {
    DefaultRecorder::new().record(whisper.into_record(), name.into())
}

use std::env;

fn main() {
    let model_name = match env::args().nth(1) {
        Some(name) => name,
        None => {
            eprintln!("Model dump folder not provided");
            return;
        }
    };

    type Backend = burn::backend::Wgpu;
    let device = Default::default();

    let (whisper, whisper_config): (Whisper<Backend>, WhisperConfig) =
        match load_whisper(&model_name, &device) {
            Ok(model) => model,
            Err(e) => {
                eprintln!("Error loading model {model_name}: {e}");
                return;
            }
        };

    println!("Saving model...");
    if let Err(e) = save_whisper(whisper, &model_name) {
        eprintln!("Error saving model {model_name}: {e}");
        return;
    }

    println!("Saving config...");
    if let Err(e) = whisper_config.save(format!("{model_name}.cfg")) {
        eprintln!("Error saving config for {model_name}: {e}");
        return;
    }

    println!("Finished.");
}
