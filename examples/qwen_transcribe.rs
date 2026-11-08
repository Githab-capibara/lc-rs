//! Transcribe a WAV file with the Qwen3-ASR backend, from an explicit
//! checkpoint directory.
//!
//! The CLI always loads the checkpoint from the model registry; this
//! example takes the directory as an argument, which is what one needs when
//! comparing two exports of the same model — the int4 decoder the registry
//! ships and the fp32 decoder — without touching the registry.
//!
//! ```text
//! cargo run --release --example qwen_transcribe -- fixture.wav <checkpoint-dir> [language]
//! ```
//!
//! The checkpoint directory must hold the eight artefacts of the model:
//! `encoder.onnx`, `decoder_init.int4.onnx`, `decoder_step.int4.onnx`,
//! `decoder_weights.int4.data`, `embed_tokens.bin`, `tokenizer.json`,
//! `config.json` and `preprocessor_config.json`.

use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let wav = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: qwen_transcribe <wav> <checkpoint-dir> [language]");
        std::process::exit(2);
    }));
    let dir = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: qwen_transcribe <wav> <checkpoint-dir> [language]");
        std::process::exit(2);
    }));
    // The language is forced by its canonical name; `auto` lets the model
    // detect it.
    let language = args.next().unwrap_or_else(|| "auto".to_string());

    let samples = lc_rs::audio::read_samples(&wav).expect("cannot read the WAV file");
    let mut model = lc_rs::ml::qwen::Qwen::load(&dir).expect("cannot load the checkpoint");
    model.set_language(lc_rs::ml::qwen::language_name(&language));

    let result = model
        .transcribe(&samples)
        .expect("the transcription failed");
    println!("text: {}", result.text);
    println!(
        "language: {}",
        result.language.as_deref().unwrap_or("undetermined")
    );
}
