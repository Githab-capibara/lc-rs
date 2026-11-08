//! Dump the Whisper log-mel features of a WAV file.
//!
//! The output is the matrix the Qwen3-ASR encoder consumes: one row per mel
//! band, one column per frame, `float32` in the `[-1, 1]` range of the
//! Whisper front-end. The tool exists to compare the Rust front-end against
//! the reference `transformers` extractor:
//!
//! ```text
//! cargo run --release --example mel_dump -- fixture.wav mel.txt
//! ```
//!
//! The matrix is written as text (one row per band, values separated by
//! spaces) so it can be diffed or loaded with any numeric tool.

use std::path::PathBuf;

use lc_rs::ml::features::{MelConfig, log_mel_spectrogram};

fn main() {
    let mut args = std::env::args().skip(1);
    let wav = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: mel_dump <wav> <output.txt>");
        std::process::exit(2);
    }));
    let output = PathBuf::from(args.next().unwrap_or_else(|| {
        eprintln!("usage: mel_dump <wav> <output.txt>");
        std::process::exit(2);
    }));

    let samples = lc_rs::audio::read_samples(&wav).expect("cannot read the WAV file");
    let features = log_mel_spectrogram(&samples, &MelConfig::whisper(128)).expect("mel failed");

    let mut text = format!("{} {}\n", features.n_mels, features.n_frames);
    for band in 0..features.n_mels {
        let row = (0..features.n_frames)
            .map(|frame| format!("{:.7}", features.at(band, frame)))
            .collect::<Vec<String>>()
            .join(" ");
        text.push_str(&row);
        text.push('\n');
    }
    std::fs::write(&output, text).expect("cannot write the output file");
    println!(
        "{}: {} bands x {} frames -> {}",
        wav.display(),
        features.n_mels,
        features.n_frames,
        output.display()
    );
}
