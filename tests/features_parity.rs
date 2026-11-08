//! Numerical parity of the Whisper front-end with the reference extractor.
//!
//! The fixture holds values produced by `transformers` 4.57.6
//! (`WhisperFeatureExtractor`) for `tests/fixtures/nemotron_uk.wav` — the
//! very code path the ONNX export was produced with. The test therefore
//! guards the mel scale, the filter-bank normalisation, the dynamic-range
//! clamp and the affine rescaling in one go.

use std::path::Path;

use lc_rs::ml::features::{MelConfig, log_mel_spectrogram};

/// Samples the Whisper front-end consumes: 30 s at 16 kHz.
const N_SAMPLES: usize = 480_000;

/// Largest deviation tolerated against the reference values.
const TOLERANCE: f32 = 5e-3;

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn whisper_front_end_matches_the_reference_extractor() {
    let wav = fixture_dir().join("nemotron_uk.wav");
    let samples = lc_rs::audio::read_samples(&wav).expect("cannot read the fixture");
    assert_eq!(
        samples.len(),
        96_000,
        "the fixture must be 6 s of 16 kHz audio"
    );

    let mut padded = samples;
    padded.truncate(N_SAMPLES);
    padded.resize(N_SAMPLES, 0.0);

    let features = log_mel_spectrogram(&padded, &MelConfig::whisper(128)).expect("mel failed");
    assert_eq!(features.n_mels, 128);
    assert_eq!(features.n_frames, 3001, "30 s of audio yields 3001 frames");

    let raw =
        std::fs::read(fixture_dir().join("whisper_mel_uk.bin")).expect("cannot read the fixture");
    assert_eq!(raw.len(), 14 * 50 * 4, "unexpected fixture size");

    let bands = [0usize, 8, 16, 24, 32, 40, 53, 64, 72, 80, 92, 104, 116, 127];
    let frames = (0..3000).step_by(60).collect::<Vec<usize>>();
    assert_eq!(frames.len(), 50);

    let mut worst = 0.0_f32;
    for (row, &band) in bands.iter().enumerate() {
        for (column, &frame) in frames.iter().enumerate() {
            let offset = (row * frames.len() + column) * 4;
            let expected = f32::from_le_bytes([
                raw[offset],
                raw[offset + 1],
                raw[offset + 2],
                raw[offset + 3],
            ]);
            let actual = features.at(band, frame);
            let deviation = (actual - expected).abs();
            assert!(
                deviation <= TOLERANCE,
                "band {band} frame {frame}: {actual} differs from {expected} by {deviation}"
            );
            worst = worst.max(deviation);
        }
    }
    assert!(
        worst > 0.0,
        "the comparison must actually compare something"
    );
}
