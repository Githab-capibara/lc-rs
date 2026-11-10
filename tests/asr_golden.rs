//! End-to-end checks of the three ASR backends against reference transcripts.
//!
//! These tests need the model checkpoints (~5.8 GB in total), so they are
//! ignored by default. Run them explicitly once the models are installed:
//!
//! ```text
//! lc-rs model download --all
//! cargo test --release -- --ignored
//! ```
//!
//! The expectations are substring checks rather than exact matches: the
//! transcripts are compared against the reference *wording*, while the
//! punctuation and the occasional word differ between the ONNX Runtime
//! builds (the reference implementation runs the same graphs through a
//! different runtime).

use std::path::Path;

use lc_rs::ml::nemotron::Nemotron;
use lc_rs::ml::parakeet::Parakeet;
use lc_rs::ml::qwen::Qwen;
use lc_rs::models::{MODEL_NEMOTRON, MODEL_PARAKEET, MODEL_QWEN, get_model};

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn model_dir(key: &str) -> std::path::PathBuf {
    get_model(key).expect("unknown model").local_dir()
}

fn samples(name: &str) -> Vec<f32> {
    lc_rs::audio::read_samples(&fixture(name)).expect("cannot read the fixture")
}

#[test]
#[ignore = "needs the Parakeet checkpoint"]
fn parakeet_transcribes_the_reference_english_sample() {
    let mut model = Parakeet::load(&model_dir(MODEL_PARAKEET)).expect("cannot load Parakeet");
    let result = model
        .transcribe(&samples("osr_us_000_0010_16k.wav"))
        .expect("Parakeet failed");
    let text = result.text.to_lowercase();
    // The wording is stable across resampling variants, the individual nouns
    // are not: the 8 kHz source is band-limited, so the model hears slightly
    // different words depending on the interpolation filter.
    assert!(
        text.contains("glue the sheet"),
        "unexpected transcript: {}",
        result.text
    );
    assert!(
        text.contains("depth of a well"),
        "unexpected transcript: {}",
        result.text
    );
}

#[test]
#[ignore = "needs the Nemotron checkpoint"]
fn nemotron_transcribes_the_ukrainian_sample() {
    let mut model = Nemotron::load(&model_dir(MODEL_NEMOTRON)).expect("cannot load Nemotron");
    model.set_language(lc_rs::ml::nemotron::prompt_for_language("uk"));
    let result = model
        .transcribe(&samples("nemotron_uk.wav"))
        .expect("Nemotron failed");
    assert!(
        result.text.contains("країни"),
        "unexpected transcript: {}",
        result.text
    );
}

#[test]
#[ignore = "needs the Nemotron checkpoint"]
fn nemotron_transcribes_the_german_sample() {
    let mut model = Nemotron::load(&model_dir(MODEL_NEMOTRON)).expect("cannot load Nemotron");
    model.set_language(lc_rs::ml::nemotron::prompt_for_language("de"));
    let result = model
        .transcribe(&samples("nemotron_de.wav"))
        .expect("Nemotron failed");
    assert!(
        result.text.contains("Wurst"),
        "unexpected transcript: {}",
        result.text
    );
}

#[test]
#[ignore = "needs the Qwen checkpoint"]
fn qwen_transcribes_the_ukrainian_sample() {
    let mut model = Qwen::load(&model_dir(MODEL_QWEN)).expect("cannot load Qwen");
    model.set_language(Some("Russian"));
    let result = model
        .transcribe(&samples("nemotron_uk.wav"))
        .expect("Qwen failed");
    assert!(
        result.text.contains("пытаешься"),
        "unexpected transcript: {}",
        result.text
    );
    // The reference implementation hears «країна» in this fixture, and so
    // does the fp32 decoder. The quantised one hears «страна» instead, so
    // this assertion also pins the fidelity of the default export.
    assert!(
        result.text.to_lowercase().contains("краина"),
        "the transcript lost the reference wording: {}",
        result.text
    );
    assert_eq!(result.language.as_deref(), Some("Russian"));
}

#[test]
#[ignore = "needs the Qwen checkpoint"]
fn qwen_detects_the_language_when_it_is_not_forced() {
    let mut model = Qwen::load(&model_dir(MODEL_QWEN)).expect("cannot load Qwen");
    let result = model
        .transcribe(&samples("nemotron_uk.wav"))
        .expect("Qwen failed");
    assert_eq!(result.language.as_deref(), Some("Russian"));
    assert!(!result.text.is_empty());
}

#[test]
#[ignore = "needs the Parakeet checkpoint"]
fn parakeet_reports_english_like_the_reference() {
    let result =
        lc_rs::ml::asr::transcribe(MODEL_PARAKEET, &fixture("osr_us_000_0010_16k.wav"), "en")
            .expect("Parakeet failed");
    // The reference implementation hard-codes `en` for this checkpoint.
    assert_eq!(result.language, "en");
}

#[test]
#[ignore = "needs the Nemotron checkpoint"]
fn nemotron_reports_auto_like_the_reference() {
    let result = lc_rs::ml::asr::transcribe(MODEL_NEMOTRON, &fixture("nemotron_uk.wav"), "uk")
        .expect("Nemotron failed");
    // The reference implementation runs this checkpoint with
    // `language="auto"` and reports that mode, not the routed code.
    assert_eq!(result.language, "auto");
}

#[test]
#[ignore = "needs the Qwen checkpoint"]
fn qwen_reports_the_language_the_model_announced() {
    let result = lc_rs::ml::asr::transcribe(MODEL_QWEN, &fixture("nemotron_uk.wav"), "ru")
        .expect("Qwen failed");
    // The model announces "Russian"; the artefacts carry its code.
    assert_eq!(result.language, "ru");
}
