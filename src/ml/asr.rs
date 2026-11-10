//! Stage 3: route a 16 kHz mono WAV to the right local ASR model.
//!
//! Language routing (decided by the language-ID stage, or the `--language`
//! flag):
//!
//! * `ru` → Qwen3-ASR 0.6B
//! * `en` → NVIDIA Parakeet TDT 0.6B v3
//! * anything else (or unclear) → NVIDIA Nemotron 3.5 ASR Streaming 0.6B
//!
//! Every model runs in-process through ONNX Runtime.
//!
//! The language the artefacts report follows the reference implementation
//! rather than the routing decision: Parakeet reports `en`, Nemotron
//! reports `auto` (the reference always runs it in automatic detection
//! mode), and Qwen reports the language the model itself announced.

use std::path::Path;

use crate::chapters::Segment;
use crate::errors::{Error, Result};
use crate::ml;
use crate::models;

/// Window length fed to the NVIDIA models, in seconds.
///
/// The models are trained on long utterances; a shorter window would throw
/// away context, a longer one would dilute it.
pub const NVIDIA_WINDOW_SECONDS: f64 = 45.0;

/// Window length fed to the Qwen model, in seconds.
///
/// The reference implementation feeds every model 45 s windows, Qwen included,
/// even though the model itself would accept much longer chunks — matching the
/// reference keeps the segment boundaries identical.
pub const QWEN_WINDOW_SECONDS: f64 = 45.0;

/// The language the Parakeet checkpoint reports.
///
/// The reference implementation hard-codes English for this checkpoint: the
/// model is English-only, so the value describes the model, not the audio.
pub const PARAKEET_LANGUAGE: &str = "en";

/// The language the Nemotron checkpoint reports.
///
/// The reference implementation always runs the multilingual checkpoint with
/// `language="auto"` and reports the mode it used, not the language it
/// detected. The routing stage has already picked the model, so the language
/// hint is deliberately ignored — see ADR 11 ("Report the language the way
/// the reference does").
pub const NEMOTRON_LANGUAGE: &str = "auto";

/// Outcome of an ASR run.
#[derive(Debug, Clone, Default)]
pub struct AsrResult {
    /// The full transcript.
    pub text: String,
    /// Detected or forced language code.
    pub language: String,
    /// Timed segments.
    pub segments: Vec<Segment>,
}

/// Which model serves a language.
pub fn model_for_language(code: &str) -> &'static str {
    match code {
        "ru" => models::MODEL_QWEN,
        "en" => models::MODEL_PARAKEET,
        _ => models::MODEL_NEMOTRON,
    }
}

/// Split `samples` into fixed windows, mirroring the reference runner.
pub fn iter_windows(samples: &[f32], window_s: f64) -> Vec<(f64, f64, Vec<f32>)> {
    let step = ((window_s * 16_000.0).round() as usize).max(1);
    let total = samples.len();
    if total == 0 {
        return vec![(0.0, 0.0, vec![0.0; step])];
    }
    let mut starts: Vec<usize> = Vec::new();
    let limit = total.saturating_sub(step).saturating_add(1).max(1);
    let mut start = 0usize;
    while start < limit {
        starts.push(start);
        start += step;
    }
    if total > step
        && let Some(last) = starts.last().copied()
        && last + step < total
    {
        starts.push(last + step);
    }
    starts
        .into_iter()
        .map(|start| {
            let end = (start + step).min(total);
            let mut chunk = samples[start..end].to_vec();
            if chunk.len() < step {
                chunk.resize(step, 0.0);
            }
            (
                start as f64 / 16_000.0,
                (start + chunk.len()) as f64 / 16_000.0,
                chunk,
            )
        })
        .collect()
}

/// Transcribe `wav_path` with the model registered under `model_key`.
pub fn transcribe(model_key: &str, wav_path: &Path, language: &str) -> Result<AsrResult> {
    let samples = crate::audio::read_samples(wav_path)?;
    match model_key {
        models::MODEL_PARAKEET | models::MODEL_PARAKEET_INT8 => {
            transcribe_parakeet(&samples, language, model_key)
        }
        models::MODEL_QWEN | models::MODEL_QWEN_INT4 => {
            transcribe_qwen(&samples, language, model_key)
        }
        models::MODEL_NEMOTRON => transcribe_nemotron(&samples, language),
        other => Err(Error::Model(format!("Unknown ASR model {other:?}"))),
    }
}

/// Transcribe with the Parakeet checkpoint.
///
/// The language hint is ignored on purpose: the checkpoint is English-only
/// and the reference implementation reports `en` for it, whatever the audio
/// contains.
///
/// `model_key` selects the export: the fp32 one (`parakeet-tdt-0.6b-v3`) or
/// the quantised one (`parakeet-tdt-0.6b-v3-int8`).
fn transcribe_parakeet(samples: &[f32], _language: &str, model_key: &str) -> Result<AsrResult> {
    let dir = models::get_model(model_key)?.local_dir();
    let mut model = ml::parakeet::Parakeet::load(&dir)?;
    let mut texts: Vec<String> = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();

    for (start, end, chunk) in iter_windows(samples, NVIDIA_WINDOW_SECONDS) {
        let result = model.transcribe(&chunk)?;
        let text = result.text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        texts.push(text.clone());
        segments.push(Segment {
            start: (start * 1000.0).round() / 1000.0,
            end: (end * 1000.0).round() / 1000.0,
            text,
        });
    }

    Ok(AsrResult {
        text: texts.join(" ").trim().to_string(),
        language: PARAKEET_LANGUAGE.to_string(),
        segments,
    })
}

/// Transcribe with the Qwen3-ASR checkpoint.
///
/// `model_key` selects the export: the fp32 one (`qwen3-asr-0.6b`) or the
/// quantised one (`qwen3-asr-0.6b-int4`). Both hold the same encoder and
/// the same tokenizer; only the decoder differs.
fn transcribe_qwen(samples: &[f32], language: &str, model_key: &str) -> Result<AsrResult> {
    let dir = models::get_model(model_key)?.local_dir();
    let mut model = ml::qwen::Qwen::load(&dir)?;
    // The model conditions on a full language *name*; codes it does not know
    // fall back to automatic detection, exactly like the reference runner.
    model.set_language(ml::qwen::language_name(language));
    // The code the reference runner falls back to when the model reports no
    // language of its own.
    let forced_code = ml::qwen::language_code_for(language);
    let mut texts: Vec<String> = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();
    let mut detected: Option<String> = None;

    for (start, end, chunk) in iter_windows(samples, QWEN_WINDOW_SECONDS) {
        let result = model.transcribe(&chunk)?;
        if let Some(name) = result.language {
            detected = Some(name);
        }
        let text = result.text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        texts.push(text.clone());
        segments.push(Segment {
            start: (start * 1000.0).round() / 1000.0,
            end: (end * 1000.0).round() / 1000.0,
            text,
        });
    }

    Ok(AsrResult {
        text: texts.join(" ").trim().to_string(),
        language: detected_language(detected, forced_code),
        segments,
    })
}

/// Turn a detected language name into the code the reference reports.
///
/// The reference runner normalises the name the model reports: the twelve
/// languages the checkpoint knows become their code, every other name is
/// lower-cased and kept as it is. When the model reports nothing at all, the
/// forced language is used — and `undetermined` if even that is missing.
fn detected_language(name: Option<String>, forced: Option<&str>) -> String {
    match name {
        Some(name) => match ml::qwen::language_code(&name) {
            Some(code) => code.to_string(),
            None => name.trim().to_lowercase(),
        },
        None => forced.unwrap_or("undetermined").to_string(),
    }
}

/// Transcribe with the Nemotron checkpoint.
///
/// The language hint is ignored on purpose: the reference implementation
/// always runs this checkpoint with `language="auto"`, so the model picks
/// the language itself and the artefacts report `auto`.
fn transcribe_nemotron(samples: &[f32], _language: &str) -> Result<AsrResult> {
    let dir = models::get_model(models::MODEL_NEMOTRON)?.local_dir();
    let mut model = ml::nemotron::Nemotron::load(&dir)?;
    // The model conditions on a language *prompt* rather than a language
    // string; the reference always asks for automatic detection.
    model.set_language(ml::nemotron::PROMPT_AUTO);
    let mut texts: Vec<String> = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();

    for (start, end, chunk) in iter_windows(samples, NVIDIA_WINDOW_SECONDS) {
        let result = model.transcribe(&chunk)?;
        let text = result.text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        texts.push(text.clone());
        segments.push(Segment {
            start: (start * 1000.0).round() / 1000.0,
            end: (end * 1000.0).round() / 1000.0,
            text,
        });
    }

    Ok(AsrResult {
        text: texts.join(" ").trim().to_string(),
        language: NEMOTRON_LANGUAGE.to_string(),
        segments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_languages_to_models() {
        assert_eq!(model_for_language("ru"), models::MODEL_QWEN);
        assert_eq!(model_for_language("en"), models::MODEL_PARAKEET);
        assert_eq!(model_for_language("de"), models::MODEL_NEMOTRON);
        assert_eq!(model_for_language("und"), models::MODEL_NEMOTRON);
    }

    #[test]
    fn reports_the_language_the_reference_reports() {
        assert_eq!(PARAKEET_LANGUAGE, "en");
        assert_eq!(NEMOTRON_LANGUAGE, "auto");
    }

    #[test]
    fn normalises_the_detected_language_like_the_reference() {
        // The twelve languages the checkpoint knows become their code.
        assert_eq!(detected_language(Some("Russian".to_string()), None), "ru");
        assert_eq!(detected_language(Some("english".to_string()), None), "en");
        // Every other name is lower-cased and kept as it is.
        assert_eq!(
            detected_language(Some("Ukrainian".to_string()), None),
            "ukrainian"
        );
        // No announcement from the model: the forced language wins, and
        // without one the language is undetermined.
        assert_eq!(detected_language(None, Some("ru")), "ru");
        assert_eq!(detected_language(None, None), "undetermined");
    }

    #[test]
    fn windows_tile_short_audio() {
        let samples = vec![0.0_f32; 16_000];
        let windows = iter_windows(&samples, 45.0);
        assert_eq!(windows.len(), 1);
        assert!((windows[0].0).abs() < 1e-9);
        assert!((windows[0].1 - 45.0).abs() < 1e-9);
        assert_eq!(windows[0].2.len(), 45 * 16_000);
    }

    #[test]
    fn windows_tile_long_audio() {
        let samples = vec![0.0_f32; 100 * 16_000];
        let windows = iter_windows(&samples, 45.0);
        // 0, 45, 90 and the tail window starting at 90 (100 s of audio).
        assert_eq!(windows.len(), 3);
        assert!((windows[1].0 - 45.0).abs() < 1e-9);
        assert!((windows[2].0 - 90.0).abs() < 1e-9);
    }
}
