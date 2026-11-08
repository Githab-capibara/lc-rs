//! Language identification with the ECAPA-TDNN classifier.
//!
//! The graph shipped in `assets/langid/langid_inline.onnx` is a
//! self-contained export of `speechbrain/lang-id-commonlanguage_ecapa`: it
//! takes raw waveform samples and returns the 45-way posterior. Because the
//! feature extraction is baked into the graph, the Rust side only has to
//! window the audio.
//!
//! The windowing, posterior averaging and vote tally replicate
//! `src/localcaption/runners/langid.py` exactly, so the routing decision is
//! identical to the Python implementation.

use std::collections::BTreeMap;
use std::path::Path;

use ort::value::Tensor;
use serde::Serialize;

use crate::errors::{Error, Result};
use crate::languages;
use crate::ml::{self, wav};

/// The 45 classes of the language-identifier, in graph order.
pub const LABELS: [&str; 45] = [
    "Basque",
    "Romansh_Sursilvan",
    "Sakha",
    "Georgian",
    "Greek",
    "Hakha_Chin",
    "Ukrainian",
    "Interlingua",
    "Persian",
    "Polish",
    "Dutch",
    "Chinese_Hongkong",
    "Japanese",
    "Portuguese",
    "Italian",
    "Catalan",
    "Chuvash",
    "Swedish",
    "Spanish",
    "Slovenian",
    "Tamil",
    "Breton",
    "Russian",
    "Czech",
    "English",
    "French",
    "Tatar",
    "Welsh",
    "Kyrgyz",
    "Esperanto",
    "Kinyarwanda",
    "Dhivehi",
    "Turkish",
    "Latvian",
    "Estonian",
    "Arabic",
    "Frisian",
    "Mongolian",
    "Chinese_Taiwan",
    "Indonesian",
    "Maltese",
    "Kabyle",
    "Romanian",
    "Chinese_China",
    "German",
];

/// Below this max posterior the language counts as undetermined and the
/// multilingual model takes over.
pub const LANGID_CONFIDENCE_THRESHOLD: f32 = 0.6;

/// When at least this fraction of the sampled windows vote for the top
/// class, the language counts as determined even if the averaged posterior
/// was diluted by a few noisy windows.
pub const LANGID_VOTE_THRESHOLD: f32 = 0.5;

/// Detector window in seconds (the model is trained on short utterances).
pub const LANGID_WINDOW_SECONDS: f32 = 6.0;

/// Upper bound on the number of windows sampled across a file.
pub const LANGID_MAX_WINDOWS: usize = 16;

/// Sample rate the classifier expects.
const SAMPLE_RATE: usize = 16_000;

/// The embedded ONNX graph.
pub fn embedded_graph() -> &'static [u8] {
    include_bytes!("../../assets/langid/langid_inline.onnx")
}

/// Outcome of a language-identification run.
#[derive(Debug, Clone, Serialize)]
pub struct LangIdResult {
    /// ISO-ish language code.
    pub code: String,
    /// Display name of the detected language.
    pub name: String,
    /// Averaged posterior of the winning class.
    pub score: f32,
    /// The five best classes and their posteriors.
    pub scores: BTreeMap<String, f32>,
    /// How many windows voted for each class.
    pub votes: BTreeMap<String, u32>,
}

impl LangIdResult {
    /// Fraction of the sampled windows that voted for the top class.
    pub fn vote_fraction(&self) -> f32 {
        let total: u32 = self.votes.values().sum();
        if total == 0 {
            return 0.0;
        }
        let best = self.votes.values().copied().max().unwrap_or(0);
        best as f32 / total as f32
    }

    /// Whether the detected language is confident enough to route on.
    pub fn confident(&self) -> bool {
        if self.score >= LANGID_CONFIDENCE_THRESHOLD {
            return true;
        }
        // A strict majority: a 50/50 split is not a decision.
        self.vote_fraction() > LANGID_VOTE_THRESHOLD
    }
}

/// A window of audio handed to the classifier.
#[derive(Debug, Clone)]
pub struct Window {
    /// Start of the window in seconds.
    pub start: f32,
    /// End of the window in seconds.
    pub end: f32,
    /// The samples themselves.
    pub samples: Vec<f32>,
}

/// Split `samples` into fixed windows.
///
/// Mirrors `localcaption.runners._wavio.iter_windows`: windows tile the
/// recording, the last one starts right after the previous one ends, and
/// when more windows than `max_windows` are available they are sampled
/// evenly across the recording.
pub fn iter_windows(samples: &[f32], window_s: f32, max_windows: Option<usize>) -> Vec<Window> {
    let step = ((window_s * SAMPLE_RATE as f32).round() as usize).max(1);
    let total = samples.len();
    let mut cuts = Vec::new();
    if total == 0 {
        cuts.push(Window {
            start: 0.0,
            end: 0.0,
            samples: vec![0.0; step],
        });
        return cuts;
    }

    let mut starts: Vec<usize> = Vec::new();
    let mut start = 0usize;
    while start < total.max(1) {
        if start + step > total && !starts.is_empty() {
            break;
        }
        starts.push(start);
        start += step;
    }
    if total > step
        && let Some(last) = starts.last().copied()
        && last + step < total
    {
        starts.push(last + step);
    }
    if let Some(max) = max_windows
        && max > 0
        && starts.len() > max
    {
        let factor = starts.len() as f64 / max as f64;
        let mut indices: Vec<usize> = (0..max)
            .map(|i| {
                let index = (i as f64 * factor) as usize;
                index.min(starts.len() - 1)
            })
            .collect();
        indices.sort_unstable();
        indices.dedup();
        starts = indices.into_iter().map(|i| starts[i]).collect();
    }

    for start in starts {
        let end = (start + step).min(total);
        let mut chunk = samples[start..end].to_vec();
        if chunk.len() < step {
            chunk.resize(step, 0.0);
        }
        cuts.push(Window {
            start: start as f32 / SAMPLE_RATE as f32,
            end: (start + chunk.len()) as f32 / SAMPLE_RATE as f32,
            samples: chunk,
        });
    }
    cuts
}

/// Classify the dominant language of a WAV file.
pub fn detect_language(wav_path: &Path) -> Result<LangIdResult> {
    let audio = wav::read_wav(wav_path)?;
    if audio.sample_rate != SAMPLE_RATE as u32 {
        return Err(Error::Transcription(format!(
            "language identification needs {SAMPLE_RATE} Hz audio, got {} Hz",
            audio.sample_rate
        )));
    }
    detect_language_in_samples(&audio.samples)
}

/// Classify the dominant language of raw samples.
pub fn detect_language_in_samples(samples: &[f32]) -> Result<LangIdResult> {
    let mut session = ml::load_session_from_memory(embedded_graph())?;
    let input_name = session
        .inputs()
        .first()
        .map_or_else(|| "wav".to_string(), |i| i.name().to_string());

    let windows = iter_windows(samples, LANGID_WINDOW_SECONDS, Some(LANGID_MAX_WINDOWS));
    if windows.is_empty() {
        return Err(Error::Transcription(
            "no audio windows classified".to_string(),
        ));
    }

    let mut sum: Vec<f32> = Vec::new();
    let mut votes: BTreeMap<String, u32> = BTreeMap::new();
    let mut count = 0usize;

    for window in &windows {
        let tensor = Tensor::from_array((
            vec![1_i64, window.samples.len() as i64],
            window.samples.clone(),
        ))
        .map_err(|e| ml::onnx_error("could not build the language-ID input", &e))?;
        let outputs = session
            .run(ort::inputs![input_name.as_str() => tensor])
            .map_err(|e| ml::onnx_error("language-ID inference failed", &e))?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the language-ID output", &e))?;
        if data.len() < LABELS.len() {
            return Err(Error::Transcription(format!(
                "language-ID model returned {} classes, expected {}",
                data.len(),
                LABELS.len()
            )));
        }
        if sum.is_empty() {
            sum = vec![0.0_f32; LABELS.len()];
        }
        for (index, value) in data.iter().take(LABELS.len()).enumerate() {
            sum[index] += value;
        }
        let best = argmax(&sum);
        if best < LABELS.len() {
            *votes.entry(LABELS[best].to_string()).or_insert(0) += 1;
        }
        count += 1;
    }

    if count == 0 {
        return Err(Error::Transcription(
            "no audio windows classified".to_string(),
        ));
    }
    for value in &mut sum {
        *value /= count as f32;
    }
    let best = argmax(&sum);
    let name = LABELS.get(best).copied().unwrap_or("Undetermined");

    // Rank the classes by posterior, keeping the index order for ties.
    let mut order: Vec<usize> = (0..sum.len()).collect();
    order.sort_by(|a, b| {
        sum[*b]
            .partial_cmp(&sum[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let mut scores = BTreeMap::new();
    for index in order.into_iter().take(5) {
        scores.insert(LABELS[index].to_string(), sum[index]);
    }

    // A clear majority of windows wins over a diluted averaged posterior.
    let mut result_name = name.to_string();
    let total_votes: u32 = votes.values().sum();
    if total_votes > 0
        && let Some((winner, count)) = votes.iter().max_by_key(|(_, count)| **count)
        && (*count as f32 / total_votes as f32) > LANGID_VOTE_THRESHOLD
    {
        result_name.clone_from(winner);
    }

    Ok(LangIdResult {
        code: languages::code_for_name(&result_name),
        name: result_name,
        score: sum[best],
        scores,
        votes,
    })
}

/// Index of the largest value.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_match_the_registry_order() {
        assert_eq!(LABELS.len(), 45);
        assert_eq!(LABELS[22], "Russian");
        assert_eq!(LABELS[24], "English");
        assert_eq!(LABELS[44], "German");
        // The display names must map back onto the routing table.
        assert_eq!(languages::code_for_name(LABELS[22]), "ru");
        assert_eq!(languages::code_for_name(LABELS[24]), "en");
    }

    /// The label order is the graph's output order; a typo in one name
    /// silently mis-routes that language. The authoritative list is the
    /// `label_encoder.txt` shipped with the classifier, so it is parsed and
    /// compared here.
    #[test]
    fn labels_match_the_shipped_label_encoder() {
        let raw = include_str!("../../assets/langid/label_encoder.txt");
        let mut parsed: Vec<(usize, &str)> = Vec::new();
        for line in raw.lines() {
            let Some((name, index)) = line.split_once("=>") else {
                continue;
            };
            let name = name.trim().trim_matches('\'');
            // `starting_index` is a bookkeeping entry, not a class.
            if name == "starting_index" {
                continue;
            }
            let index: usize = index
                .trim()
                .parse()
                .expect("the label encoder must contain integer indices");
            parsed.push((index, name));
        }
        parsed.sort_unstable_by_key(|(index, _)| *index);

        let expected: Vec<&str> = parsed.iter().map(|(_, name)| *name).collect();
        assert_eq!(expected.len(), LABELS.len());
        for (index, (expected_name, actual_name)) in expected.iter().zip(LABELS).enumerate() {
            assert_eq!(actual_name, *expected_name, "label {index} differs");
        }

        // Every label must resolve to a code through the routing table.
        for label in LABELS {
            assert_ne!(
                languages::code_for_name(label),
                "und",
                "{label:?} does not map onto a language code"
            );
        }
    }

    #[test]
    fn windows_tile_the_recording() {
        let samples = vec![0.0_f32; 16_000 * 10];
        let windows = iter_windows(&samples, 6.0, Some(16));
        assert_eq!(windows.len(), 2);
        assert!((windows[0].start).abs() < 1e-6);
        assert!((windows[0].end - 6.0).abs() < 1e-6);
        assert!((windows[1].start - 6.0).abs() < 1e-6);
    }

    #[test]
    fn windows_are_capped_and_sampled_evenly() {
        let samples = vec![0.0_f32; 16_000 * 600];
        let windows = iter_windows(&samples, 6.0, Some(16));
        assert_eq!(windows.len(), 16);
        // The samples are spread over the whole recording.
        assert!(windows[0].start < 1.0);
        assert!(windows[15].start > 500.0);
    }

    #[test]
    fn empty_audio_yields_one_padded_window() {
        let windows = iter_windows(&[], 6.0, Some(16));
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].samples.len(), 96_000);
    }

    #[test]
    fn confidence_uses_votes() {
        let mut votes = BTreeMap::new();
        votes.insert("Russian".to_string(), 3);
        votes.insert("English".to_string(), 1);
        let result = LangIdResult {
            code: "ru".into(),
            name: "Russian".into(),
            score: 0.4,
            scores: BTreeMap::new(),
            votes,
        };
        assert!(result.confident());
        assert!((result.vote_fraction() - 0.75).abs() < 1e-6);
    }
}
