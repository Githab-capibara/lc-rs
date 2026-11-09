//! Qwen3-ASR 0.6B — ONNX Runtime inference.
//!
//! The checkpoint is an audio encoder plus an autoregressive language model:
//!
//! | Graph | Purpose |
//! |-------|---------|
//! | `encoder.onnx` | 128-band log-mel features → 1024-dimensional audio embeddings |
//! | `decoder_init.onnx` | prompt + audio embeddings → first token and the KV cache |
//! | `decoder_step.onnx` | one token + the KV cache → next token and the updated cache |
//!
//! The quantised export has the same shape with `decoder_init.int4.onnx`
//! and `decoder_step.int4.onnx`; both layouts are loaded by the same code
//! path. The registry ships the fp32 one — see
//! [`crate::models`] and ADR 10.
//!
//! The front-end is the `transformers` `WhisperFeatureExtractor`: 30 s of audio
//! at 16 kHz, 128 mel bands, `log10` with a dynamic-range clamp. The prompt
//! follows the reference implementation:
//!
//! ```text
//! <|im_start|>user\n<|audio_start|><|audio_pad|>…<|audio_end|><|im_end|>\n<|im_start|>assistant\n
//! ```
//!
//! With a forced language the assistant turn continues with
//! `language {Name}<asr_text>`, otherwise the model emits the language itself
//! in the same shape and the transcript follows the tag.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::errors::{Error, Result};
use crate::ml;
use crate::ml::features::{MelConfig, MelFeatures, log_mel_spectrogram};

/// Samples the Whisper front-end consumes: 30 s at 16 kHz.
const N_SAMPLES: usize = 480_000;

/// Mel frames the Whisper front-end produces: 30 s at 10 ms per frame.
const NB_MAX_FRAMES: usize = 3_000;

/// Mel bands of the Whisper front-end.
const NUM_MELS: usize = 128;

/// Maximum number of tokens generated for one window.
const MAX_NEW_TOKENS: usize = 1024;

/// Tag that separates the metadata from the transcript.
const ASR_TEXT_TAG: &str = "<asr_text>";

/// Prefix of the metadata line.
const LANGUAGE_PREFIX: &str = "language ";

/// Result of one Qwen decoding pass.
#[derive(Debug, Clone, Default)]
pub struct QwenResult {
    /// The recognised text.
    pub text: String,
    /// The detected language, in the canonical spelling (`"Russian"`).
    pub language: Option<String>,
}

/// The `config.json` of the checkpoint.
#[derive(Debug, Deserialize)]
struct QwenConfig {
    encoder: EncoderConfig,
    decoder: DecoderConfig,
    mel: MelConfigJson,
    special_tokens: SpecialTokens,
}

/// Encoder geometry.
#[derive(Debug, Deserialize)]
struct EncoderConfig {
    num_mel_bins: usize,
    output_dim: usize,
}

/// Decoder geometry.
#[derive(Debug, Deserialize)]
struct DecoderConfig {
    num_layers: usize,
    hidden_size: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
}

/// Mel geometry.
#[derive(Debug, Deserialize)]
struct MelConfigJson {
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
}

/// Special token ids.
#[derive(Debug, Deserialize)]
struct SpecialTokens {
    eos_token_ids: Vec<i64>,
    audio_pad_token_id: i64,
}

/// A loaded Qwen3-ASR checkpoint.
pub struct Qwen {
    encoder: Session,
    decoder_init: Session,
    decoder_step: Session,
    tokenizer: Tokenizer,
    embeddings: Embeddings,
    config: QwenConfig,
    mel: MelConfig,
    /// Forced language, in the canonical spelling (`"Russian"`).
    language: Option<String>,
}

impl std::fmt::Debug for Qwen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen")
            .field("vocab_size", &self.config.decoder.vocab_size)
            .field("language", &self.language)
            .finish_non_exhaustive()
    }
}

/// Locate one decoder graph in a checkpoint directory.
///
/// The registry ships the fp32 decoder — `decoder_init.onnx` next to
/// `decoder_weights.data` — because it is the counterpart of what the
/// reference implementation computes. The same model also exists as a
/// quantised export, whose graphs are named `decoder_init.int4.onnx`.
/// Both are accepted, so that a quantised export can be measured against
/// the fp32 one without touching the registry.
///
/// The quantised graph is preferred when both are present, because a
/// directory that holds it was deliberately prepared for it.
fn decoder_graph(dir: &Path, stem: &str) -> PathBuf {
    let quantised = dir.join(format!("{stem}.int4.onnx"));
    if quantised.is_file() {
        return quantised;
    }
    dir.join(format!("{stem}.onnx"))
}

impl Qwen {
    /// Load the three graphs, the tokenizer and the embedding table.
    pub fn load(dir: &Path) -> Result<Self> {
        let config: QwenConfig = read_json(&dir.join("config.json"))?;
        if config.encoder.num_mel_bins != config.mel.n_mels {
            return Err(Error::Model(format!(
                "the Qwen checkpoint disagrees about the mel bands: encoder says {}, mel says {}",
                config.encoder.num_mel_bins, config.mel.n_mels
            )));
        }
        let encoder = ml::load_session(dir.join("encoder.onnx"))?;
        let decoder_init = ml::load_session(decoder_graph(dir, "decoder_init"))?;
        let decoder_step = ml::load_session(decoder_graph(dir, "decoder_step"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::Model(format!("could not load the Qwen tokenizer: {e}")))?;
        let embeddings = Embeddings::load(
            &dir.join("embed_tokens.bin"),
            config.decoder.vocab_size,
            config.decoder.hidden_size,
        )?;
        let mel = MelConfig {
            n_fft: config.mel.n_fft,
            hop_length: config.mel.hop_length,
            win_length: config.mel.n_fft,
            n_mels: config.mel.n_mels,
            fmin: config.mel.fmin,
            fmax: config.mel.fmax,
            sample_rate: config.mel.sample_rate,
            mel_scale: crate::ml::features::MelScale::Slaney,
            norm: crate::ml::features::MelNorm::Slaney,
            center: true,
            preemphasis: 0.0,
            dither: 0.0,
            natural_log: false,
            log_floor: 1e-10,
            log_range: Some(8.0),
            log_offset: 4.0,
            log_scale: 4.0,
        };
        Ok(Self {
            encoder,
            decoder_init,
            decoder_step,
            tokenizer,
            embeddings,
            config,
            mel,
            language: None,
        })
    }

    /// Force a language (`"Russian"`) or let the model detect it (`None`).
    pub fn set_language(&mut self, language: Option<&str>) {
        self.language = language.map(str::to_string);
    }

    /// Transcribe mono 16 kHz samples.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<QwenResult> {
        let features = self.compute_features(samples)?;
        let (audio, audio_len) = self.encode(&features)?;
        let prompt = self.build_prompt(audio_len)?;
        let (logits, keys, values) = self.prefill(&prompt, &audio, audio_len)?;

        let mut generated: Vec<u32> = Vec::new();
        let mut next = argmax(&logits);
        let mut position = prompt.len() as i64;
        let mut past_len = prompt.len();
        let (mut keys, mut values) = (keys, values);

        while generated.len() < MAX_NEW_TOKENS {
            if self.is_eos(next) {
                break;
            }
            generated.push(next as u32);
            let (logits, new_keys, new_values) =
                self.step(next, position, &keys, &values, past_len)?;
            keys = new_keys;
            values = new_values;
            past_len += 1;
            position += 1;
            next = argmax(&logits);
        }

        self.parse_output(&generated)
    }

    /// Run the Whisper front-end.
    ///
    /// The reference extractor truncates the waveform to 30 s, pads it with
    /// zeros and keeps the first 3000 mel frames.
    fn compute_features(&self, samples: &[f32]) -> Result<MelFeatures> {
        let mut padded = samples.to_vec();
        padded.truncate(N_SAMPLES);
        padded.resize(N_SAMPLES, 0.0);

        let features = log_mel_spectrogram(&padded, &self.mel)?;
        if features.n_frames < NB_MAX_FRAMES {
            return Err(Error::Transcription(format!(
                "the Qwen front-end produced {} frames, expected at least {NB_MAX_FRAMES}",
                features.n_frames
            )));
        }
        let mut data = vec![0.0_f32; NUM_MELS * NB_MAX_FRAMES];
        for band in 0..NUM_MELS {
            for frame in 0..NB_MAX_FRAMES {
                data[band * NB_MAX_FRAMES + frame] = features.at(band, frame);
            }
        }
        Ok(MelFeatures {
            n_mels: NUM_MELS,
            n_frames: NB_MAX_FRAMES,
            data,
        })
    }

    /// Run the audio encoder.
    fn encode(&mut self, features: &MelFeatures) -> Result<(Vec<f32>, usize)> {
        let mel = Tensor::from_array((
            vec![1_i64, NUM_MELS as i64, NB_MAX_FRAMES as i64],
            features.data.clone(),
        ))
        .map_err(|e| ml::onnx_error("could not build the Qwen mel tensor", &e))?;
        let outputs = self
            .encoder
            .run(ort::inputs!["mel" => mel])
            .map_err(|e| ml::onnx_error("the Qwen encoder failed", &e))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Qwen encoder output", &e))?;
        let audio_len = shape.get(1).copied().unwrap_or(0).max(0) as usize;
        Ok((data.to_vec(), audio_len))
    }

    /// Build the prompt token ids.
    fn build_prompt(&self, audio_len: usize) -> Result<Vec<i64>> {
        let mut text = String::from("<|im_start|>user\n<|audio_start|>");
        for _ in 0..audio_len {
            text.push_str("<|audio_pad|>");
        }
        text.push_str("<|audio_end|><|im_end|>\n<|im_start|>assistant\n");
        if let Some(language) = &self.language {
            text.push_str(LANGUAGE_PREFIX);
            text.push_str(language);
            text.push_str(ASR_TEXT_TAG);
        }
        let encoding = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| Error::Model(format!("could not encode the Qwen prompt: {e}")))?;
        Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    /// Run the decoder over the prompt and return the first-token logits.
    fn prefill(
        &mut self,
        prompt: &[i64],
        audio: &[f32],
        audio_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let seq = prompt.len() as i64;
        let input_ids = Tensor::from_array((vec![1_i64, seq], prompt.to_vec()))
            .map_err(|e| ml::onnx_error("could not build the Qwen input-ids tensor", &e))?;
        let position_ids =
            Tensor::from_array((vec![1_i64, seq], (0..seq).collect::<Vec<i64>>()))
                .map_err(|e| ml::onnx_error("could not build the Qwen position-ids tensor", &e))?;
        let audio_features = Tensor::from_array((
            vec![
                1_i64,
                audio_len as i64,
                self.config.encoder.output_dim as i64,
            ],
            audio.to_vec(),
        ))
        .map_err(|e| ml::onnx_error("could not build the Qwen audio tensor", &e))?;
        let offset = prompt
            .iter()
            .position(|&id| id == self.config.special_tokens.audio_pad_token_id)
            .unwrap_or(0);
        let audio_offset = Tensor::from_array((vec![1_i64], vec![offset as i64]))
            .map_err(|e| ml::onnx_error("could not build the Qwen audio-offset tensor", &e))?;

        let outputs = self
            .decoder_init
            .run(ort::inputs![
                "input_ids" => input_ids,
                "position_ids" => position_ids,
                "audio_features" => audio_features,
                "audio_offset" => audio_offset,
            ])
            .map_err(|e| ml::onnx_error("the Qwen decoder prefill failed", &e))?;

        let (_, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Qwen prefill logits", &e))?;
        let vocab = self.config.decoder.vocab_size;
        let last = logits[(prompt.len() - 1) * vocab..prompt.len() * vocab].to_vec();
        let keys = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Qwen key cache", &e))?
            .1
            .to_vec();
        let values = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Qwen value cache", &e))?
            .1
            .to_vec();
        Ok((last, keys, values))
    }

    /// Run one decoder step.
    fn step(
        &mut self,
        token: i64,
        position: i64,
        keys: &[f32],
        values: &[f32],
        past_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let embedding = self
            .embeddings
            .row(token.max(0) as usize)
            .ok_or_else(|| Error::Model(format!("token {token} is outside the vocabulary")))?;
        let input_embeds = Tensor::from_array((
            vec![1_i64, 1_i64, self.config.decoder.hidden_size as i64],
            embedding,
        ))
        .map_err(|e| ml::onnx_error("could not build the Qwen embedding tensor", &e))?;
        let position_ids = Tensor::from_array((vec![1_i64, 1_i64], vec![position]))
            .map_err(|e| ml::onnx_error("could not build the Qwen position tensor", &e))?;
        let shape = vec![
            self.config.decoder.num_layers as i64,
            1_i64,
            self.config.decoder.num_key_value_heads as i64,
            past_len as i64,
            self.config.decoder.head_dim as i64,
        ];
        let past_keys = Tensor::from_array((shape.clone(), keys.to_vec()))
            .map_err(|e| ml::onnx_error("could not build the Qwen key-cache tensor", &e))?;
        let past_values = Tensor::from_array((shape, values.to_vec()))
            .map_err(|e| ml::onnx_error("could not build the Qwen value-cache tensor", &e))?;

        let outputs = self
            .decoder_step
            .run(ort::inputs![
                "input_embeds" => input_embeds,
                "position_ids" => position_ids,
                "past_keys" => past_keys,
                "past_values" => past_values,
            ])
            .map_err(|e| ml::onnx_error("the Qwen decoder step failed", &e))?;

        let (_, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Qwen step logits", &e))?;
        let new_keys = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Qwen key cache", &e))?
            .1
            .to_vec();
        let new_values = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Qwen value cache", &e))?
            .1
            .to_vec();
        Ok((logits.to_vec(), new_keys, new_values))
    }

    /// Whether `token` ends the generation.
    fn is_eos(&self, token: i64) -> bool {
        self.config.special_tokens.eos_token_ids.contains(&token)
    }

    /// Turn the generated ids into text and a language.
    fn parse_output(&self, generated: &[u32]) -> Result<QwenResult> {
        let raw = self
            .tokenizer
            .decode(generated, true)
            .map_err(|e| Error::Model(format!("could not decode the Qwen output: {e}")))?;
        let repaired = repair_repetitions(raw.trim());

        if let Some(language) = &self.language {
            return Ok(QwenResult {
                text: repaired.trim().to_string(),
                language: Some(language.clone()),
            });
        }

        match repaired.split_once(ASR_TEXT_TAG) {
            Some((meta, text)) => Ok(QwenResult {
                text: text.trim().to_string(),
                language: parse_language(meta),
            }),
            None => Ok(QwenResult {
                text: repaired.trim().to_string(),
                language: None,
            }),
        }
    }
}

/// The embedding table of the decoder, stored as raw little-endian `float16`.
struct Embeddings {
    data: Vec<u16>,
    rows: usize,
    dim: usize,
}

impl Embeddings {
    /// Read `embed_tokens.bin`.
    fn load(path: &Path, rows: usize, dim: usize) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        let expected = rows * dim * 2;
        if bytes.len() != expected {
            return Err(Error::Model(format!(
                "{} has {} bytes, expected {expected}",
                path.display(),
                bytes.len()
            )));
        }
        let (pairs, remainder) = bytes.as_slice().as_chunks::<2>();
        if !remainder.is_empty() {
            return Err(Error::Model(format!(
                "{} has an odd byte count",
                path.display()
            )));
        }
        let data = pairs
            .iter()
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        Ok(Self { data, rows, dim })
    }

    /// The embedding of one token.
    fn row(&self, id: usize) -> Option<Vec<f32>> {
        if id >= self.rows {
            return None;
        }
        let start = id * self.dim;
        Some(
            self.data[start..start + self.dim]
                .iter()
                .map(|&value| half_to_f32(value))
                .collect(),
        )
    }
}

/// Convert an IEEE-754 half-precision value into `f32`.
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    if exponent == 0 {
        // Subnormal: the value is `mantissa * 2^-24`.
        f32::from_bits(sign) + (mantissa as f32) * 2.0_f32.powi(-24)
    } else if exponent == 0x1f {
        // Infinity or NaN.
        f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13))
    } else {
        let unbiased = exponent + 127 - 15;
        f32::from_bits(sign | (unbiased << 23) | (mantissa << 13))
    }
}
/// Index of the largest value.
fn argmax(values: &[f32]) -> i64 {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as i64
}

/// Read a JSON file into a typed value.
fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    serde_json::from_str(&text)
        .map_err(|e| Error::Model(format!("could not parse {}: {e}", path.display())))
}

/// Extract the language from the metadata part of the model's output.
fn parse_language(meta: &str) -> Option<String> {
    for line in meta.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.to_lowercase().starts_with(LANGUAGE_PREFIX) {
            let value = line[LANGUAGE_PREFIX.len()..].trim();
            if !value.is_empty() {
                return Some(normalize_language_name(value));
            }
            break;
        }
    }
    None
}

/// Capitalise the first letter and lower-case the rest.
fn normalize_language_name(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
        None => String::new(),
    }
}

/// Collapse the degenerate repetitions the model sometimes falls into.
///
/// The reference implementation collapses a run of more than 20 identical
/// characters and any pattern that repeats at least 20 times.
fn repair_repetitions(text: &str) -> String {
    let collapsed = collapse_character_runs(text, 20);
    collapse_pattern_runs(&collapsed, 20)
}

/// Collapse runs of identical characters longer than `threshold`.
fn collapse_character_runs(text: &str, threshold: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(chars.len());
    let mut index = 0usize;
    while index < chars.len() {
        let mut count = 1usize;
        while index + count < chars.len() && chars[index + count] == chars[index] {
            count += 1;
        }
        if count > threshold {
            out.push(chars[index]);
        } else {
            out.extend(chars[index..index + count].iter());
        }
        index += count;
    }
    out
}

/// Collapse a pattern that repeats at least `threshold` times.
fn collapse_pattern_runs(text: &str, threshold: usize) -> String {
    let minimum = threshold * 2;
    if text.len() < minimum {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let max_len = 20usize;
    let mut index = 0usize;
    while index + minimum <= chars.len() {
        for len in 1..=max_len {
            if index + len * threshold > chars.len() {
                break;
            }
            let pattern: Vec<char> = chars[index..index + len].to_vec();
            let mut repeats = true;
            for repetition in 1..threshold {
                let start = index + repetition * len;
                if chars[start..start + len] != pattern[..] {
                    repeats = false;
                    break;
                }
            }
            if repeats {
                let mut end = index + threshold * len;
                while end + len <= chars.len() && chars[end..end + len] == pattern[..] {
                    end += len;
                }
                let tail: String = chars[end..].iter().collect();
                let mut result: String = pattern.iter().collect();
                result.push_str(&collapse_pattern_runs(&tail, threshold));
                return result;
            }
        }
        index += 1;
    }
    text.to_string()
}

/// Canonical language names accepted by the model prompt.
pub const LANGUAGE_NAMES: [(&str, &str); 12] = [
    ("ru", "Russian"),
    ("en", "English"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("pt", "Portuguese"),
    ("tr", "Turkish"),
    ("ar", "Arabic"),
];

/// The canonical name of a language code, if the model accepts it.
pub fn language_name(code: &str) -> Option<&'static str> {
    LANGUAGE_NAMES
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(code))
        .map(|(_, name)| *name)
}

/// The language code of a canonical language name, if it is known.
pub fn language_code(name: &str) -> Option<&'static str> {
    let lower = name.trim().to_lowercase();
    LANGUAGE_NAMES
        .iter()
        .find(|(_, canonical)| canonical.to_lowercase() == lower)
        .map(|(code, _)| *code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_half_precision() {
        assert_eq!(half_to_f32(0x0000).to_bits(), 0.0_f32.to_bits());
        assert_eq!(half_to_f32(0x3c00).to_bits(), 1.0_f32.to_bits());
        assert_eq!(half_to_f32(0xc000).to_bits(), (-2.0_f32).to_bits());
        assert_eq!(half_to_f32(0x3800).to_bits(), 0.5_f32.to_bits());
        assert!((half_to_f32(0x3555) - 0.333_251_95).abs() < 1e-6);
        assert!((half_to_f32(0x0400) - 6.103_515_6e-5).abs() < 1e-9);
    }

    #[test]
    fn parses_the_language_from_the_metadata() {
        assert_eq!(
            parse_language("language Russian").as_deref(),
            Some("Russian")
        );
        assert_eq!(
            parse_language("language russian\n").as_deref(),
            Some("Russian")
        );
        assert_eq!(parse_language("Language German").as_deref(), Some("German"));
        assert_eq!(parse_language("no metadata"), None);
        assert_eq!(parse_language("language "), None);
    }

    #[test]
    fn repairs_degenerate_repetitions() {
        assert_eq!(collapse_character_runs(&"a".repeat(25), 20), "a");
        assert_eq!(collapse_character_runs(&"a".repeat(10), 20), "a".repeat(10));
        let repeated = "ab".repeat(25);
        assert_eq!(collapse_pattern_runs(&repeated, 20), "ab");
    }

    #[test]
    fn maps_languages_both_ways() {
        assert_eq!(language_name("ru"), Some("Russian"));
        assert_eq!(language_name("RU"), Some("Russian"));
        assert_eq!(language_name("xx"), None);
        assert_eq!(language_code("Russian"), Some("ru"));
        assert_eq!(language_code("russian"), Some("ru"));
        assert_eq!(language_code("Klingon"), None);
    }

    #[test]
    fn finds_the_argmax() {
        assert_eq!(argmax(&[1.0, 5.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0]), 0);
    }

    /// The quantised decoder is preferred when the directory holds it.
    #[test]
    fn the_quantised_decoder_graph_is_preferred() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let stem = dir.path().join("decoder_init");
        assert_eq!(
            decoder_graph(dir.path(), "decoder_init"),
            stem.with_extension("onnx")
        );

        std::fs::write(stem.with_extension("int4.onnx"), b"x").expect("a graph");
        assert_eq!(
            decoder_graph(dir.path(), "decoder_init"),
            stem.with_extension("int4.onnx")
        );
    }

    /// A directory without the quantised graphs falls back to the fp32 ones.
    #[test]
    fn the_fp32_decoder_graph_is_the_fallback() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        assert_eq!(
            decoder_graph(dir.path(), "decoder_step"),
            dir.path().join("decoder_step.onnx")
        );
    }
}
