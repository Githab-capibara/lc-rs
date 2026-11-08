//! NVIDIA Parakeet TDT 0.6B v3 — ONNX Runtime inference.
//!
//! Three graphs make up the checkpoint:
//!
//! | Graph | Purpose |
//! |-------|---------|
//! | `nemo128.onnx` | waveform → 128-band log-mel features (with `NeMo`'s utterance normalisation) |
//! | `encoder-model.int8.onnx` | features → encoder states |
//! | `decoder_joint-model.int8.onnx` | encoder frame + decoder state → token and duration logits |
//!
//! The decoding loop is the greedy TDT transducer search: at every encoder
//! frame the joint network predicts a token *and* a duration; the duration
//! says how many frames to skip before the next prediction.
//!
//! The graphs were exported from `NVIDIA` `NeMo`, so the semantics follow the
//! reference implementation shipped with `onnx-asr`.

use std::collections::HashMap;
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

use crate::errors::{Error, Result};
use crate::ml;

/// Duration classes the joint network can predict, in logit order.
pub const DURATIONS: [i64; 5] = [0, 1, 2, 3, 4];

/// Maximum number of tokens emitted on a single encoder frame.
pub const MAX_TOKENS_PER_STEP: usize = 10;

/// Frame rate of the encoder output, in milliseconds per frame.
pub const FRAME_STEP_SECONDS: f64 = 0.08;

/// One recognised token.
#[derive(Debug, Clone)]
pub struct Token {
    /// Vocabulary id.
    pub id: usize,
    /// Piece text (with the `SentencePiece` space marker already replaced).
    pub text: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
}

/// Result of one Parakeet decoding pass.
#[derive(Debug, Clone, Default)]
pub struct ParakeetResult {
    /// The recognised text.
    pub text: String,
    /// The recognised tokens with their timestamps.
    pub tokens: Vec<Token>,
}

/// A loaded Parakeet checkpoint.
pub struct Parakeet {
    feature: Session,
    encoder: Session,
    joint: Session,
    vocab: Vec<String>,
    blank_idx: usize,
    max_tokens_per_step: usize,
}

impl std::fmt::Debug for Parakeet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parakeet")
            .field("vocab_size", &self.vocab.len())
            .field("blank_idx", &self.blank_idx)
            .finish_non_exhaustive()
    }
}

impl Parakeet {
    /// Load the three graphs and the vocabulary from `dir`.
    pub fn load(dir: &Path) -> Result<Self> {
        let feature = ml::load_session(&dir.join("nemo128.onnx"))?;
        let encoder = ml::load_session(&dir.join("encoder-model.int8.onnx"))?;
        let joint = ml::load_session(&dir.join("decoder_joint-model.int8.onnx"))?;
        let vocab = read_vocab(&dir.join("vocab.txt"))?;
        let blank_idx = vocab
            .iter()
            .position(|token| token == "<blk>")
            .ok_or_else(|| Error::Model("vocab.txt has no <blk> entry".to_string()))?;
        Ok(Self {
            feature,
            encoder,
            joint,
            vocab,
            blank_idx,
            max_tokens_per_step: MAX_TOKENS_PER_STEP,
        })
    }

    /// Transcribe mono 16 kHz samples.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<ParakeetResult> {
        let features = self.compute_features(samples)?;
        let (encoder_out, encoder_len) = self.encode(&features)?;
        self.decode(&encoder_out, encoder_len)
    }

    /// Run the log-mel front-end.
    ///
    /// Returns the feature tensor (`[1, 128, frames]`), its shape and the
    /// number of valid frames.
    fn compute_features(&mut self, samples: &[f32]) -> Result<(Vec<f32>, Vec<i64>, i64)> {
        let n = samples.len() as i64;
        let waveforms = Tensor::from_array((vec![1_i64, n], samples.to_vec()))
            .map_err(|e| ml::onnx_error("could not build the waveform tensor", &e))?;
        let lens = Tensor::from_array((vec![1_i64], vec![n]))
            .map_err(|e| ml::onnx_error("could not build the waveform-length tensor", &e))?;
        let outputs = self
            .feature
            .run(ort::inputs!["waveforms" => waveforms, "waveforms_lens" => lens])
            .map_err(|e| ml::onnx_error("the Parakeet feature extractor failed", &e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Parakeet features", &e))?;
        let features_len = outputs[1]
            .try_extract_tensor::<i64>()
            .map_err(|e| ml::onnx_error("could not read the Parakeet feature lengths", &e))?
            .1
            .first()
            .copied()
            .unwrap_or(0);
        Ok((data.to_vec(), shape.to_vec(), features_len))
    }

    /// Run the Conformer encoder.
    ///
    /// Returns the encoder states in `[frames, hidden]` order together with
    /// the number of valid frames.
    fn encode(&mut self, features: &(Vec<f32>, Vec<i64>, i64)) -> Result<(Vec<f32>, usize)> {
        let (data, shape, features_len) = features;
        let audio = Tensor::from_array((shape.clone(), data.clone()))
            .map_err(|e| ml::onnx_error("could not build the feature tensor", &e))?;
        let length = Tensor::from_array((vec![1_i64], vec![*features_len]))
            .map_err(|e| ml::onnx_error("could not build the length tensor", &e))?;
        let outputs = self
            .encoder
            .run(ort::inputs!["audio_signal" => audio, "length" => length])
            .map_err(|e| ml::onnx_error("the Parakeet encoder failed", &e))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Parakeet encoder output", &e))?;
        let encoded_len = outputs[1]
            .try_extract_tensor::<i64>()
            .map_err(|e| ml::onnx_error("could not read the Parakeet encoded lengths", &e))?
            .1
            .first()
            .copied()
            .unwrap_or(0)
            .max(0) as usize;

        // The graph emits `[batch, hidden, frames]`; the decoder wants
        // `[frames, hidden]`.
        let hidden = shape.get(1).copied().unwrap_or(0).max(0) as usize;
        let frames = shape.get(2).copied().unwrap_or(0).max(0) as usize;
        let mut transposed = vec![0.0_f32; frames * hidden];
        for frame in 0..frames {
            for channel in 0..hidden {
                let index = channel * frames + frame;
                transposed[frame * hidden + channel] = data[index];
            }
        }
        Ok((transposed, encoded_len.min(frames)))
    }

    /// Greedy TDT transducer search.
    fn decode(&mut self, encoder_out: &[f32], encoder_len: usize) -> Result<ParakeetResult> {
        let hidden = 1024usize;
        let vocab_size = self.vocab.len();
        let mut hidden_state = vec![0.0_f32; 2 * 640];
        let mut cell_state = vec![0.0_f32; 2 * 640];
        let mut tokens: Vec<Token> = Vec::new();
        let mut last_token: Option<usize> = None;
        let mut frame = 0usize;
        let mut emitted = 0usize;

        while frame < encoder_len {
            let frame_slice = &encoder_out[frame * hidden..(frame + 1) * hidden];
            let encoder_frame = frame_slice.to_vec();
            let target = last_token.unwrap_or(self.blank_idx);

            let (logits, new_hidden, new_cell) =
                self.joint_step(&encoder_frame, target, &hidden_state, &cell_state)?;

            let token = argmax(&logits[..vocab_size]);
            let duration_index = argmax(&logits[vocab_size..]);
            let duration = DURATIONS.get(duration_index).copied().unwrap_or(0).max(0) as usize;

            if token != self.blank_idx {
                hidden_state = new_hidden;
                cell_state = new_cell;
                last_token = Some(token);
                tokens.push(Token {
                    id: token,
                    text: self.vocab.get(token).cloned().unwrap_or_default(),
                    start: frame as f64 * FRAME_STEP_SECONDS,
                    end: (frame + duration.max(1)) as f64 * FRAME_STEP_SECONDS,
                });
                emitted += 1;
            }

            if duration > 0 {
                frame += duration;
                emitted = 0;
            } else if token == self.blank_idx || emitted == self.max_tokens_per_step {
                frame += 1;
                emitted = 0;
            }
        }

        let text = decode_text(&tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>());
        Ok(ParakeetResult { text, tokens })
    }

    /// One joint-network step.
    fn joint_step(
        &mut self,
        encoder_frame: &[f32],
        target: usize,
        hidden_state: &[f32],
        cell_state: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let encoder = Tensor::from_array((
            vec![1_i64, encoder_frame.len() as i64, 1_i64],
            encoder_frame.to_vec(),
        ))
        .map_err(|e| ml::onnx_error("could not build the encoder-frame tensor", &e))?;
        let targets = Tensor::from_array((vec![1_i64, 1_i64], vec![target as i32]))
            .map_err(|e| ml::onnx_error("could not build the target tensor", &e))?;
        let target_length = Tensor::from_array((vec![1_i64], vec![1_i32]))
            .map_err(|e| ml::onnx_error("could not build the target-length tensor", &e))?;
        let states_1 = Tensor::from_array((vec![2_i64, 1_i64, 640_i64], hidden_state.to_vec()))
            .map_err(|e| ml::onnx_error("could not build the hidden-state tensor", &e))?;
        let states_2 = Tensor::from_array((vec![2_i64, 1_i64, 640_i64], cell_state.to_vec()))
            .map_err(|e| ml::onnx_error("could not build the cell-state tensor", &e))?;

        let outputs = self
            .joint
            .run(ort::inputs![
                "encoder_outputs" => encoder,
                "targets" => targets,
                "target_length" => target_length,
                "input_states_1" => states_1,
                "input_states_2" => states_2,
            ])
            .map_err(|e| ml::onnx_error("the Parakeet joint network failed", &e))?;

        let (_, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the joint-network output", &e))?;
        let (_, new_hidden) = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the joint hidden state", &e))?;
        let (_, new_cell) = outputs[3]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the joint cell state", &e))?;
        Ok((logits.to_vec(), new_hidden.to_vec(), new_cell.to_vec()))
    }
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

/// Read the `vocab.txt` file: one `"<piece> <id>"` line per entry.
fn read_vocab(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    let mut entries: HashMap<usize, String> = HashMap::new();
    for line in content.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        let Some((piece, id)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(id) = id.trim().parse::<usize>() else {
            continue;
        };
        entries.insert(id, piece.replace('\u{2581}', " "));
    }
    let max_id = entries.keys().copied().max().unwrap_or(0);
    let mut vocab = vec![String::new(); max_id + 1];
    for (id, piece) in entries {
        vocab[id] = piece;
    }
    Ok(vocab)
}

/// Join the token pieces and normalise the whitespace.
///
/// The `SentencePiece` space marker has already been replaced by a real space
/// when the vocabulary was read. The remaining clean-up removes the space
/// before punctuation and at the end of the sentence, and drops the leading
/// space.
pub fn decode_text(pieces: &[&str]) -> String {
    let joined: String = pieces.concat();
    normalize_spaces(&joined)
}

/// Normalise whitespace the way the reference implementation does.
///
/// * a leading space is dropped,
/// * a space before punctuation is dropped,
/// * a space before a word is kept as a single space,
/// * trailing spaces are dropped.
pub fn normalize_spaces(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(chars.len());
    let mut index = 0usize;
    while index < chars.len() {
        let current = chars[index];
        if !current.is_whitespace() {
            out.push(current);
            index += 1;
            continue;
        }
        // A whitespace run: keep one space only when a word follows.
        let mut end = index;
        while end < chars.len() && chars[end].is_whitespace() {
            end += 1;
        }
        let next = chars.get(end).copied();
        let keep = next.is_some_and(is_word_char) && index > 0;
        if keep {
            out.push(' ');
        }
        index = end;
    }
    out
}

/// Whether `c` counts as a word character.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_vocab() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.txt");
        std::fs::write(&path, "<unk> 0\n▁hello 1\n<blk> 2\n").unwrap();
        let vocab = read_vocab(&path).unwrap();
        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab[0], "<unk>");
        assert_eq!(vocab[1], " hello");
        assert_eq!(vocab[2], "<blk>");
    }

    #[test]
    fn normalises_spaces() {
        assert_eq!(normalize_spaces(" hello , world "), "hello, world");
        assert_eq!(normalize_spaces("hello world"), "hello world");
        assert_eq!(normalize_spaces(" hello"), "hello");
    }

    #[test]
    fn decodes_pieces() {
        assert_eq!(decode_text(&[" hello", ",", " world"]), "hello, world");
    }

    #[test]
    fn argmax_finds_the_peak() {
        assert_eq!(argmax(&[1.0, 5.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0]), 0);
    }
}
