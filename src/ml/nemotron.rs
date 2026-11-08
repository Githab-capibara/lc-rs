//! NVIDIA Nemotron 3.5 ASR Streaming 0.6B — ONNX Runtime inference.
//!
//! The checkpoint is three graphs plus a front-end that has to live in our
//! own code, because the published export does not ship a feature extractor:
//!
//! | Stage | Purpose |
//! |-------|---------|
//! | [`StreamingMel`] | raw 16 kHz chunk → 56 log-mel frames (pre-emphasis + Slaney filterbank) |
//! | `encoder.onnx` | 65 mel rows (9 of them the tail of the previous chunk) → 7 encoder states per 0.56 s chunk |
//! | `decoder.onnx` | prediction network: last token + LSTM state → decoder output |
//! | `joint.onnx` | encoder frame + decoder output → vocabulary log-probabilities |
//!
//! The model is a *streaming* transducer: it consumes fixed 0.56 s chunks and
//! carries three caches — two Conformer caches and the mel rows the next chunk
//! sees again. The search is the greedy RNN-T loop of `NeMo`'s
//! `GreedyRNNTInfer`:
//! for every encoder frame, repeat "predict → join → argmax"; a blank advances
//! the frame and leaves the decoder state alone, a non-blank becomes the new
//! conditioning token.
//!
//! The semantics follow the reference C# port of the same ONNX export
//! (`nemotron-speech-csharp`, `SpeechLib.Providers.Nemotron`), which in turn
//! matches the `nemotron_speech` pipeline of ONNX Runtime `GenAI` token for
//! token.
//!
//! ## Language conditioning
//!
//! The encoder takes a *language prompt id* instead of a language string: the
//! first entries of the vocabulary are the prompts the model was trained with
//! (`<ru>`, `<de>`, `<auto>`, …). Id 101 selects automatic detection. Prompt
//! ids can surface as tokens, so they are dropped when the text is assembled.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

use crate::errors::{Error, Result};
use crate::ml;
use crate::ml::parakeet::decode_text;

/// Mel bands of the front-end.
const NUM_MELS: usize = 128;

/// Hidden size of the encoder output.
const ENCODER_HIDDEN: usize = 1024;

/// Hidden size of the prediction network and its states.
const DECODER_HIDDEN: usize = 640;

/// Samples per encoder chunk: 0.56 s at 16 kHz.
const CHUNK_SAMPLES: usize = 8_960;

/// Mel rows the encoder re-reads from the previous chunk.
const PRE_CACHE_ROWS: usize = 9;

/// Mel frames the front-end produces per chunk.
const CHUNK_FRAMES: usize = 56;

/// Rows of the encoder input: the cache rows plus the new frames.
const MEL_ROWS: usize = PRE_CACHE_ROWS + CHUNK_FRAMES;

/// Encoder frames the encoder yields per chunk.
const CHUNK_OUTPUT_FRAMES: usize = 7;

/// FFT size of the analysis.
const FFT_SIZE: usize = 512;

/// Analysis window length, in samples.
const WIN_LENGTH: usize = 400;

/// Hop between frames, in samples.
const HOP: usize = 160;

/// Samples of the pre-emphasised stream carried into the next chunk.
const OVERLAP: usize = FFT_SIZE / 2;

/// Zero-padding the extractor appends after a chunk (also the number of
/// samples a *clean* mode would need as look-ahead).
const WIN_OFFSET: usize = (FFT_SIZE - WIN_LENGTH) / 2;

/// Frequency bins of the power spectrum.
const BINS: usize = FFT_SIZE / 2 + 1;

/// `2^-24`: the additive guard of the logarithm.
const LOG_EPS: f32 = 5.960_464_5e-8;

/// Pre-emphasis coefficient.
const PREEMPH: f32 = 0.97;

/// Language prompt id that lets the model pick the language itself.
pub const PROMPT_AUTO: i64 = 101;

/// One recognised token.
#[derive(Debug, Clone)]
pub struct Token {
    /// Vocabulary id.
    pub id: usize,
    /// Piece text (the `SentencePiece` space marker already replaced).
    pub text: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
}

/// Result of one Nemotron decoding pass.
#[derive(Debug, Clone, Default)]
pub struct NemotronResult {
    /// The recognised text.
    pub text: String,
    /// The recognised tokens with their timestamps.
    pub tokens: Vec<Token>,
}

/// A loaded Nemotron checkpoint.
pub struct Nemotron {
    encoder: Session,
    decoder: Session,
    joint: Session,
    /// Pieces, index = token id; prompt entries are empty.
    vocab: Vec<String>,
    /// Language prompt id passed to the encoder.
    lang_id: i64,
    /// Vocabulary id of the blank token.
    blank_idx: usize,
    /// Maximum number of tokens emitted on a single encoder frame.
    max_symbols_per_step: usize,
}

impl std::fmt::Debug for Nemotron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Nemotron")
            .field("vocab_size", &self.vocab.len())
            .field("lang_id", &self.lang_id)
            .field("blank_idx", &self.blank_idx)
            .finish_non_exhaustive()
    }
}

/// Everything the model carries from one chunk of a stream to the next: the
/// encoder caches, the committed decoder state and its last token.
struct StreamState {
    /// `[1, 24, 70, 1024]` — attention padding cache.
    channel: Vec<f32>,
    /// `[1, 24, 1024, 8]` — convolution cache.
    time: Vec<f32>,
    /// `[1]` — how many frames of the caches are valid so far.
    channel_len: Vec<i64>,
    /// `[2, 1, 640]` — committed LSTM hidden state.
    hidden: Vec<f32>,
    /// `[2, 1, 640]` — committed LSTM cell state.
    cell: Vec<f32>,
}

impl StreamState {
    /// Fresh state, as if the stream had just been reset.
    fn new() -> Self {
        Self {
            channel: vec![0.0; 24 * 70 * ENCODER_HIDDEN],
            time: vec![0.0; 24 * ENCODER_HIDDEN * 8],
            channel_len: vec![0],
            hidden: vec![0.0; 2 * DECODER_HIDDEN],
            cell: vec![0.0; 2 * DECODER_HIDDEN],
        }
    }
}

impl Nemotron {
    /// Load the three graphs and the vocabulary from `dir`.
    pub fn load(dir: &Path) -> Result<Self> {
        let encoder = ml::load_session(&dir.join("encoder.onnx"))?;
        let decoder = ml::load_session(&dir.join("decoder.onnx"))?;
        let joint = ml::load_session(&dir.join("joint.onnx"))?;
        let (vocab, blank_idx) = read_vocab(&dir.join("vocab.txt"))?;
        Ok(Self {
            encoder,
            decoder,
            joint,
            vocab,
            lang_id: PROMPT_AUTO,
            blank_idx,
            max_symbols_per_step: 10,
        })
    }

    /// Select the language conditioning of the encoder.
    pub fn set_language(&mut self, prompt: i64) {
        self.lang_id = prompt;
    }

    /// Vocabulary id of the blank token.
    pub fn blank_idx(&self) -> usize {
        self.blank_idx
    }

    /// Transcribe mono 16 kHz samples.
    ///
    /// The whole buffer is one stream: chunk by chunk through the front-end
    /// and the encoder, with a greedy transducer search over every chunk's
    /// frames. A short tail is zero-padded, exactly like the reference.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<NemotronResult> {
        if samples.is_empty() {
            return Ok(NemotronResult::default());
        }

        let mut mel = StreamingMel::new();
        let mut state = StreamState::new();
        let mut input = vec![0.0_f32; MEL_ROWS * NUM_MELS];
        let mut tokens: Vec<Token> = Vec::new();
        let mut last = self.blank_idx;

        let mut position = 0usize;
        let mut chunk_index = 0usize;
        while position < samples.len() {
            // The reference zero-pads the final chunk instead of dropping it.
            let end = (position + CHUNK_SAMPLES).min(samples.len());
            let mut chunk = samples[position..end].to_vec();
            let padded = chunk.len() < CHUNK_SAMPLES;
            if padded {
                chunk.resize(CHUNK_SAMPLES, 0.0);
            }

            input[PRE_CACHE_ROWS * NUM_MELS..].copy_from_slice(&mel.process(&chunk));

            let encoded = self.encode_chunk(&input, &mut state)?;
            let steps = encoded.len() / ENCODER_HIDDEN;

            // The next chunk re-reads the last nine rows of this input.
            let split = (MEL_ROWS - PRE_CACHE_ROWS) * NUM_MELS;
            input.copy_within(split.., 0);

            for frame in 0..steps {
                let offset = frame * ENCODER_HIDDEN;
                let encoder_frame = &encoded[offset..offset + ENCODER_HIDDEN];
                let mut symbol = 0usize;
                while symbol < self.max_symbols_per_step {
                    let (logits, new_hidden, new_cell) = self.step(encoder_frame, last, &state)?;
                    let token = argmax(&logits, self.blank_idx);
                    if token == self.blank_idx {
                        // A blank ends the step and leaves the state alone.
                        break;
                    }
                    state.hidden = new_hidden;
                    state.cell = new_cell;
                    last = token;
                    let start = (chunk_index * CHUNK_OUTPUT_FRAMES + frame) as f64 * 0.08;
                    tokens.push(Token {
                        id: token,
                        text: self.vocab.get(token).cloned().unwrap_or_default(),
                        start,
                        end: start + 0.08,
                    });
                    symbol += 1;
                }
            }

            let _ = padded;
            position += CHUNK_SAMPLES;
            chunk_index += 1;
        }

        let text = decode_text(&tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>());
        Ok(NemotronResult { text, tokens })
    }

    /// Run the encoder on one chunk of mel rows and refresh the caches.
    fn encode_chunk(&mut self, input: &[f32], state: &mut StreamState) -> Result<Vec<f32>> {
        let audio = Tensor::from_array((
            vec![1_i64, MEL_ROWS as i64, NUM_MELS as i64],
            input.to_vec(),
        ))
        .map_err(|e| ml::onnx_error("could not build the audio tensor", &e))?;
        let length = Tensor::from_array((vec![1_i64], vec![MEL_ROWS as i64]))
            .map_err(|e| ml::onnx_error("could not build the length tensor", &e))?;
        let channel = Tensor::from_array((
            vec![1_i64, 24, 70, ENCODER_HIDDEN as i64],
            state.channel.clone(),
        ))
        .map_err(|e| ml::onnx_error("could not build the channel cache", &e))?;
        let time = Tensor::from_array((
            vec![1_i64, 24, ENCODER_HIDDEN as i64, 8],
            state.time.clone(),
        ))
        .map_err(|e| ml::onnx_error("could not build the time cache", &e))?;
        let channel_len = Tensor::from_array((vec![1_i64], state.channel_len.clone()))
            .map_err(|e| ml::onnx_error("could not build the cache length", &e))?;
        let lang = Tensor::from_array((vec![1_i64], vec![self.lang_id]))
            .map_err(|e| ml::onnx_error("could not build the language tensor", &e))?;

        let outputs = self
            .encoder
            .run(ort::inputs![
                "audio_signal" => audio,
                "length" => length,
                "cache_last_channel" => channel,
                "cache_last_time" => time,
                "cache_last_channel_len" => channel_len,
                "lang_id" => lang,
            ])
            .map_err(|e| ml::onnx_error("the Nemotron encoder failed", &e))?;

        let (_, encoded) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Nemotron encoder output", &e))?;
        state.channel = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Nemotron channel cache", &e))?
            .1
            .to_vec();
        state.time = outputs[3]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the Nemotron time cache", &e))?
            .1
            .to_vec();
        state.channel_len = outputs[4]
            .try_extract_tensor::<i64>()
            .map_err(|e| ml::onnx_error("could not read the Nemotron cache length", &e))?
            .1
            .to_vec();
        Ok(encoded.to_vec())
    }

    /// One prediction-network step plus the joint network over one frame.
    fn step(
        &mut self,
        encoder_frame: &[f32],
        target: usize,
        state: &StreamState,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let targets = Tensor::from_array((vec![1_i64, 1_i64], vec![target as i64]))
            .map_err(|e| ml::onnx_error("could not build the target tensor", &e))?;
        let hidden = Tensor::from_array((
            vec![2_i64, 1_i64, DECODER_HIDDEN as i64],
            state.hidden.clone(),
        ))
        .map_err(|e| ml::onnx_error("could not build the hidden-state tensor", &e))?;
        let cell = Tensor::from_array((
            vec![2_i64, 1_i64, DECODER_HIDDEN as i64],
            state.cell.clone(),
        ))
        .map_err(|e| ml::onnx_error("could not build the cell-state tensor", &e))?;

        let prediction = self
            .decoder
            .run(ort::inputs![
                "targets" => targets,
                "h_in" => hidden,
                "c_in" => cell,
            ])
            .map_err(|e| ml::onnx_error("the Nemotron prediction network failed", &e))?;

        // The graph emits `[batch, hidden, 1]`; the joint wants `[batch, 1, hidden]`.
        let (pred_shape, pred_data) = prediction[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the decoder output", &e))?;
        let pred_hidden = pred_shape.get(1).copied().unwrap_or(0).max(0) as usize;
        let pred_len = pred_shape.get(2).copied().unwrap_or(0).max(0) as usize;
        let mut transposed = vec![0.0_f32; pred_hidden * pred_len];
        for step in 0..pred_len {
            for channel in 0..pred_hidden {
                transposed[step * pred_hidden + channel] = pred_data[channel * pred_len + step];
            }
        }

        let encoder = Tensor::from_array((
            vec![1_i64, 1_i64, encoder_frame.len() as i64],
            encoder_frame.to_vec(),
        ))
        .map_err(|e| ml::onnx_error("could not build the encoder-frame tensor", &e))?;
        let decoder_out =
            Tensor::from_array((vec![1_i64, pred_len as i64, pred_hidden as i64], transposed))
                .map_err(|e| ml::onnx_error("could not build the decoder-output tensor", &e))?;

        let logits = self
            .joint
            .run(ort::inputs![
                "encoder_output" => encoder,
                "decoder_output" => decoder_out,
            ])
            .map_err(|e| ml::onnx_error("the Nemotron joint network failed", &e))?;

        let row = logits[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the joint output", &e))?
            .1
            .to_vec();
        let new_hidden = prediction[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the new hidden state", &e))?
            .1
            .to_vec();
        let new_cell = prediction[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| ml::onnx_error("could not read the new cell state", &e))?
            .1
            .to_vec();
        Ok((row, new_hidden, new_cell))
    }
}

/// Index of the largest finite value; ties keep the lowest index.
///
/// Non-finite logits are skipped so a `NaN` in a quantised graph cannot turn
/// into a random token. When nothing is left the blank wins and the search
/// simply advances a frame.
fn argmax(values: &[f32], blank: usize) -> usize {
    let mut best: Option<usize> = None;
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        if best.is_none_or(|current| *value > values[current]) {
            best = Some(index);
        }
    }
    best.unwrap_or(blank)
}

/// Streaming log-mel front-end of the Nemotron export.
///
/// Port of the ORT-Extensions `NemoStreamingMelExtractor` that ONNX Runtime
/// `GenAI` drives
/// chunk by chunk: pre-emphasis with a carried sample, a 256-sample left
/// overlap, a symmetric Hann window, a 512-point power spectrum, a Slaney
/// filterbank and `ln(x + 2^-24)`. No dither, no normalisation.
struct StreamingMel {
    window: Vec<f32>,
    weights: Vec<f32>,
    first_bin: Vec<usize>,
    bin_count: Vec<usize>,
    offset: Vec<usize>,
    overlap: Vec<f32>,
    previous: f32,
}

impl StreamingMel {
    /// Build the filterbank and the analysis window.
    fn new() -> Self {
        let window: Vec<f32> = (0..WIN_LENGTH)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (WIN_LENGTH - 1) as f32).cos())
            })
            .collect();

        let dense = mel_filterbank();
        let mut weights: Vec<f32> = Vec::new();
        let mut first_bin = vec![0usize; NUM_MELS];
        let mut bin_count = vec![0usize; NUM_MELS];
        let mut offset = vec![0usize; NUM_MELS];
        for band in 0..NUM_MELS {
            let mut first = None;
            let mut last = 0usize;
            for (bin, weight) in dense[band].iter().enumerate() {
                if *weight == 0.0 {
                    continue;
                }
                if first.is_none() {
                    first = Some(bin);
                }
                last = bin;
            }
            offset[band] = weights.len();
            if let Some(first) = first {
                first_bin[band] = first;
                bin_count[band] = last - first + 1;
                weights.extend_from_slice(&dense[band][first..=last]);
            }
        }

        Self {
            window,
            weights,
            first_bin,
            bin_count,
            offset,
            overlap: vec![0.0; OVERLAP],
            previous: 0.0,
        }
    }

    /// Mel frames of one chunk, band-major flattened (`[frames, 128]`).
    fn process(&mut self, chunk: &[f32]) -> Vec<f32> {
        let count = frames_for(chunk.len());
        let mut padded = vec![0.0_f32; OVERLAP + chunk.len() + WIN_OFFSET];
        padded[..OVERLAP].copy_from_slice(&self.overlap);
        padded[OVERLAP] = chunk[0] - PREEMPH * self.previous;
        for (index, sample) in chunk.iter().enumerate().skip(1) {
            padded[OVERLAP + index] = sample - PREEMPH * chunk[index - 1];
        }

        // The overlap of the next chunk is the last `OVERLAP` samples of the
        // pre-emphasised stream so far, so no sample is analysed twice at a
        // chunk boundary.
        self.overlap
            .copy_from_slice(&padded[chunk.len()..chunk.len() + OVERLAP]);
        self.previous = chunk[chunk.len() - 1];

        let mut frames = vec![0.0_f32; count * NUM_MELS];
        let mut windowed: Vec<rustfft::num_complex::Complex32> =
            vec![rustfft::num_complex::Complex32::default(); FFT_SIZE];
        let mut power = vec![0.0_f32; BINS];
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        for index in 0..count {
            let start = index * HOP + WIN_OFFSET;
            // The window fills the first `WIN_LENGTH` bins; the rest is the
            // zero-padding of the 512-point transform, which has to be reset
            // every frame because the FFT runs in place.
            for slot in 0..WIN_LENGTH {
                windowed[slot] = (padded[start + slot] * self.window[slot]).into();
            }
            for slot in windowed.iter_mut().take(FFT_SIZE).skip(WIN_LENGTH) {
                *slot = rustfft::num_complex::Complex32::default();
            }
            fft.process(&mut windowed);
            for (slot, value) in power.iter_mut().enumerate() {
                *value = windowed[slot].norm_sqr();
            }
            let out = &mut frames[index * NUM_MELS..(index + 1) * NUM_MELS];
            for (band, slot) in out.iter_mut().enumerate() {
                let first = self.first_bin[band];
                let count = self.bin_count[band];
                let offset = self.offset[band];
                let mut sum = 0.0_f32;
                for j in 0..count {
                    sum += self.weights[offset + j] * power[first + j];
                }
                *slot = (sum + LOG_EPS).ln();
            }
        }
        frames
    }
}

/// Mel frames a chunk of `samples` produces.
fn frames_for(samples: usize) -> usize {
    // `OVERLAP + WIN_LENGTH` is 656, so the subtraction only underflows for
    // chunks shorter than the analysis window, which have no complete frame.
    match samples.checked_sub(WIN_LENGTH) {
        Some(span) => (OVERLAP + span) / HOP + 1,
        None => 0,
    }
}

/// Slaney filterbank: `[128, 257]`, area-normalised triangles from 0 Hz to Nyquist.
fn mel_filterbank() -> Vec<Vec<f32>> {
    const LIN_SCALE: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOG_STEP: f32 = 0.068_751_78;
    const EPS: f32 = 1e-10;

    let hz_to_mel = |hz: f32| {
        if hz < MIN_LOG_HZ {
            hz / LIN_SCALE
        } else {
            MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
        }
    };
    let mel_to_hz = |mel: f32| {
        if mel < MIN_LOG_MEL {
            mel * LIN_SCALE
        } else {
            MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
        }
    };

    let low = hz_to_mel(0.0);
    let high = hz_to_mel(8000.0);
    let mel_hz: Vec<f32> = (0..NUM_MELS + 2)
        .map(|i| mel_to_hz(low + (high - low) * i as f32 / (NUM_MELS + 1) as f32))
        .collect();
    let diff: Vec<f32> = (0..=NUM_MELS).map(|i| mel_hz[i + 1] - mel_hz[i]).collect();

    (0..NUM_MELS)
        .map(|band| {
            let row: Vec<f32> = (0..BINS)
                .map(|bin| {
                    let hz = bin as f32 * 16_000.0 / FFT_SIZE as f32;
                    let lower = (hz - mel_hz[band]) / (diff[band] + EPS);
                    let upper = (mel_hz[band + 2] - hz) / (diff[band + 1] + EPS);
                    lower.max(0.0).min(upper.max(0.0))
                })
                .collect();
            let enorm = 2.0 / (mel_hz[band + 2] - mel_hz[band] + EPS);
            row.iter().map(|weight| weight * enorm).collect()
        })
        .collect()
}

/// Read `vocab.txt`: one piece per line, id = line index.
///
/// Returns the pieces and the id of the blank token. The `SentencePiece`
/// space marker becomes a real space, and the special entries — the language
/// prompts at the head of the file, `<unk>` and `<blank>` — become empty
/// strings: they stay index aligned but never reach the transcript, which is
/// what `skip_special_tokens` does in the reference implementation.
fn read_vocab(path: &Path) -> Result<(Vec<String>, usize)> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    let mut pieces = Vec::new();
    let mut blank = None;
    for line in content.lines() {
        let raw = line.trim_end_matches(['\r', '\n']);
        let index = pieces.len();
        let piece = raw.replace('\u{2581}', " ");
        let special = piece.starts_with('<') && piece.ends_with('>');
        if special && piece == "<blank>" {
            blank = Some(index);
        }
        pieces.push(if special { String::new() } else { piece });
    }
    let blank = blank.unwrap_or_else(|| pieces.len().saturating_sub(1));
    Ok((pieces, blank))
}

/// Map a language code to the Nemotron language prompt id.
///
/// Unknown codes fall back to [`PROMPT_AUTO`], which lets the model decide.
/// Both the bare code (`de`) and the regional form (`de-DE`) are accepted.
pub fn prompt_for_language(code: &str) -> i64 {
    let code = code.trim();
    match PROMPTS
        .iter()
        .find(|(code_pattern, _)| code_pattern == &code)
        .map(|(_, prompt)| *prompt)
    {
        Some(prompt) => prompt,
        None => PROMPT_AUTO,
    }
}

/// Language prompt ids, in the order of `processor_config.json`'s
/// `prompt_dictionary`. The shortest spelling of each language is used.
/// Language prompt ids, taken from the model's `processor_config.json`
/// `prompt_dictionary`: every spelling of every language the model was
/// trained on maps to the prompt id that conditions the encoder.
const PROMPTS: &[(&str, i64)] = &[
    ("en", 0),
    ("en-US", 0),
    ("en-GB", 1),
    ("enGB", 1),
    ("es-ES", 2),
    ("esES", 2),
    ("es", 3),
    ("es-US", 3),
    ("zh-CN", 4),
    ("zh-ZH", 4),
    ("zh-TW", 5),
    ("hi", 6),
    ("hi-HI", 6),
    ("hi-IN", 6),
    ("ar", 7),
    ("ar-AR", 7),
    ("fr", 8),
    ("fr-FR", 8),
    ("de", 9),
    ("de-DE", 9),
    ("ja-JA", 10),
    ("ja-JP", 10),
    ("ru", 11),
    ("ru-RU", 11),
    ("pt-BR", 12),
    ("pt", 13),
    ("pt-PT", 13),
    ("ko", 14),
    ("ko-KO", 14),
    ("ko-KR", 14),
    ("it", 15),
    ("it-IT", 15),
    ("nl", 16),
    ("nl-NL", 16),
    ("pl", 17),
    ("pl-PL", 17),
    ("tr", 18),
    ("tr-TR", 18),
    ("uk", 19),
    ("uk-UA", 19),
    ("ro", 20),
    ("ro-RO", 20),
    ("el", 21),
    ("el-GR", 21),
    ("cs", 22),
    ("cs-CZ", 22),
    ("hu", 23),
    ("hu-HU", 23),
    ("sv", 24),
    ("sv-SE", 24),
    ("da", 25),
    ("da-DK", 25),
    ("fi", 26),
    ("fi-FI", 26),
    ("no", 27),
    ("no-NO", 27),
    ("sk", 28),
    ("sk-SK", 28),
    ("hr", 29),
    ("hr-HR", 29),
    ("bg", 30),
    ("bg-BG", 30),
    ("lt", 31),
    ("lt-LT", 31),
    ("th-TH", 32),
    ("vi-VN", 33),
    ("id-ID", 34),
    ("ms-MY", 35),
    ("bn-IN", 36),
    ("ur-PK", 37),
    ("fa-IR", 38),
    ("ta-IN", 39),
    ("te-IN", 40),
    ("mr-IN", 41),
    ("gu-IN", 42),
    ("kn-IN", 43),
    ("ml-IN", 44),
    ("si-LK", 45),
    ("ne-NP", 46),
    ("km-KH", 47),
    ("sw-KE", 48),
    ("am-ET", 49),
    ("ha-NG", 50),
    ("zu-ZA", 51),
    ("yo-NG", 52),
    ("ig-NG", 53),
    ("af-ZA", 54),
    ("rw-RW", 55),
    ("so-SO", 56),
    ("ny-MW", 57),
    ("ln-CD", 58),
    ("or-KE", 59),
    ("et", 60),
    ("et-EE", 60),
    ("lv", 61),
    ("lv-LV", 61),
    ("sl", 62),
    ("sl-SI", 62),
    ("he-IL", 64),
    ("ku-TR", 65),
    ("az-AZ", 66),
    ("ka-GE", 67),
    ("hy-AM", 68),
    ("uz-UZ", 69),
    ("tg-TJ", 70),
    ("ky-KG", 71),
    ("qu-PE", 80),
    ("ay-BO", 81),
    ("gn-PY", 82),
    ("nah-MX", 83),
    ("mi-NZ", 96),
    ("haw-US", 97),
    ("sm-WS", 98),
    ("to-TO", 99),
    ("fr-CA", 100),
    ("auto", 101),
    ("mt-MT", 102),
    ("nb", 103),
    ("nb-NO", 103),
    ("nn", 104),
    ("nn-NO", 104),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_for_a_full_chunk() {
        assert_eq!(frames_for(CHUNK_SAMPLES), CHUNK_FRAMES);
    }

    #[test]
    fn frames_for_short_chunks() {
        // Shorter than one analysis window: no complete frame fits.
        assert_eq!(frames_for(WIN_LENGTH - 1), 0);
        // One window plus the 256-sample overlap on both sides: two frames.
        assert_eq!(frames_for(WIN_LENGTH), 2);
        assert_eq!(frames_for(WIN_LENGTH + HOP), 3);
    }

    #[test]
    fn mel_rows_match_the_encoder_input() {
        assert_eq!(MEL_ROWS, 65);
        assert_eq!(MEL_ROWS * NUM_MELS, 8_320);
    }

    #[test]
    fn filterbank_covers_the_band() {
        let fb = mel_filterbank();
        assert_eq!(fb.len(), NUM_MELS);
        assert_eq!(fb[0].len(), BINS);
        // The first band starts at 0 Hz, so its first weight is zero.
        assert!(fb[0][0].abs() < 1e-6);
        // The last band reaches Nyquist.
        assert!(fb[NUM_MELS - 1][BINS - 1].abs() < 1e-6);
        let sum: f32 = fb[64].iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn window_is_symmetric_hann() {
        let mel = StreamingMel::new();
        assert!(mel.window[0].abs() < 1e-6);
        assert!(mel.window[WIN_LENGTH - 1].abs() < 1e-6);
        assert!(mel.window[WIN_LENGTH / 2] > 0.9);
    }

    #[test]
    fn filterbank_is_compacted() {
        let mel = StreamingMel::new();
        let packed: usize = mel.bin_count.iter().sum();
        let dense = mel_filterbank();
        let dense_nonzero: usize = dense
            .iter()
            .map(|row| row.iter().filter(|weight| **weight != 0.0).count())
            .sum();
        assert_eq!(packed, dense_nonzero);
        // DC carries no weight in the first band, so it starts at bin 1.
        assert_eq!(mel.first_bin[0], 1);
    }

    #[test]
    fn reads_prompts_and_pieces() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.txt");
        std::fs::write(&path, "\u{2581}The\n\u{2581}B\nde\n<ru>\n<blank>\n").unwrap();
        let (vocab, blank) = read_vocab(&path).unwrap();
        assert_eq!(vocab.len(), 5);
        assert_eq!(vocab[0], " The");
        // Index alignment is kept, but the special entries carry no text.
        assert_eq!(vocab[3], "");
        assert_eq!(vocab[4], "");
        assert_eq!(blank, 4);
    }

    #[test]
    fn maps_language_prompts() {
        assert_eq!(prompt_for_language("ru"), 11);
        assert_eq!(prompt_for_language("de"), 9);
        assert_eq!(prompt_for_language("en-US"), 0);
        assert_eq!(prompt_for_language("zz"), PROMPT_AUTO);
        assert_eq!(prompt_for_language(""), PROMPT_AUTO);
    }

    #[test]
    fn argmax_skips_non_finite() {
        assert_eq!(argmax(&[f32::NAN, 0.5, 2.0, 1.0], 99), 2);
        assert_eq!(argmax(&[f32::NAN, f32::NAN], 7), 7);
        assert_eq!(argmax(&[3.0], 0), 0);
    }

    #[test]
    fn empty_input_is_empty() {
        // Only the pure parts are exercised: loading a graph needs the model.
        let mel = StreamingMel::new();
        assert_eq!(frames_for(0), 0);
        assert_eq!(mel.overlap.len(), OVERLAP);
    }
}
