//! Log-mel spectrogram front-ends.
//!
//! Two flavours are needed:
//!
//! * **Whisper-style** — used by the Qwen3-ASR encoder: HTK or Slaney mel
//!   scale, optional Slaney normalisation, power spectrogram, `log10` with
//!   a dynamic-range clamp.
//! * **NeMo-style** — used by the Nemotron encoder: pre-emphasis, dither,
//!   `center`-padded STFT and a plain natural logarithm.
//!
//! The implementations follow `torch.stft` semantics exactly (periodic Hann
//! window, reflect padding, `1 + n // hop` frames) so the numbers match the
//! Python reference bit for bit within floating-point tolerance.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

use crate::errors::{Error, Result};

/// Mel scale used to space the filterbank centres.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelScale {
    /// HTK formula: `2595 * log10(1 + f / 700)`.
    Htk,
    /// Slaney formula: linear below 1 kHz, logarithmic above.
    Slaney,
}

/// Normalisation applied to the filterbank weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelNorm {
    /// No normalisation.
    None,
    /// Slaney normalisation: each band is scaled to roughly constant energy.
    Slaney,
}

/// Configuration of the mel front-end.
#[derive(Debug, Clone, Copy)]
pub struct MelConfig {
    /// FFT size.
    pub n_fft: usize,
    /// Hop between successive frames.
    pub hop_length: usize,
    /// Window length (may be smaller than `n_fft`).
    pub win_length: usize,
    /// Number of mel bands.
    pub n_mels: usize,
    /// Lower cut-off frequency in Hz.
    pub fmin: f32,
    /// Upper cut-off frequency in Hz.
    pub fmax: f32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Mel scale.
    pub mel_scale: MelScale,
    /// Filterbank normalisation.
    pub norm: MelNorm,
    /// Whether the STFT is centred (reflect-padded by `n_fft / 2`).
    pub center: bool,
    /// Pre-emphasis coefficient (`0.0` disables it).
    pub preemphasis: f32,
    /// Dither amplitude (`0.0` disables it).
    pub dither: f32,
    /// Whether the natural logarithm is used instead of `log10`.
    pub natural_log: bool,
    /// Minimum value before taking the logarithm.
    pub log_floor: f32,
    /// Dynamic range clamp applied after the logarithm (`None` disables it).
    pub log_range: Option<f32>,
    /// Offset and scale applied after the logarithm (`(x + offset) / scale`).
    pub log_offset: f32,
    pub log_scale: f32,
}

impl MelConfig {
    /// Whisper-style configuration used by the Qwen3-ASR encoder.
    ///
    /// `transformers`' `WhisperFeatureExtractor` builds its filter bank with
    /// `mel_filter_bank(..., norm="slaney", mel_scale="slaney")`, so the
    /// Slaney scale *and* the Slaney normalisation are part of the contract —
    /// the HTK variant would scale the low bands differently and change the
    /// encoder input.
    pub fn whisper(n_mels: usize) -> Self {
        Self {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels,
            fmin: 0.0,
            fmax: 8000.0,
            sample_rate: 16_000,
            mel_scale: MelScale::Slaney,
            norm: MelNorm::Slaney,
            center: true,
            preemphasis: 0.0,
            dither: 0.0,
            natural_log: false,
            log_floor: 1e-10,
            log_range: Some(8.0),
            log_offset: 4.0,
            log_scale: 4.0,
        }
    }

    /// NeMo-style configuration used by the Nemotron encoder.
    pub fn nemo(n_mels: usize) -> Self {
        Self {
            n_fft: 512,
            hop_length: 160,
            win_length: 400,
            n_mels,
            fmin: 0.0,
            fmax: 8000.0,
            sample_rate: 16_000,
            mel_scale: MelScale::Slaney,
            norm: MelNorm::None,
            center: true,
            preemphasis: 0.97,
            dither: 1e-5,
            natural_log: true,
            log_floor: 1e-10,
            log_range: None,
            log_offset: 0.0,
            log_scale: 1.0,
        }
    }
}

/// Computed log-mel features.
#[derive(Debug, Clone)]
pub struct MelFeatures {
    /// Number of mel bands.
    pub n_mels: usize,
    /// Number of frames.
    pub n_frames: usize,
    /// Flattened features, band-major (`n_mels * n_frames` values).
    pub data: Vec<f32>,
}

impl MelFeatures {
    /// Value of one band/frame pair.
    pub fn at(&self, band: usize, frame: usize) -> f32 {
        self.data[band * self.n_frames + frame]
    }
}

/// Convert Hz to mels.
pub fn hz_to_mel(hz: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
        MelScale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0_f32;
            let min_log_mel = min_log_hz / f_sp;
            let logstep = 6.4_f32.ln() / 27.0;
            if hz >= min_log_hz {
                min_log_mel + (hz / min_log_hz).ln() / logstep
            } else {
                hz / f_sp
            }
        }
    }
}

/// Convert mels back to Hz.
pub fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0_f32;
            let min_log_mel = min_log_hz / f_sp;
            let logstep = 6.4_f32.ln() / 27.0;
            if mel >= min_log_mel {
                min_log_hz * (logstep * (mel - min_log_mel)).exp()
            } else {
                f_sp * mel
            }
        }
    }
}

/// Build the mel filterbank matrix (`n_mels` x `n_fft / 2 + 1`).
pub fn mel_filterbank(config: &MelConfig) -> Vec<Vec<f32>> {
    let n_bins = config.n_fft / 2 + 1;
    let mut centres = Vec::with_capacity(config.n_mels + 2);
    let min_mel = hz_to_mel(config.fmin, config.mel_scale);
    let max_mel = hz_to_mel(config.fmax, config.mel_scale);
    for i in 0..(config.n_mels + 2) {
        let mel = min_mel + (max_mel - min_mel) * (i as f32) / ((config.n_mels + 1) as f32);
        centres.push(mel_to_hz(mel, config.mel_scale));
    }

    let mut weights = vec![vec![0.0_f32; n_bins]; config.n_mels];
    for (band, row) in weights.iter_mut().enumerate() {
        let lower_edge = centres[band];
        let centre = centres[band + 1];
        let upper_edge = centres[band + 2];
        for (bin, weight) in row.iter_mut().enumerate() {
            let freq = bin as f32 * config.sample_rate as f32 / config.n_fft as f32;
            let lower = (freq - lower_edge) / (centre - lower_edge);
            let upper = (upper_edge - freq) / (upper_edge - centre);
            *weight = lower.min(upper).max(0.0);
        }
    }

    if config.norm == MelNorm::Slaney {
        for (band, row) in weights.iter_mut().enumerate() {
            let enorm = 2.0 / (centres[band + 2] - centres[band]);
            for weight in row.iter_mut() {
                *weight *= enorm;
            }
        }
    }
    weights
}

/// Compute the log-mel spectrogram of `samples`.
pub fn log_mel_spectrogram(samples: &[f32], config: &MelConfig) -> Result<MelFeatures> {
    if config.n_fft == 0 || config.hop_length == 0 || config.win_length == 0 {
        return Err(Error::Transcription(
            "invalid mel configuration (zero n_fft/hop/win)".to_string(),
        ));
    }
    if config.win_length > config.n_fft {
        return Err(Error::Transcription(
            "mel window length exceeds the FFT size".to_string(),
        ));
    }

    let mut signal: Vec<f32> = Vec::with_capacity(samples.len());
    if config.preemphasis == 0.0 {
        signal.extend_from_slice(samples);
    } else {
        signal.push(samples.first().copied().unwrap_or(0.0));
        for pair in samples.windows(2) {
            signal.push(pair[1] - config.preemphasis * pair[0]);
        }
    }

    if config.dither > 0.0 {
        // Deterministic dither: a fixed low-discrepancy sequence keeps the
        // result reproducible while still breaking up quantisation noise.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for sample in &mut signal {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let uniform = ((state >> 11) as f32 / (1u64 << 53) as f32) - 0.5;
            *sample += config.dither * uniform;
        }
    }

    let frames = if config.center {
        // torch.stft pads `n_fft / 2` samples on both sides, so the frame
        // count is `1 + n // hop`.
        signal.len() / config.hop_length + 1
    } else if signal.len() >= config.n_fft {
        1 + (signal.len() - config.n_fft) / config.hop_length
    } else {
        0
    };
    let frames = frames.max(1);

    let window = hann_window(config.win_length, config.n_fft);
    let padded = if config.center {
        reflect_pad(&signal, config.n_fft / 2)
    } else {
        signal.clone()
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(config.n_fft);
    let n_bins = config.n_fft / 2 + 1;
    let mut spectra: Vec<Vec<f32>> = Vec::with_capacity(frames);

    for frame in 0..frames {
        let start = frame * config.hop_length;
        let mut buffer: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); config.n_fft];
        for (i, slot) in buffer.iter_mut().enumerate() {
            let value = padded.get(start + i).copied().unwrap_or(0.0) * window[i];
            *slot = Complex32::new(value, 0.0);
        }
        fft.process(&mut buffer);
        let mut spectrum = vec![0.0_f32; n_bins];
        for (bin, value) in spectrum.iter_mut().enumerate() {
            *value = buffer[bin].norm_sqr();
        }
        spectra.push(spectrum);
    }

    let filters = mel_filterbank(config);
    let n_frames = frames;
    let mut data = vec![0.0_f32; config.n_mels * n_frames];
    for (band, row) in filters.iter().enumerate() {
        for frame in 0..n_frames {
            let mut sum = 0.0_f32;
            for (bin, weight) in row.iter().enumerate() {
                if *weight == 0.0 {
                    continue;
                }
                sum += weight * spectra[frame][bin];
            }
            data[band * n_frames + frame] = sum;
        }
    }

    // Logarithm, dynamic-range clamp and affine rescaling.
    let mut max_log = f32::NEG_INFINITY;
    if config.log_range.is_some() {
        for value in &data {
            let logged = if config.natural_log {
                (*value + config.log_floor).ln()
            } else {
                (*value).max(config.log_floor).log10()
            };
            if logged > max_log {
                max_log = logged;
            }
        }
    }
    let threshold = config
        .log_range
        .map_or(f32::NEG_INFINITY, |range| max_log - range);
    for value in &mut data {
        let mut logged = if config.natural_log {
            (*value + config.log_floor).ln()
        } else {
            (*value).max(config.log_floor).log10()
        };
        if logged < threshold {
            logged = threshold;
        }
        *value = (logged + config.log_offset) / config.log_scale;
    }

    Ok(MelFeatures {
        n_mels: config.n_mels,
        n_frames,
        data,
    })
}

/// Periodic Hann window of `win_length` samples, zero-padded to `n_fft`.
fn hann_window(win_length: usize, n_fft: usize) -> Vec<f32> {
    let mut window = vec![0.0_f32; n_fft];
    for (i, slot) in window.iter_mut().enumerate().take(win_length) {
        *slot = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / win_length as f32).cos();
    }
    window
}

/// Pad `signal` with `pad` reflected samples on both sides.
///
/// Mirrors `torch`'s `reflect` mode: the left pad is `x[pad] … x[1]` and the
/// right pad is `x[n-2] … x[n-pad-1]`.
fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 || signal.is_empty() {
        return signal.to_vec();
    }
    let n = signal.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in 0..pad {
        let index = if n > 1 { (pad - i).min(n - 1) } else { 0 };
        out.push(signal[index]);
    }
    out.extend_from_slice(signal);
    for i in 0..pad {
        let index = if n > 1 { n.saturating_sub(2 + i) } else { 0 };
        out.push(signal[index]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn htk_mel_round_trips() {
        assert!((hz_to_mel(0.0, MelScale::Htk)).abs() < 1e-6);
        let hz = 4000.0;
        let back = mel_to_hz(hz_to_mel(hz, MelScale::Htk), MelScale::Htk);
        assert!((back - hz).abs() < 1.0);
    }

    #[test]
    fn slaney_mel_round_trips() {
        for hz in [0.0_f32, 500.0, 1000.0, 4000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz, MelScale::Slaney), MelScale::Slaney);
            assert!((back - hz).abs() < 1.0, "hz={hz} back={back}");
        }
    }

    #[test]
    fn filterbank_has_expected_shape() {
        let config = MelConfig::whisper(128);
        let filters = mel_filterbank(&config);
        assert_eq!(filters.len(), 128);
        assert_eq!(filters[0].len(), 201);
        assert!(filters.iter().all(|row| row.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn spectrogram_frame_count_matches_torch() {
        let config = MelConfig::whisper(128);
        let samples = vec![0.0_f32; 16_000];
        let features = log_mel_spectrogram(&samples, &config).unwrap();
        // torch.stft(center=True) yields 1 + n // hop frames.
        assert_eq!(features.n_frames, 1 + 16_000 / 160);
        assert_eq!(features.data.len(), 128 * features.n_frames);
    }
}
