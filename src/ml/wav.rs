//! Minimal RIFF/WAVE reader.
//!
//! The models all expect 16 kHz mono audio, so the parser focuses on that
//! shape while still accepting the common variants (8/16/24/32-bit PCM and
//! 32-bit float, any channel count) so a hand-made fixture does not need a
//! conversion step first.

use std::path::Path;

use crate::errors::{Error, Result};

/// Sample rate expected by every model in the registry.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Decoded audio.
#[derive(Debug, Clone)]
pub struct Audio {
    /// Mono samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels in the source file.
    pub channels: u16,
}

impl Audio {
    /// Duration of the audio in seconds.
    pub fn duration_s(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
}

/// Parse a WAV file into mono float samples.
pub fn read_wav(path: &Path) -> Result<Audio> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    parse_wav(&bytes)
        .map_err(|message| Error::AudioConversion(format!("{}: {message}", path.display())))
}

/// Parse a WAV byte buffer into mono float samples.
pub fn parse_wav(bytes: &[u8]) -> std::result::Result<Audio, String> {
    if bytes.len() < 12 {
        return Err("file is too short to be a WAV".to_string());
    }
    if &bytes[0..4] != b"RIFF" {
        return Err("missing RIFF header".to_string());
    }
    if &bytes[8..12] != b"WAVE" {
        return Err("missing WAVE header".to_string());
    }

    let mut offset = 12usize;
    let mut format_tag: u16 = 0;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data: Option<&[u8]> = None;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        let body_start = offset + 8;
        let body_end = body_start.saturating_add(chunk_size).min(bytes.len());
        let body = &bytes[body_start..body_end];

        match chunk_id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err("truncated fmt chunk".to_string());
                }
                format_tag = u16::from_le_bytes([body[0], body[1]]);
                channels = u16::from_le_bytes([body[2], body[3]]);
                sample_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                bits_per_sample = u16::from_le_bytes([body[14], body[15]]);
            }
            b"data" => {
                data = Some(body);
            }
            _ => {}
        }

        // Chunks are word-aligned.
        offset = body_start + chunk_size + (chunk_size & 1);
        if chunk_size == 0 {
            break;
        }
    }

    let data = data.ok_or_else(|| "no data chunk".to_string())?;
    if channels == 0 {
        return Err("channel count is zero".to_string());
    }
    if sample_rate == 0 {
        return Err("sample rate is zero".to_string());
    }

    let samples = decode_samples(data, format_tag, bits_per_sample, channels)?;
    Ok(Audio {
        samples,
        sample_rate,
        channels,
    })
}

/// Decode interleaved sample data into mono float samples.
fn decode_samples(
    data: &[u8],
    format_tag: u16,
    bits_per_sample: u16,
    channels: u16,
) -> std::result::Result<Vec<f32>, String> {
    let bytes_per_sample = usize::from(bits_per_sample / 8);
    if bytes_per_sample == 0 {
        return Err("unsupported bit depth".to_string());
    }
    let frame_size = bytes_per_sample * usize::from(channels);
    if frame_size == 0 {
        return Err("invalid frame size".to_string());
    }
    let frames = data.len() / frame_size;
    let mut mono = Vec::with_capacity(frames);

    for frame in 0..frames {
        let base = frame * frame_size;
        let mut sum = 0.0f32;
        for channel in 0..usize::from(channels) {
            let start = base + channel * bytes_per_sample;
            let value = decode_sample(
                &data[start..start + bytes_per_sample],
                format_tag,
                bits_per_sample,
            )?;
            sum += value;
        }
        mono.push(sum / f32::from(channels));
    }
    Ok(mono)
}

/// Decode one sample of the given bit depth.
fn decode_sample(
    bytes: &[u8],
    format_tag: u16,
    bits_per_sample: u16,
) -> std::result::Result<f32, String> {
    // 0xFFFE is WAVE_FORMAT_EXTENSIBLE; only plain IEEE float (tag 3) is
    // decoded as float here.
    let is_float = format_tag == 3;
    match bits_per_sample {
        8 => Ok((f32::from(bytes[0]) - 128.0) / 128.0),
        16 => {
            let raw = i16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(f32::from(raw) / 32768.0)
        }
        24 => {
            let raw = i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]);
            Ok(raw as f32 / 8_388_608.0)
        }
        32 => {
            if is_float {
                let raw = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                Ok(raw)
            } else {
                let raw = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                Ok(raw as f32 / 2_147_483_648.0)
            }
        }
        _ => Err(format!("unsupported bit depth {bits_per_sample}")),
    }
}

/// Build a 16 kHz mono PCM WAV buffer from float samples.
///
/// Used by the test-suite and by the fixture generator.
pub fn encode_wav16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let value = (clamped * 32767.0).round() as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_pcm16() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode_wav16(&samples, 16_000);
        let audio = parse_wav(&bytes).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 5);
        assert!((audio.samples[1] - 0.5).abs() < 1e-4);
        assert!((audio.samples[4] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn rejects_non_riff() {
        assert!(parse_wav(b"not a wav file at all").is_err());
    }

    #[test]
    fn averages_channels() {
        let mut bytes = encode_wav16(&[0.0, 1.0], 16_000);
        // Rewrite the channel count to two: the four mono sample bytes now
        // describe one interleaved stereo frame.
        bytes[22] = 2;
        bytes[23] = 0;
        let audio = parse_wav(&bytes).unwrap();
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples.len(), 1);
        assert!((audio.samples[0] - 0.5).abs() < 1e-4);
    }
}
