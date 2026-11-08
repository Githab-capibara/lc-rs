//! Stage 2: re-encode arbitrary audio to 16 kHz mono PCM WAV.
//!
//! All models expect 16 kHz mono input, so this stage is a hard requirement
//! even when the source is already a `.wav` (the sample rate or the channel
//! count may not match).
//!
//! `ffmpeg` is intentionally *not* routed through the proxy: it only touches
//! local files.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::errors::{Error, Result};
use crate::ml::wav;
use crate::proc;

/// Sample rate every model expects.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Channel count every model expects.
pub const TARGET_CHANNELS: u32 = 1;

/// Hard timeout for one `ffmpeg` invocation.
const FFMPEG_TIMEOUT: Duration = Duration::from_mins(60);

/// Convert `src` to a 16 kHz mono PCM WAV at `dst`.
pub async fn to_wav(src: &Path, dst: &Path) -> Result<PathBuf> {
    let program = proc::require_executable(
        "ffmpeg",
        "On macOS: `brew install ffmpeg`; on Debian/Ubuntu: `sudo apt install ffmpeg`.",
    )?;
    if let Some(parent) = dst.parent() {
        crate::paths::ensure_dir(parent)?;
    }

    let args: Vec<String> = vec![
        "-y".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-i".to_string(),
        src.to_string_lossy().into_owned(),
        "-ac".to_string(),
        TARGET_CHANNELS.to_string(),
        "-ar".to_string(),
        TARGET_SAMPLE_RATE.to_string(),
        "-vn".to_string(),
        "-c:a".to_string(),
        "pcm_s16le".to_string(),
        dst.to_string_lossy().into_owned(),
    ];

    tracing::info!("ffmpeg: re-encoding to {TARGET_SAMPLE_RATE} Hz mono WAV");
    let output = proc::run_with_timeout(&program, &args, FFMPEG_TIMEOUT, "ffmpeg").await?;
    if !output.success() {
        let stderr = output.stderr.trim();
        return Err(Error::AudioConversion(format!(
            "ffmpeg failed (exit {}) while converting {}{}",
            output.code.unwrap_or(-1),
            src.display(),
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        )));
    }
    if !dst.is_file() {
        return Err(Error::AudioConversion(format!(
            "ffmpeg ran but {} was not created",
            dst.display()
        )));
    }
    // Validate the produced header so a broken conversion never reaches the
    // model stage.
    let audio = wav::read_wav(dst)?;
    if audio.sample_rate != TARGET_SAMPLE_RATE || audio.channels != 1 {
        return Err(Error::AudioConversion(format!(
            "ffmpeg produced {} Hz / {} channels, expected {TARGET_SAMPLE_RATE} Hz mono",
            audio.sample_rate, audio.channels
        )));
    }
    Ok(dst.to_path_buf())
}

/// Duration of a WAV file in seconds, when it can be determined.
pub fn wav_duration_s(path: &Path) -> Option<f64> {
    wav::read_wav(path).ok().map(|audio| audio.duration_s())
}

/// Read a WAV file into mono float samples.
pub fn read_samples(path: &Path) -> Result<Vec<f32>> {
    Ok(wav::read_wav(path)?.samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn converts_to_mono_16k() {
        if proc::find_executable("ffmpeg").is_none() {
            return;
        }
        let dir = tempdir().unwrap();
        let src = dir.path().join("src.wav");
        let dst = dir.path().join("dst.wav");
        // 8 kHz stereo input.
        let samples: Vec<f32> = (0..8000).map(|i| (i as f32 / 8000.0).sin()).collect();
        std::fs::write(&src, wav::encode_wav16(&samples, 8_000)).unwrap();
        to_wav(&src, &dst).await.unwrap();
        let audio = wav::read_wav(&dst).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert!((audio.duration_s() - 1.0).abs() < 0.01);
    }
}
