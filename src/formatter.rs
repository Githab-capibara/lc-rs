//! Transcript serialisation: Markdown, plain text, SRT, `WebVTT` and JSON.
//!
//! The byte-level output mirrors `src/localcaption/formats.py` of the Python
//! implementation exactly, including the clock format (`HH:MM:SS,mmm` for
//! SRT, `HH:MM:SS.mmm` for VTT) and the trailing-newline rules.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::chapters::Segment;
use crate::errors::{IoContext, Result};

/// Render seconds as a subtitle clock.
///
/// SRT uses `HH:MM:SS,mmm`, `WebVTT` uses `HH:MM:SS.mmm`. Negative values are
/// clamped to zero.
pub fn clock(seconds: f64, vtt: bool) -> String {
    let ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let hours = ms / 3_600_000;
    let ms = ms % 3_600_000;
    let minutes = ms / 60_000;
    let ms = ms % 60_000;
    let secs = ms / 1000;
    let ms = ms % 1000;
    let sep = if vtt { "." } else { "," };
    format!("{hours:02}:{minutes:02}:{secs:02}{sep}{ms:03}")
}

/// Write plain text (trailing whitespace stripped, one trailing newline).
pub fn write_txt(path: &Path, text: &str) -> Result<PathBuf> {
    let body = text.trim_end();
    let content = if body.is_empty() {
        String::new()
    } else {
        format!("{body}\n")
    };
    std::fs::write(path, content).with_path(path)?;
    Ok(path.to_path_buf())
}

/// Write a Markdown transcript.
///
/// The transcript body is followed by a `## Timestamps` section when timed
/// segments are available.
pub fn write_md(path: &Path, text: &str, segments: &[Segment]) -> Result<PathBuf> {
    let body = text.trim_end();
    let mut lines: Vec<String> = Vec::new();
    if !body.is_empty() {
        lines.push(body.to_string());
        lines.push(String::new());
    }
    let timed: Vec<&Segment> = segments
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .collect();
    if !timed.is_empty() {
        lines.push("## Timestamps".to_string());
        lines.push(String::new());
        for segment in timed {
            lines.push(format!(
                "- `{}` — {}",
                clock(segment.start, false),
                segment.text.trim()
            ));
        }
        lines.push(String::new());
    }
    let mut content = lines.join("\n");
    content = content.trim_end().to_string();
    if !lines.is_empty() {
        content.push('\n');
    }
    std::fs::write(path, content).with_path(path)?;
    Ok(path.to_path_buf())
}

/// Write a `SubRip` (`.srt`) subtitle file.
pub fn write_srt(path: &Path, segments: &[Segment]) -> Result<PathBuf> {
    let mut lines: Vec<String> = Vec::new();
    let mut index = 0usize;
    for segment in segments {
        if segment.text.trim().is_empty() {
            continue;
        }
        index += 1;
        lines.push(index.to_string());
        lines.push(format!(
            "{} --> {}",
            clock(segment.start, false),
            clock(segment.end, false)
        ));
        lines.push(segment.text.trim().to_string());
        lines.push(String::new());
    }
    std::fs::write(path, lines.join("\n")).with_path(path)?;
    Ok(path.to_path_buf())
}

/// Write a `WebVTT` (`.vtt`) subtitle file.
pub fn write_vtt(path: &Path, segments: &[Segment]) -> Result<PathBuf> {
    let mut lines: Vec<String> = vec!["WEBVTT".to_string(), String::new()];
    for segment in segments {
        if segment.text.trim().is_empty() {
            continue;
        }
        lines.push(format!(
            "{} --> {}",
            clock(segment.start, true),
            clock(segment.end, true)
        ));
        lines.push(segment.text.trim().to_string());
        lines.push(String::new());
    }
    std::fs::write(path, lines.join("\n")).with_path(path)?;
    Ok(path.to_path_buf())
}

/// Write a pretty-printed JSON document.
pub fn write_json<T: Serialize>(path: &Path, payload: &T) -> Result<PathBuf> {
    let mut content = serde_json::to_string_pretty(payload)?;
    content.push('\n');
    std::fs::write(path, content).with_path(path)?;
    Ok(path.to_path_buf())
}

/// Append a suffix to a filename without `Path::with_suffix` stripping extra
/// dots from the stem (`lecture.2024` must stay `lecture.2024.txt`).
pub fn output_file(out_basename: &Path, suffix: &str) -> PathBuf {
    crate::paths::append_suffix(out_basename, suffix)
}

/// Ensure the parent directory of `path` exists.
pub fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_path(parent)?;
    }
    Ok(())
}

/// Read a file, mapping IO failures onto [`Error::Io`].
pub fn read_to_string(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn seg(start: f64, end: f64, text: &str) -> Segment {
        Segment {
            start,
            end,
            text: text.to_string(),
        }
    }

    #[test]
    fn clock_uses_comma_for_srt() {
        assert_eq!(clock(0.0, false), "00:00:00,000");
        assert_eq!(clock(3661.5, false), "01:01:01,500");
        assert_eq!(clock(-1.0, false), "00:00:00,000");
    }

    #[test]
    fn clock_uses_dot_for_vtt() {
        assert_eq!(clock(1.25, true), "00:00:01,250".replace(',', "."));
    }

    #[test]
    fn md_has_timestamps_section() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.md");
        write_md(
            &path,
            "Hello world",
            &[seg(0.0, 1.0, "Hello"), seg(1.0, 2.0, "  ")],
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text,
            "Hello world\n\n## Timestamps\n\n- `00:00:00,000` — Hello\n"
        );
    }

    #[test]
    fn md_without_segments_is_plain() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.md");
        write_md(&path, "Only body", &[]).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "Only body\n");
    }

    #[test]
    fn srt_numbers_and_separates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.srt");
        write_srt(
            &path,
            &[
                seg(0.0, 1.5, "one"),
                seg(1.5, 2.0, "two"),
                seg(2.0, 3.0, "   "),
            ],
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        // Python `write_srt` joins the lines with "\n": the trailing empty
        // element contributes none, so the file ends with ONE newline.
        assert_eq!(
            text,
            "1\n00:00:00,000 --> 00:00:01,500\none\n\n2\n00:00:01,500 --> 00:00:02,000\ntwo\n"
        );
    }

    #[test]
    fn vtt_has_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.vtt");
        write_vtt(&path, &[seg(0.0, 1.0, "one")]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("WEBVTT\n\n"));
        assert!(text.contains("00:00:00.000 --> 00:00:01.000"));
    }

    #[test]
    fn txt_strips_trailing_whitespace() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.txt");
        write_txt(&path, "line  \n\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "line\n");
    }
}
