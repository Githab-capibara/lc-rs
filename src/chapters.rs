//! Chapter markers: persist metadata and slice a transcript by time.
//!
//! Mirrors `src/localcaption/chapters.py`. Chapters come from the yt-dlp
//! metadata (`YouTube` chapter markers); timed segments come from the
//! transcript artefacts themselves.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{IoContext, Result};
use crate::formatter;

/// A chapter marker with an optional end time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chapter {
    /// Human-readable chapter title.
    pub title: String,
    /// Chapter start, in seconds from the beginning of the media.
    pub start_time: f64,
    /// Chapter end, in seconds, when known.
    pub end_time: Option<f64>,
}

/// One timed piece of a transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// Segment start, in seconds.
    pub start: f64,
    /// Segment end, in seconds.
    pub end: f64,
    /// Segment text.
    pub text: String,
}

/// Suffixes recognised as transcript artefacts, longest first so that
/// `.chaptered.md` is not mistaken for `.md`.
const KNOWN_SUFFIXES: [&str; 7] = [
    ".chapters.json",
    ".chaptered.md",
    ".txt",
    ".json",
    ".srt",
    ".vtt",
    ".md",
];

/// Extract chapter markers from a yt-dlp info object.
pub fn chapters_from_info(info: Option<&Value>) -> Vec<Chapter> {
    let Some(info) = info else {
        return Vec::new();
    };
    let Some(raw) = info.get("chapters").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out: Vec<Chapter> = Vec::new();
    for (index, item) in raw.iter().enumerate() {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(start) = obj.get("start_time").and_then(value_as_f64) else {
            continue;
        };
        let title = obj
            .get("title")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map_or_else(|| format!("Chapter {}", index + 1), str::to_string);
        let end = obj.get("end_time").and_then(value_as_f64);
        out.push(Chapter {
            title,
            start_time: start,
            end_time: end,
        });
    }
    // Fill missing end times from the next chapter's start.
    let mut filled: Vec<Chapter> = Vec::with_capacity(out.len());
    for (index, chapter) in out.iter().enumerate() {
        let end = chapter
            .end_time
            .or_else(|| out.get(index + 1).map(|next| next.start_time));
        filled.push(Chapter {
            title: chapter.title.clone(),
            start_time: chapter.start_time,
            end_time: end,
        });
    }
    filled
}

/// Coerce a JSON value to `f64` when it is numeric.
fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Chapter list as plain JSON objects.
pub fn chapters_as_dicts(chapters: &[Chapter]) -> Vec<Value> {
    chapters
        .iter()
        .map(|c| {
            serde_json::json!({
                "title": c.title,
                "start_time": c.start_time,
                "end_time": c.end_time,
            })
        })
        .collect()
}

/// Render seconds as `MM:SS` or `H:MM:SS`.
pub fn format_timestamp(seconds: f64) -> String {
    let total = seconds.max(0.0) as i64;
    let hours = total / 3600;
    let rem = total % 3600;
    let minutes = rem / 60;
    let secs = rem % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes:02}:{secs:02}")
    }
}

/// Write the chapter list as JSON.
pub fn write_chapters_json(path: &Path, chapters: &[Chapter]) -> Result<PathBuf> {
    formatter::write_json(path, &chapters_as_dicts(chapters))
}

/// Strip a known transcript suffix from a path, keeping extra dots.
fn without_known_suffix(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let lower = name.to_lowercase();
    for suffix in KNOWN_SUFFIXES {
        if lower.ends_with(suffix) {
            let stem = &name[..name.len() - suffix.len()];
            return path.with_file_name(stem);
        }
    }
    path.to_path_buf()
}

/// Load timed segments from the sibling transcript `.json`, else `.srt`.
pub fn load_segments(transcript_path: &Path) -> Vec<Segment> {
    let base = without_known_suffix(transcript_path);
    let json_path = base.with_extension("json");
    let srt_path = base.with_extension("srt");
    if json_path.is_file()
        && let Ok(segs) = segments_from_json(&json_path)
        && !segs.is_empty()
    {
        return segs;
    }
    if srt_path.is_file()
        && let Ok(segs) = segments_from_srt(&srt_path)
    {
        return segs;
    }
    Vec::new()
}

/// Render `## MM:SS Title` headings with the transcript sliced underneath.
pub fn build_chaptered_markdown(
    chapters: &[Chapter],
    segments: &[Segment],
    fallback_text: &str,
) -> String {
    if chapters.is_empty() {
        return String::new();
    }
    let mut buckets: Vec<Vec<&str>> = vec![Vec::new(); chapters.len()];
    if !segments.is_empty() {
        for segment in segments {
            let index = chapter_index(chapters, segment.start);
            buckets[index].push(segment.text.as_str());
        }
    } else if !fallback_text.trim().is_empty() {
        buckets[0].push(fallback_text.trim());
    }

    let mut lines: Vec<String> = Vec::new();
    for (chapter, texts) in chapters.iter().zip(buckets.iter()) {
        lines.push(format!(
            "## {} {}",
            format_timestamp(chapter.start_time),
            chapter.title
        ));
        lines.push(String::new());
        if !texts.is_empty() {
            lines.push(texts.join("\n"));
            lines.push(String::new());
        }
    }
    let mut out = lines.join("\n");
    out = out.trim_end().to_string();
    out.push('\n');
    out
}

/// Write the chaptered Markdown transcript.
pub fn write_chaptered_md(
    path: &Path,
    chapters: &[Chapter],
    segments: &[Segment],
    fallback_text: &str,
) -> Result<PathBuf> {
    let content = build_chaptered_markdown(chapters, segments, fallback_text);
    std::fs::write(path, content).with_path(path)?;
    Ok(path.to_path_buf())
}

/// Index of the chapter containing time `t`.
fn chapter_index(chapters: &[Chapter], t: f64) -> usize {
    let mut index = 0usize;
    for (i, chapter) in chapters.iter().enumerate() {
        if chapter.start_time <= t {
            index = i;
        } else {
            break;
        }
    }
    index
}

/// Read timed segments out of a transcript JSON document.
fn segments_from_json(path: &Path) -> Result<Vec<Segment>> {
    let text = std::fs::read_to_string(path).with_path(path)?;
    let data: Value = serde_json::from_str(&text)?;
    let Some(obj) = data.as_object() else {
        return Ok(Vec::new());
    };
    let items = obj
        .get("transcription")
        .or_else(|| obj.get("segments"))
        .and_then(Value::as_array);
    let Some(items) = items else {
        return Ok(Vec::new());
    };
    let mut segs = Vec::new();
    for item in items {
        let Some(entry) = item.as_object() else {
            continue;
        };
        let text = entry
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        if text.is_empty() {
            continue;
        }
        let Some((start, end)) = segment_times(entry) else {
            continue;
        };
        segs.push(Segment {
            start,
            end,
            text: text.to_string(),
        });
    }
    Ok(segs)
}

/// Extract `(start, end)` from a segment object, supporting the three
/// shapes the Python implementation accepts.
fn segment_times(item: &serde_json::Map<String, Value>) -> Option<(f64, f64)> {
    if let Some(offsets) = item.get("offsets").and_then(Value::as_object) {
        let from = offsets.get("from").and_then(value_as_f64)?;
        let to = offsets.get("to").and_then(value_as_f64).unwrap_or(from);
        return Some((from / 1000.0, to / 1000.0));
    }
    if let Some(start) = item.get("start").and_then(value_as_f64) {
        let end = item.get("end").and_then(value_as_f64).unwrap_or(start);
        return Some((start, end));
    }
    let ts = item.get("timestamps").and_then(Value::as_object);
    let from = ts
        .and_then(|t| t.get("from"))
        .and_then(Value::as_str)
        .map_or(0.0, parse_clock);
    let to = ts
        .and_then(|t| t.get("to"))
        .and_then(Value::as_str)
        .map_or(0.0, parse_clock);
    Some((from, to))
}

/// Parse `HH:MM:SS,mmm` / `MM:SS.mmm` / plain seconds into seconds.
pub fn parse_clock(value: &str) -> f64 {
    let value = value.trim().replace(',', ".");
    let parts: Vec<&str> = value.split(':').collect();
    match parts.len() {
        3 => {
            let h: f64 = parts[0].parse().unwrap_or(0.0);
            let m: f64 = parts[1].parse().unwrap_or(0.0);
            let s: f64 = parts[2].parse().unwrap_or(0.0);
            h * 3600.0 + m * 60.0 + s
        }
        2 => {
            let m: f64 = parts[0].parse().unwrap_or(0.0);
            let s: f64 = parts[1].parse().unwrap_or(0.0);
            m * 60.0 + s
        }
        _ => value.parse().unwrap_or(0.0),
    }
}

/// Read timed segments out of an SRT file.
fn segments_from_srt(path: &Path) -> Result<Vec<Segment>> {
    let text = std::fs::read_to_string(path).with_path(path)?;
    let mut segs = Vec::new();
    for block in text.trim().split("\n\n") {
        let lines: Vec<&str> = block.lines().collect();
        if lines.is_empty() {
            continue;
        }
        let (ts_line, body_start) = if lines[0].contains("-->") {
            (lines[0], 1usize)
        } else if lines.len() > 1 {
            (lines[1], 2usize)
        } else {
            ("", 0)
        };
        let Some((start, end)) = parse_srt_timing(ts_line) else {
            continue;
        };
        let body = lines
            .get(body_start..)
            .unwrap_or_default()
            .join(" ")
            .trim()
            .to_string();
        if !body.is_empty() {
            segs.push(Segment {
                start,
                end,
                text: body,
            });
        }
    }
    Ok(segs)
}

/// Parse an SRT timing line into `(start, end)` seconds.
fn parse_srt_timing(line: &str) -> Option<(f64, f64)> {
    let arrow = line.find("-->")?;
    let left = line[..arrow].trim();
    let right = line[arrow + 3..].trim();
    Some((parse_clock(left), parse_clock(right)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn format_timestamp_switches_on_hours() {
        assert_eq!(format_timestamp(0.0), "00:00");
        assert_eq!(format_timestamp(65.0), "01:05");
        assert_eq!(format_timestamp(3665.0), "1:01:05");
        assert_eq!(format_timestamp(-3.0), "00:00");
    }

    #[test]
    fn chapters_from_info_fills_end_times() {
        let info = serde_json::json!({
            "chapters": [
                {"title": "Intro", "start_time": 0.0},
                {"title": "Main", "start_time": 10.5, "end_time": 20.0},
                {"start_time": 30.0}
            ]
        });
        let chapters = chapters_from_info(Some(&info));
        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].title, "Intro");
        assert_eq!(chapters[0].end_time, Some(10.5));
        assert_eq!(chapters[1].end_time, Some(20.0));
        assert_eq!(chapters[2].title, "Chapter 3");
        assert_eq!(chapters[2].end_time, None);
    }

    #[test]
    fn chaptered_markdown_slices_segments() {
        let chapters = vec![
            Chapter {
                title: "One".into(),
                start_time: 0.0,
                end_time: None,
            },
            Chapter {
                title: "Two".into(),
                start_time: 10.0,
                end_time: None,
            },
        ];
        let segments = vec![
            Segment {
                start: 1.0,
                end: 2.0,
                text: "first".into(),
            },
            Segment {
                start: 11.0,
                end: 12.0,
                text: "second".into(),
            },
        ];
        let md = build_chaptered_markdown(&chapters, &segments, "");
        assert!(md.starts_with("## 00:00 One\n\nfirst\n"));
        assert!(md.contains("## 00:10 Two\n\nsecond"));
    }

    #[test]
    fn load_segments_prefers_json() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("a.json"),
            r#"{"segments":[{"start":1.5,"end":2.5,"text":"hi"}]}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("a.srt"),
            "1\n00:00:09,000 --> 00:00:10,000\nnope\n\n",
        )
        .unwrap();
        let segs = load_segments(&dir.path().join("a.md"));
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hi");
        assert!((segs[0].start - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn load_segments_falls_back_to_srt() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("b.srt"),
            "1\n00:00:01,000 --> 00:00:02,500\nhello\n\n",
        )
        .unwrap();
        let segs = load_segments(&dir.path().join("b.md"));
        assert_eq!(segs.len(), 1);
        assert!((segs[0].end - 2.5).abs() < 1e-9);
    }

    #[test]
    fn parse_clock_handles_variants() {
        assert!((parse_clock("00:00:01,500") - 1.5).abs() < 1e-9);
        assert!((parse_clock("01:30") - 90.0).abs() < 1e-9);
        assert!((parse_clock("2.5") - 2.5).abs() < 1e-9);
        assert!((parse_clock("nope") - 0.0).abs() < 1e-9);
    }
}
