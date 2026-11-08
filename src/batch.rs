//! Batch transcription: many URLs (or local files) → one summary table.
//!
//! Sources are processed sequentially. A single model already saturates the
//! CPU, so a worker pool would only add contention.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::errors::{Error, Result};
use crate::pipeline::{self, PipelineOptions};

/// Outcome of one source in a batch run.
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// The source that was transcribed.
    pub source: String,
    /// Stable identifier used for the output directory.
    pub video_id: String,
    /// `ok`, `failed` or `skipped`.
    pub status: String,
    /// Media duration in seconds, when known.
    pub duration_s: Option<f64>,
    /// Wall-clock time spent on this item.
    pub elapsed_s: f64,
    /// Error message for failed items.
    pub error: Option<String>,
}

/// Aggregated result of a batch run.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// One entry per source.
    pub items: Vec<BatchItem>,
    /// Total wall-clock time.
    pub wall_clock_s: f64,
}

impl BatchResult {
    /// Process exit code: non-zero when any item failed.
    pub fn exit_code(&self) -> i32 {
        i32::from(self.items.iter().any(|i| i.status == "failed"))
    }

    /// Human-readable summary table.
    pub fn summary(&self) -> String {
        let id_width = self
            .items
            .iter()
            .map(|i| i.video_id.chars().count())
            .max()
            .unwrap_or(0);
        let dur_width = self
            .items
            .iter()
            .map(|i| fmt_hms(i.duration_s).chars().count())
            .max()
            .unwrap_or(2);
        let mut lines = Vec::new();
        for item in &self.items {
            let mark = match item.status.as_str() {
                "ok" => "✅",
                "failed" => "❌",
                _ => "⏭",
            };
            let tail = match item.status.as_str() {
                "ok" => fmt_elapsed(item.elapsed_s),
                "skipped" => "skipped".to_string(),
                _ => item.error.clone().unwrap_or_else(|| "failed".to_string()),
            };
            lines.push(format!(
                "{mark} {:<id_width$}  {:>dur_width$}  {tail}",
                item.video_id,
                fmt_hms(item.duration_s)
            ));
        }
        lines.push("--------------------------------".to_string());
        let ok = self.items.iter().filter(|i| i.status == "ok").count();
        let failed = self.items.iter().filter(|i| i.status == "failed").count();
        let skipped = self.items.iter().filter(|i| i.status == "skipped").count();
        let mut parts = vec![format!("total: {}", self.items.len()), format!("ok: {ok}")];
        if skipped > 0 {
            parts.push(format!("skipped: {skipped}"));
        }
        parts.push(format!("failed: {failed}"));
        parts.push(format!("{} wall-clock", fmt_elapsed(self.wall_clock_s)));
        lines.push(parts.join(" · "));
        lines.join("\n")
    }
}

/// Read sources from a list file: one per line, skipping blanks and `#`
/// comments.
///
/// Local paths expand `~`. Relative local paths are resolved against the
/// list file's directory (not the process working directory), so a queue
/// sitting next to its media files still works when an absolute `--batch`
/// path is given.
pub fn read_url_list(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    let content = content.strip_prefix('\u{feff}').unwrap_or(&content);
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut urls = Vec::new();
    for line in content.lines() {
        let stripped = line.trim();
        if stripped.is_empty() || stripped.starts_with('#') {
            continue;
        }
        urls.push(normalize_local_source(stripped, base));
    }
    Ok(urls)
}

/// Stable id used for `<out>/<id>/<id>.md`.
///
/// `YouTube` URLs use the 11-character video id. Local files use a sanitised
/// path so two `episode.mp3` files in different folders do not share a
/// directory. Other URLs use host plus path so two sites cannot clobber each
/// other.
pub fn video_id_for(source: &str) -> String {
    if !source.contains("://") {
        return safe_id(&expand_tilde(Path::new(source)).to_string_lossy());
    }
    if let Some(id) = youtube_id(source) {
        return id;
    }
    let parsed = url::Url::parse(source).ok();
    let host = parsed
        .as_ref()
        .and_then(|u| u.host_str())
        .unwrap_or("url")
        .trim_start_matches("www.")
        .to_string();
    let tail = parsed
        .as_ref()
        .map(|u| {
            u.path_segments()
                .map(|segments| {
                    segments
                        .filter(|s| !s.is_empty())
                        .map(percent_decode)
                        .collect::<Vec<_>>()
                        .join("_")
                })
                .unwrap_or_default()
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "video".to_string());
    safe_id(&format!("{host}_{tail}"))
}

/// Extract the `YouTube` video id from a URL, when there is one.
pub fn youtube_id(source: &str) -> Option<String> {
    if let Some(rest) = source.split("youtu.be/").nth(1) {
        let id: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .collect();
        if id.len() == 11 {
            return Some(id);
        }
    }
    let marker = "youtube.com/";
    let rest = &source[source.find(marker)? + marker.len()..];
    for prefix in ["shorts/", "embed/", "live/", "v/"] {
        if let Some(tail) = rest.strip_prefix(prefix) {
            let id: String = tail
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .collect();
            if id.len() == 11 {
                return Some(id);
            }
        }
    }
    let tail = rest.strip_prefix("watch?")?;
    for param in tail.split(['&', '#']) {
        if let Some(value) = param.strip_prefix("v=") {
            let id: String = value
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                .collect();
            if id.len() == 11 {
                return Some(id);
            }
        }
    }
    None
}

/// Percent-decode a URL path segment.
fn percent_decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
            if let Ok(value) = u8::from_str_radix(hex, 16) {
                out.push(value);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Expand a leading `~` in a path.
pub fn expand_tilde(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if let Some(rest) = raw.strip_prefix('~')
        && (rest.is_empty() || rest.starts_with('/'))
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest.trim_start_matches('/'));
    }
    path.to_path_buf()
}

/// Make a relative local path relative to `base`.
fn normalize_local_source(source: &str, base: &Path) -> String {
    if source.contains("://") {
        return source.to_string();
    }
    let path = expand_tilde(Path::new(source));
    if path.is_absolute() {
        return path.to_string_lossy().into_owned();
    }
    base.join(path).to_string_lossy().into_owned()
}

/// Sanitise a string into a safe directory name.
fn safe_id(raw: &str) -> String {
    let mut cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    cleaned = cleaned.trim_matches(|c| c == '.' || c == '_').to_string();
    if cleaned.is_empty() {
        return "video".to_string();
    }
    if cleaned.chars().count() <= 120 {
        return cleaned;
    }
    // Keep the tail so long temporary prefixes do not drop the filename.
    let tail: String = cleaned
        .chars()
        .skip(cleaned.chars().count() - 120)
        .collect();
    let tail = tail.trim_matches(['.', '_']).to_string();
    if tail.is_empty() {
        "video".to_string()
    } else {
        tail
    }
}

/// Format a duration as `M:SS` or `H:MM:SS`.
pub fn fmt_hms(seconds: Option<f64>) -> String {
    let Some(seconds) = seconds else {
        return "--".to_string();
    };
    let total = seconds.max(0.0).round() as i64;
    let hours = total / 3600;
    let rem = total % 3600;
    let minutes = rem / 60;
    let secs = rem % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes}:{secs:02}")
    }
}

/// Format an elapsed time as `12s`, `3m 4s` or `1h 2m`.
pub fn fmt_elapsed(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as i64;
    if total < 60 {
        return format!("{total}s");
    }
    let mut minutes = total / 60;
    let secs = total % 60;
    if minutes < 60 {
        return if secs == 0 {
            format!("{minutes}m")
        } else {
            format!("{minutes}m {secs}s")
        };
    }
    let hours = minutes / 60;
    minutes %= 60;
    if minutes == 0 {
        format!("{hours}h")
    } else {
        format!("{hours}h {minutes}m")
    }
}

/// File name whose presence marks an item as already done.
///
/// Mirrors the output-format resolution: a single-format run writes only that
/// format's extension, while `all`/`full` writes `.md` first and can use it
/// as the marker.
pub fn skip_marker_name(video_id: &str, output_format: &str) -> String {
    match output_format.trim().to_lowercase().as_str() {
        "txt" | "srt" | "vtt" | "json" => format!("{video_id}.{output_format}"),
        _ => format!("{video_id}.md"),
    }
}

/// Transcribe each source sequentially.
///
/// Per-source output goes to `out_dir/<videoId>/`. If the marker file already
/// exists, that source is skipped. Failures are recorded and the rest of the
/// list still runs.
pub async fn transcribe_urls(urls: &[String], options: &PipelineOptions) -> Result<BatchResult> {
    let out_dir = options.out_dir.clone();
    crate::paths::ensure_dir(&out_dir)?;
    let mut items = Vec::new();
    let wall_start = Instant::now();

    for url in urls {
        items.push(transcribe_one(url, options).await);
    }
    Ok(BatchResult {
        items,
        wall_clock_s: wall_start.elapsed().as_secs_f64(),
    })
}

/// Transcribe one source of a batch run.
async fn transcribe_one(url: &str, options: &PipelineOptions) -> BatchItem {
    let source = if url.contains("://") {
        url.to_string()
    } else {
        expand_tilde(Path::new(url)).to_string_lossy().into_owned()
    };
    let video_id = video_id_for(&source);
    let item_dir = options.out_dir.join(&video_id);
    let marker = item_dir.join(skip_marker_name(&video_id, &options.output_format));
    if marker.is_file() {
        tracing::info!("skip {video_id}: already exists ({})", marker.display());
        return BatchItem {
            source,
            video_id,
            status: "skipped".to_string(),
            duration_s: None,
            elapsed_s: 0.0,
            error: None,
        };
    }

    let started = Instant::now();
    let mut item_options = options.clone();
    item_options.out_dir = item_dir;
    item_options.stem = Some(video_id.clone());
    match pipeline::transcribe_url(&source, &item_options).await {
        Ok(result) => BatchItem {
            source,
            video_id,
            status: "ok".to_string(),
            duration_s: result.duration_s,
            elapsed_s: started.elapsed().as_secs_f64(),
            error: None,
        },
        Err(err) => {
            let elapsed = started.elapsed().as_secs_f64();
            let message = format_error(&err);
            tracing::error!("{video_id}: {message}");
            BatchItem {
                source,
                video_id,
                status: "failed".to_string(),
                duration_s: None,
                elapsed_s: elapsed,
                error: Some(message),
            }
        }
    }
}

/// Render an error as a single line.
fn format_error(err: &Error) -> String {
    let message = err.to_string();
    let first = message.lines().next().unwrap_or_default().trim();
    if first.is_empty() {
        "unknown error".to_string()
    } else {
        first.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn reads_list_and_skips_comments() {
        let dir = tempdir().unwrap();
        let list = dir.path().join("queue.txt");
        std::fs::write(
            &list,
            "# a comment\n\nhttps://example.com/a\n  \nrelative/file.mp3\n",
        )
        .unwrap();
        let urls = read_url_list(&list).unwrap();
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://example.com/a");
        assert!(urls[1].ends_with("relative/file.mp3"));
        assert!(Path::new(&urls[1]).is_absolute());
    }

    #[test]
    fn extracts_youtube_ids() {
        assert_eq!(
            youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
        assert_eq!(
            youtube_id("https://youtu.be/dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
        assert_eq!(
            youtube_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
        assert_eq!(
            youtube_id("https://www.youtube.com/watch?list=PL123&v=dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
        assert_eq!(youtube_id("https://example.com/video"), None);
    }

    #[test]
    fn video_id_for_local_paths_is_sanitised() {
        let id = video_id_for("/tmp/my folder/episode 1.mp3");
        assert!(!id.contains('/'));
        assert!(id.contains('_'));
    }

    #[test]
    fn video_id_for_other_urls_uses_host_and_path() {
        let id = video_id_for("https://vimeo.com/12345/clip");
        assert!(id.starts_with("vimeo.com_"));
    }

    #[test]
    fn formats_durations() {
        assert_eq!(fmt_hms(Some(0.0)), "0:00");
        assert_eq!(fmt_hms(Some(65.0)), "1:05");
        assert_eq!(fmt_hms(Some(3665.0)), "1:01:05");
        assert_eq!(fmt_hms(None), "--");
    }

    #[test]
    fn formats_elapsed() {
        assert_eq!(fmt_elapsed(12.0), "12s");
        assert_eq!(fmt_elapsed(184.0), "3m 4s");
        assert_eq!(fmt_elapsed(180.0), "3m");
        assert_eq!(fmt_elapsed(3720.0), "1h 2m");
        assert_eq!(fmt_elapsed(3600.0), "1h");
    }

    #[test]
    fn skip_marker_follows_output_format() {
        assert_eq!(skip_marker_name("abc", "md"), "abc.md");
        assert_eq!(skip_marker_name("abc", "txt"), "abc.txt");
        assert_eq!(skip_marker_name("abc", "all"), "abc.md");
    }
}
