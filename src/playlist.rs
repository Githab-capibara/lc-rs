//! Playlist expansion.
//!
//! A URL that carries a `list=` (or `playlist=`) query parameter points at a
//! playlist rather than a single video. The downloader always asks `yt-dlp`
//! for a single file, so the rest of the list would be dropped silently.
//! This module detects the situation up front and expands the list to
//! individual video URLs.
//!
//! The expansion is a metadata-only round-trip (`--flat-playlist`), so it is
//! fast even over a Tor proxy.

use std::time::Duration;

use serde_json::Value;

use crate::config;
use crate::downloader::{self, CookieSource};
use crate::errors::{Error, Result};
use crate::proc;

/// Hard timeout for one playlist expansion.
const EXPANSION_TIMEOUT: Duration = Duration::from_mins(10);

/// The result of expanding one playlist URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaylistInfo {
    /// The playlist id, when the URL carried one.
    pub playlist_id: String,
    /// The playlist title, when known.
    pub title: Option<String>,
    /// Number of entries in the playlist.
    pub entry_count: usize,
    /// The individual video URLs.
    pub urls: Vec<String>,
}

/// Whether `url` carries a `list=` (or `playlist=`) query parameter.
pub fn has_playlist_query(url: &str) -> bool {
    playlist_query_id(url).is_some()
}

/// The playlist id carried by `url`, if any.
pub fn playlist_query_id(url: &str) -> Option<String> {
    if !url.contains("://") {
        return None;
    }
    let parsed = url::Url::parse(url).ok()?;
    for key in ["list", "playlist"] {
        for (name, value) in parsed.query_pairs() {
            if name == key && !value.trim().is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

/// Expand a playlist URL to its individual video URLs.
pub async fn expand_playlist(
    url: &str,
    proxy: Option<&str>,
    cookies: Option<&str>,
) -> Result<PlaylistInfo> {
    downloader::validate_source(url)?;
    let program = downloader::yt_dlp_path()?;
    let resolved_proxy = match proxy {
        Some(value) => value.to_string(),
        None => config::proxy(),
    };
    let cookie_source = match cookies {
        Some(raw) => Some(CookieSource::parse(raw)),
        None => downloader::detect_cookies(),
    };

    let mut args: Vec<String> = vec![
        "--flat-playlist".to_string(),
        "--dump-single-json".to_string(),
        "--skip-download".to_string(),
        "--no-warnings".to_string(),
        "--quiet".to_string(),
    ];
    if !resolved_proxy.is_empty() {
        args.push("--proxy".to_string());
        args.push(resolved_proxy);
    }
    match cookie_source {
        Some(CookieSource::File(path)) => {
            args.push("--cookies".to_string());
            args.push(path.to_string_lossy().into_owned());
        }
        Some(CookieSource::Browser(name)) => {
            args.push("--cookies-from-browser".to_string());
            args.push(name);
        }
        None => {}
    }
    args.push("--".to_string());
    args.push(url.to_string());

    let output = proc::run_with_timeout(
        &program,
        &args,
        EXPANSION_TIMEOUT,
        "yt-dlp (playlist expansion)",
    )
    .await?;

    if !output.success() {
        let detail = output.stderr.trim();
        return Err(Error::Playlist(format!(
            "failed to expand playlist {url:?}: {}",
            if detail.is_empty() {
                format!("yt-dlp exited with code {:?}", output.code)
            } else {
                detail.to_string()
            }
        )));
    }

    let info: Value = serde_json::from_str(output.stdout.trim())
        .map_err(|e| Error::Playlist(format!("could not parse the playlist metadata: {e}")))?;

    let entries = info
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Error::Playlist(format!(
                "URL is not a playlist (no entries returned): {url}"
            ))
        })?;

    let mut urls = Vec::new();
    for entry in entries {
        let Some(entry) = entry.as_object() else {
            continue;
        };
        let candidate = ["webpage_url", "original_url", "url"]
            .iter()
            .filter_map(|key| entry.get(*key).and_then(Value::as_str))
            .find(|value| !value.trim().is_empty());
        let Some(candidate) = candidate else {
            continue;
        };
        urls.push(candidate.to_string());
    }

    if urls.is_empty() {
        return Err(Error::Playlist(format!(
            "playlist {url} resolved to zero videos"
        )));
    }

    let title = info
        .get("title")
        .and_then(Value::as_str)
        .map(str::to_string);
    tracing::info!(
        "playlist: {} videos{}",
        urls.len(),
        title
            .as_ref()
            .map(|t| format!(" ({t})"))
            .unwrap_or_default()
    );

    Ok(PlaylistInfo {
        playlist_id: playlist_query_id(url)
            .or_else(|| info.get("id").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default(),
        title,
        entry_count: urls.len(),
        urls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_playlist_queries() {
        assert!(has_playlist_query(
            "https://www.youtube.com/watch?v=abc&list=PL123"
        ));
        assert!(has_playlist_query("https://example.com/feed?playlist=xyz"));
        assert!(!has_playlist_query("https://www.youtube.com/watch?v=abc"));
        assert!(!has_playlist_query("/tmp/local.mp4"));
        assert!(!has_playlist_query("not a url"));
    }

    #[test]
    fn extracts_playlist_ids() {
        assert_eq!(
            playlist_query_id("https://www.youtube.com/watch?v=abc&list=PL123"),
            Some("PL123".to_string())
        );
        assert_eq!(
            playlist_query_id("https://example.com/feed?playlist=xyz"),
            Some("xyz".to_string())
        );
        assert_eq!(playlist_query_id("https://example.com/feed"), None);
    }
}
