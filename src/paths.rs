//! Filesystem layout, mirroring ``src/localcaption/paths.py``.
//!
//! Two roots, both overridable by environment variables:
//!
//! * ``models/`` — one directory per model checkpoint. ``LCRS_MODELS_DIR``
//!   overrides; defaults to the platform data dir + ``lc-rs/models``.
//! * the search index lives in the platform data dir + ``lc-rs``.

use std::path::{Path, PathBuf};

use crate::errors::{Error, IoContext, Result};

/// Environment variable overriding the model root.
pub const MODELS_DIR_ENV: &str = "LCRS_MODELS_DIR";
/// Environment variable overriding the index location.
pub const INDEX_PATH_ENV: &str = "LCRS_INDEX_PATH";
/// Environment variable overriding the data root.
pub const DATA_HOME_ENV: &str = "XDG_DATA_HOME";
/// Environment variable overriding the config root.
pub const CONFIG_HOME_ENV: &str = "XDG_CONFIG_HOME";
/// Environment variable overriding the cache root.
pub const CACHE_HOME_ENV: &str = "XDG_CACHE_HOME";

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("/tmp"), PathBuf::from)
}

/// Platform data root (``~/.local/share`` on Linux, ``~/Library`` on macOS).
pub fn data_root() -> PathBuf {
    if let Some(p) = env_path(DATA_HOME_ENV) {
        return p;
    }
    if cfg!(target_os = "macos") {
        return home().join("Library").join("Application Support");
    }
    home().join(".local").join("share")
}

/// Platform config root.
pub fn config_root() -> PathBuf {
    if let Some(p) = env_path(CONFIG_HOME_ENV) {
        return p;
    }
    if cfg!(target_os = "macos") {
        return home().join("Library").join("Preferences");
    }
    home().join(".config")
}

/// Platform cache root.
pub fn cache_root() -> PathBuf {
    if let Some(p) = env_path(CACHE_HOME_ENV) {
        return p;
    }
    if cfg!(target_os = "macos") {
        return home().join("Library").join("Caches");
    }
    home().join(".cache")
}

/// Root directory holding every model checkpoint.
pub fn models_root() -> PathBuf {
    crate::config::models_dir()
}

/// Location of the transcript search index.
pub fn index_path() -> PathBuf {
    if let Some(p) = env_path(INDEX_PATH_ENV) {
        return p;
    }
    data_root().join("lc-rs").join("index.redb")
}

/// Directory for a single model key.
pub fn model_dir(key: &str) -> PathBuf {
    sanitize_dir_component(key)
        .map_or_else(|| models_root().join("unknown"), |k| models_root().join(k))
}

/// Turn a model key into a single safe path component.
///
/// Rejects absolute paths and traversal so a hostile key cannot escape the
/// models root. Mirrors the Python implementation's assumption that keys are
/// registry-controlled; here they are validated explicitly.
pub fn sanitize_dir_component(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "." || trimmed == ".." {
        return None;
    }
    if trimmed.contains('/') || trimmed.contains('\\') || trimmed.contains('\0') {
        return None;
    }
    // Reject Windows-reserved characters for cross-platform safety.
    if trimmed
        .chars()
        .any(|c| matches!(c, '<' | '>' | ':' | '"' | '|' | '?' | '*'))
    {
        return None;
    }
    if trimmed.starts_with('-') {
        return None;
    }
    Some(trimmed.to_string())
}

/// Join `base` with `component`, refusing to escape `base`.
pub fn safe_join(base: &Path, component: &str) -> Result<PathBuf> {
    let name = sanitize_dir_component(component).ok_or_else(|| {
        Error::Usage(format!(
            "refusing unsafe path component {component:?} (must be a plain name, no separators)"
        ))
    })?;
    Ok(base.join(name))
}

/// Make sure `dir` exists.
pub fn ensure_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_path(dir)
}

/// Append a suffix to a filename without ``Path::with_suffix`` stripping
/// extra dots from the stem (``lecture.2024`` must stay ``lecture.2024.txt``).
pub fn append_suffix(basename: &Path, suffix: &str) -> PathBuf {
    let suffix = if suffix.starts_with('.') {
        suffix.to_string()
    } else {
        format!(".{suffix}")
    };
    match basename.parent() {
        Some(parent) => parent.join(format!(
            "{}{}",
            basename.file_name().unwrap_or_default().to_string_lossy(),
            suffix
        )),
        None => PathBuf::from(format!("{}{}", basename.to_string_lossy(), suffix)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_rejects_traversal() {
        assert_eq!(sanitize_dir_component(".."), None);
        assert_eq!(sanitize_dir_component("../etc"), None);
        assert_eq!(sanitize_dir_component("a/b"), None);
        assert_eq!(sanitize_dir_component(""), None);
        assert_eq!(
            sanitize_dir_component("qwen3-asr-0.6b"),
            Some("qwen3-asr-0.6b".into())
        );
    }

    #[test]
    fn safe_join_stays_inside_base() {
        let base = Path::new("/tmp/models");
        assert_eq!(safe_join(base, "ok").unwrap(), base.join("ok"));
        assert!(safe_join(base, "../escape").is_err());
        assert!(safe_join(base, "/etc").is_err());
    }

    #[test]
    fn append_suffix_keeps_extra_dots() {
        let p = Path::new("/tmp/out/lecture.2024");
        assert_eq!(
            append_suffix(p, ".txt"),
            PathBuf::from("/tmp/out/lecture.2024.txt")
        );
        assert_eq!(
            append_suffix(p, "srt"),
            PathBuf::from("/tmp/out/lecture.2024.srt")
        );
        assert_eq!(
            append_suffix(Path::new("a.md"), ".txt"),
            PathBuf::from("a.md.txt")
        );
    }
}
