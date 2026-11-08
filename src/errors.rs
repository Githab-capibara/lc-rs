//! Error hierarchy, mirroring ``src/localcaption/errors.py``.
//!
//! Every failure mode the Python implementation modelled as a distinct
//! exception class gets its own variant here, so callers (and tests) can
//! match on a specific failure rather than scraping strings.

use std::path::PathBuf;

/// Result alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Root of every lc-rs failure. Renamed from the Python
/// ``LocalCaptionError`` but semantically identical.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A required external tool (ffmpeg, yt-dlp) or model is missing.
    #[error("dependency error: {0}")]
    Dependency(String),

    /// yt-dlp (or a direct HTTP fetch) failed to fetch the requested media.
    #[error("download error: {0}")]
    Download(String),

    /// ffmpeg failed to produce the expected WAV file, or the WAV itself was
    /// not in the expected shape.
    #[error("audio conversion error: {0}")]
    AudioConversion(String),

    /// An ASR/LangID model failed to produce a transcript.
    #[error("transcription error: {0}")]
    Transcription(String),

    /// An automated install step (model download, model extraction) failed.
    #[error("install error: {0}")]
    Install(String),

    /// The model registry rejected a key, or a checkpoint is incomplete.
    #[error("model registry error: {0}")]
    Model(String),

    /// Playlist expansion failed.
    #[error("playlist error: {0}")]
    Playlist(String),

    /// A local-capability check (CLI argument combination) failed.
    #[error("usage error: {0}")]
    Usage(String),

    /// Anything about the filesystem: creating directories, writing files,
    /// reading the batch list or the index.
    #[error("io error: {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// JSON (de)serialisation failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// An ONNX Runtime call failed.
    #[error("onnx error: {0}")]
    Onnx(String),

    /// Tokenizer construction or encoding failed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// The user interrupted the run (Ctrl-C).
    #[error("interrupted")]
    Interrupted,
}

impl Error {
    /// Construct an [`Error::Io`] from a path and a [`std::io::Error`].
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// Convenience trait for mapping a std IO error with the path it relates to.
pub trait IoContext<T> {
    /// Attach `path` to any [`std::io::Error`] this result carries.
    fn with_path(self, path: impl Into<PathBuf>) -> Result<T>;
}

impl<T> IoContext<T> for std::result::Result<T, std::io::Error> {
    fn with_path(self, path: impl Into<PathBuf>) -> Result<T> {
        self.map_err(|e| Error::io(path, e))
    }
}

/// True when the failure is a user interrupt (so the CLI can exit 130).
pub fn is_interrupt(err: &Error) -> bool {
    matches!(err, Error::Interrupted)
}
