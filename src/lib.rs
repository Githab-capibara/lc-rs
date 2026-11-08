//! `lc-rs` — fully-local video → transcript.
//!
//! This crate is a from-scratch Rust port of the Python project
//! `localcaption`. It keeps the original command-line surface one-to-one
//! while replacing the Python ML runtimes with ONNX Runtime sessions driven
//! directly from Rust.
//!
//! # Module map
//!
//! | Module | Responsibility |
//! |--------|----------------|
//! | [`cli`] | clap command-line definitions and dispatch |
//! | [`config`] | environment-driven configuration (proxy, models, index) |
//! | [`downloader`] | `yt-dlp` driver (media download, metadata, playlists) |
//! | [`audio`] | `ffmpeg` driver + 16 kHz mono WAV normalisation |
//! | [`formatter`] | transcript serialisers (md/txt/srt/vtt/json) |
//! | [`chapters`] | chapter markers and transcript slicing |
//! | [`search`] | redb-backed transcript index and substring search |
//! | [`summary`] | optional local-Ollama summary |
//! | [`pipeline`] | end-to-end orchestration |
//! | [`ml`] | ONNX Runtime inference (LangID + the three ASR models) |

#![deny(warnings)]
#![forbid(unsafe_code)]
// Lints opted out at crate level, with the reasoning recorded here rather
// than repeated at every call site:
//
// * `must_use_candidate`, `missing_errors_doc`, `missing_panics_doc`,
//   `module_name_repetitions` — a library-first design exposes many small
//   helpers; annotating each one adds noise without catching defects.
// * `cast_precision_loss`, `cast_possible_truncation`, `cast_sign_loss`,
//   `cast_possible_wrap` — the audio/feature pipelines deliberately convert
//   between sample counts, seconds, frame indices and ONNX tensor indices.
//   Every conversion site is range-checked or documented where a truncation
//   would change the result (see `audio`, `ml/features`, `ml/parakeet`).
// * `too_many_lines`, `items_after_statements` — long linear DSP loops and
//   table-driven data stay more readable un-split; decomposition happens along
//   model boundaries instead.
#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::items_after_statements
)]

pub mod audio;
pub mod batch;
pub mod chapters;
pub mod cli;
pub mod config;
pub mod doctor;
pub mod downloader;
pub mod errors;
pub mod formatter;
pub mod languages;
pub mod ml;
pub mod models;
pub mod net;
pub mod paths;
pub mod pipeline;
pub mod playlist;
pub mod proc;
pub mod search;
pub mod summary;

/// Version reported by `lc-rs --version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Program name used in help output and user-facing messages.
pub const PROGRAM: &str = "lc-rs";
