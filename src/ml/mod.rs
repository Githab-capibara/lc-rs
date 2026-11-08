//! ONNX Runtime inference for the local models.
//!
//! Every model runs in-process through ONNX Runtime; no Python runtime and
//! no child process is involved. Sessions are created lazily and cached so a
//! batch run loads each graph once.

pub mod asr;
pub mod features;
pub mod langid;
pub mod nemotron;
pub mod parakeet;
pub mod qwen;
pub mod wav;

use std::path::Path;
use std::sync::Arc;

use ort::session::Session;

use crate::errors::{Error, Result};

/// Shared ONNX Runtime environment.
///
/// The environment owns the global thread pool and the execution providers,
/// so it is created once per process and reused by every session.
pub fn environment() -> Result<Arc<ort::environment::Environment>> {
    use std::sync::OnceLock;
    static ENV: OnceLock<std::sync::Mutex<Option<Arc<ort::environment::Environment>>>> =
        OnceLock::new();
    let slot = ENV.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = slot
        .lock()
        .map_err(|_| Error::Onnx("ONNX Runtime environment lock was poisoned".to_string()))?;
    if let Some(env) = guard.as_ref() {
        return Ok(Arc::clone(env));
    }

    #[allow(unused_mut)]
    let mut builder = ort::init().with_name("lc-rs");
    #[cfg(feature = "cuda")]
    {
        if crate::config::use_cuda() {
            builder = builder.with_execution_providers([ort::ep::CUDA::default().build()]);
        }
    }
    let env = builder.build().map_err(|e| {
        Error::Onnx(format!(
            "could not create the ONNX Runtime environment: {e}"
        ))
    })?;
    let env = Arc::new(env);
    *guard = Some(Arc::clone(&env));
    Ok(env)
}

/// Apply the session options shared by every model.
///
/// [`GraphOptimizationLevel::All`] (`ORT_ENABLE_ALL`) is requested explicitly.
/// It is also ONNX Runtime's default, but pinning it documents the choice:
/// `ort`'s `Level3` variant is *not* the same level — it maps to
/// `ORT_ENABLE_LAYOUT`, which produces measurably different numerics on the
/// quantised Conformer graphs.
fn configure_session(
    mut builder: ort::session::builder::SessionBuilder,
) -> Result<ort::session::builder::SessionBuilder> {
    builder = builder
        .with_intra_threads(crate::config::inference_threads())
        .map_err(|e| Error::Onnx(format!("could not configure the session: {e}")))?;
    builder = builder
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)
        .map_err(|e| Error::Onnx(format!("could not configure the session: {e}")))?;
    Ok(builder)
}

/// Load an ONNX graph from disk.
pub fn load_session(path: &Path) -> Result<Session> {
    let env = environment()?;
    let mut builder = configure_session(
        Session::builder(&env)
            .map_err(|e| Error::Onnx(format!("could not create a session builder: {e}")))?,
    )?;
    builder
        .commit_from_file(path)
        .map_err(|e| Error::Onnx(format!("could not load {}: {e}", path.display())))
}

/// Load an ONNX graph from memory.
pub fn load_session_from_memory(bytes: &[u8]) -> Result<Session> {
    let env = environment()?;
    let mut builder = configure_session(
        Session::builder(&env)
            .map_err(|e| Error::Onnx(format!("could not create a session builder: {e}")))?,
    )?;
    builder
        .commit_from_memory(bytes)
        .map_err(|e| Error::Onnx(format!("could not load the embedded graph: {e}")))
}

/// Convert an ONNX Runtime error into our error type.
pub fn onnx_error(context: &str, err: &ort::Error) -> Error {
    Error::Onnx(format!("{context}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_is_created_once() {
        let a = environment().unwrap();
        let b = environment().unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }
}
