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
pub fn load_session(path: impl AsRef<Path>) -> Result<Session> {
    let env = environment()?;
    let mut builder = configure_session(
        Session::builder(&env)
            .map_err(|e| Error::Onnx(format!("could not create a session builder: {e}")))?,
    )?;
    builder
        .commit_from_file(path.as_ref())
        .map_err(|e| Error::Onnx(format!("could not load {}: {e}", path.as_ref().display())))
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

/// A cache slot holding a loaded model once it exists.
type ModelCell<T> = std::sync::OnceLock<Arc<std::sync::Mutex<T>>>;

/// The shared handle around a [`ModelCell`], so that every caller of one key
/// locks the very same cell.
type ModelSlot<T> = Arc<ModelCell<T>>;

/// The process-wide model cache.
///
/// The slots are type-erased (`dyn Any`) so that one map serves every model
/// type; a slot is only handed out after it has been downcast back to the
/// requested type, so two different models can never be confused for one
/// another just because they share a key.
type ModelCache = std::sync::Mutex<
    std::collections::HashMap<std::path::PathBuf, Arc<dyn std::any::Any + Send + Sync>>,
>;

fn model_cache() -> &'static ModelCache {
    static CACHE: std::sync::OnceLock<ModelCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Load a model once per process and share it afterwards.
///
/// Loading a checkpoint is the most expensive thing this crate does: the
/// Parakeet graphs take ~4 s, the Qwen decoder ~9 s and both hold gigabytes
/// of weights in memory. Every caller used to pay that cost per file, so a
/// batch of N files reloaded the same checkpoint N times. The first caller
/// with a given `key` builds the model; every later caller receives the
/// shared `Arc<Mutex<T>>` and locks it for the duration of its inference.
///
/// The `Mutex` is what makes the sharing safe: `ort::Session` is neither
/// `Clone` nor usable through a shared reference (`run` takes `&mut self`,
/// because ONNX Runtime's internals are not safe under concurrent
/// inference), and a batch transcribes strictly one file at a time, so the
/// lock never contends — it exists to satisfy the borrow checker.
///
/// A load that fails is *not* cached, so a transient error (a checkpoint
/// still being written, a busy disk) is retried on the next call rather
/// than poisoning the run.
///
/// The model stays alive for the lifetime of the process. For a CLI that is
/// exactly one run's worth of memory, and it is memory the per-file reload
/// pattern was already paying.
pub fn cached_model<T, F>(key: impl AsRef<Path>, load: F) -> Result<Arc<std::sync::Mutex<T>>>
where
    T: Send + Sync + 'static,
    F: FnOnce() -> Result<T>,
{
    let path = key.as_ref();
    // Canonicalise so that two spellings of one checkpoint share a slot; fall
    // back to the raw path when it does not exist (an embedded-graph key).
    let cache_key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());

    let slot: ModelSlot<T> = {
        let mut guard = model_cache()
            .lock()
            .map_err(|_| Error::Onnx("the model cache lock was poisoned".to_string()))?;
        let stored = guard.get(&cache_key).cloned();
        match stored.map(std::sync::Arc::downcast::<ModelCell<T>>) {
            Some(Ok(slot)) => slot,
            Some(Err(_)) => {
                // A different model type already owns this key: it is stale,
                // so replace it rather than hand out the wrong model.
                guard.remove(&cache_key);
                Arc::new(ModelCell::new())
            }
            None => {
                let slot: ModelSlot<T> = Arc::new(ModelCell::new());
                guard.insert(cache_key, slot.clone());
                slot
            }
        }
    };

    if let Some(model) = slot.get() {
        return Ok(Arc::clone(model));
    }
    // Loading happens outside the cache lock. Two callers racing on an unseen
    // checkpoint both load, and the first one to store its model wins; the
    // loser drops its copy and uses the winner's, so the outcome is the same
    // one model, and neither call fails.
    let model = Arc::new(std::sync::Mutex::new(load()?));
    Ok(match slot.set(Arc::clone(&model)) {
        Ok(()) => model,
        Err(previous) => previous,
    })
}

/// Lock a cached model for inference.
///
/// A poisoned lock means a previous inference panicked, which leaves the
/// session's internal state undefined: that is reported instead of being
/// papered over.
pub fn lock_model<T>(model: &std::sync::Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    model
        .lock()
        .map_err(|_| Error::Onnx("a model lock was poisoned".to_string()))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[test]
    fn environment_is_created_once() {
        let a = environment().unwrap();
        let b = environment().unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    /// A stand-in for a real checkpoint: loading counts how often it ran.
    #[derive(Debug)]
    struct Probe {
        load_count: usize,
    }

    fn probe_directory() -> tempfile::TempDir {
        tempfile::tempdir().expect("a temporary directory")
    }

    /// Build the stand-in model, counting how often the cache had to load.
    fn load_probe(counter: &Arc<std::sync::atomic::AtomicUsize>) -> Probe {
        Probe {
            load_count: counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1,
        }
    }

    /// The cache exists to stop a batch reloading the same checkpoint per
    /// file: the second call must not call the loader again, and both callers
    /// must end up with the very same model.
    #[test]
    fn the_model_is_loaded_once_and_shared() {
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let dir = probe_directory();
        let counted = Arc::clone(&loads);
        let first: Arc<Mutex<Probe>> =
            cached_model(dir.path(), || Ok(load_probe(&counted))).expect("the first load");
        let second: Arc<Mutex<Probe>> =
            cached_model(dir.path(), || Ok(load_probe(&counted))).expect("the second load");
        assert_eq!(
            first.lock().expect("the lock").load_count,
            1,
            "the second call reused the cached model"
        );
        assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(
            Arc::ptr_eq(&first, &second),
            "both callers must share one model"
        );
        // And the shared model is what gets locked for inference.
        let guard = lock_model(&second).expect("the shared model can be locked");
        assert_eq!(guard.load_count, 1, "the locked model is the loaded one");
    }

    /// A load that fails is not cached: a transient error must not poison
    /// every later call for the rest of the process.
    #[test]
    fn a_failed_load_is_retried_on_the_next_call() {
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let dir = probe_directory();
        let counted = Arc::clone(&loads);
        let failing: Result<Arc<Mutex<Probe>>> = cached_model(dir.path(), || {
            if counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                Err(Error::Onnx("the checkpoint is half-written".to_string()))
            } else {
                Ok(load_probe(&counted))
            }
        });
        assert!(failing.is_err(), "the first load must fail");
        assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 1);

        let counted = Arc::clone(&loads);
        let recovered: Arc<Mutex<Probe>> =
            cached_model(dir.path(), || Ok(load_probe(&counted))).expect("the retry must succeed");
        assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(recovered.lock().expect("the lock").load_count, 2);
    }

    /// One key never serves two model types: the second request must get its
    /// own model, not the first one's.
    #[test]
    fn one_key_never_serves_two_model_types() {
        let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let dir = probe_directory();

        let counted = Arc::clone(&loads);
        let probe: Arc<Mutex<Probe>> =
            cached_model(dir.path(), || Ok(load_probe(&counted))).expect("the probe load");

        #[derive(Debug)]
        struct Other {
            tag: &'static str,
        }
        let other: Arc<Mutex<Other>> = cached_model(dir.path(), || {
            Ok(Other {
                tag: "another model type",
            })
        })
        .expect("the other load");
        assert_eq!(other.lock().expect("the lock").tag, "another model type");

        // The first type is rebuilt the next time it is asked for.
        let counted = Arc::clone(&loads);
        let again: Arc<Mutex<Probe>> =
            cached_model(dir.path(), || Ok(load_probe(&counted))).expect("the rebuild");
        assert_eq!(loads.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert!(!Arc::ptr_eq(&probe, &again));
    }
}
