//! ONNX Runtime inference for the local models.
//!
//! Every model runs in-process through ONNX Runtime; no Python runtime and
//! no child process is involved. Sessions are created lazily and cached so a
//! batch run loads each graph once, and every session is placed on the GPU or
//! the CPU per its own VRAM budget — see [`crate::gpu`].

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
/// The environment owns the global thread pool, so it is created once per
/// process and reused by every session.
///
/// It deliberately does *not* carry execution providers: the placement is a
/// per-model decision (a 640 MB int8 encoder and a 3.8 GB fp32 decoder do
/// not get the same VRAM budget), and providers registered here cannot be
/// capped per session. Providers are attached in
/// `attach_execution_providers`, which sees the checkpoint each session
/// belongs to.
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

    let env = ort::init().with_name("lc-rs").build().map_err(|e| {
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

/// The size of the checkpoint a session belongs to, in bytes.
///
/// The budget has to describe the whole checkpoint, not the one file being
/// loaded: weights are often external to the graph (Qwen keeps 3 GB of them
/// in `decoder_weights.data`), one model is split across several graphs
/// (the Parakeet feature extractor, the encoder and the joint network are
/// three sessions out of one directory), and those graphs are resident at
/// the same time. Every regular file of the directory is counted — the
/// vocabularies and configs next to the weights are kilobytes against the
/// gigabytes that matter.
pub fn checkpoint_bytes(path: &Path) -> u64 {
    let Some(dir) = path.parent() else {
        return std::fs::metadata(path).map_or(0, |meta| meta.len());
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return std::fs::metadata(path).map_or(0, |meta| meta.len());
    };
    entries
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_file()))
        .filter_map(|entry| entry.metadata().ok())
        .map(|meta| meta.len())
        .sum()
}

/// Attach the execution providers for a session of a checkpoint of
/// `model_bytes` bytes.
///
/// The placement is [`crate::gpu::Decision`] applied to the session builder:
/// a session that fits on the card gets the CUDA provider with an unbounded
/// arena, a session that only partly fits gets it with the arena capped at
/// the free VRAM, and a session that does not fit at all is left on the CPU
/// without ever registering the provider.
///
/// A provider that cannot be registered — no CUDA runtime on the machine,
/// a driver that refuses the device — is reported and recovered from, not
/// swallowed: the builder is handed back configured for the CPU instead.
/// Registering it silently would leave a run on the CPU with no way to tell
/// whether that is what the user asked for.
///
/// The second element of the result says whether the GPU was actually
/// requested, which is what lets the caller retry the whole session on the
/// CPU when the placement turns out to be unworkable.
fn attach_execution_providers(
    env: &ort::environment::Environment,
    builder: ort::session::builder::SessionBuilder,
    decision: &crate::gpu::Decision,
) -> (ort::session::builder::SessionBuilder, bool) {
    #[cfg(feature = "cuda")]
    {
        use ort::ep::CUDA;

        if !decision.placement.on_gpu() {
            tracing::debug!("CPU placement: {}", decision.summary());
            return (builder, false);
        }
        if let Err(reason) = cuda_registers(env) {
            tracing::debug!("the CUDA execution provider is unavailable here ({reason})");
            return (builder, false);
        }

        let mut provider = CUDA::default();
        if let crate::gpu::Placement::Gpu {
            memory_limit_mib: Some(limit),
        } = decision.placement
        {
            // A bounded arena with the requested size rather than the next
            // power of two: the budget is the free VRAM the card actually
            // has, and rounding it up would overshoot it.
            provider = provider
                .with_memory_limit(limit * crate::gpu::MIB)
                .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested);
        }
        tracing::info!("on-device inference: {}", decision.summary());
        match builder.with_execution_providers([provider.build().error_on_failure()]) {
            Ok(builder) => (builder, true),
            Err(failure) => {
                tracing::warn!(
                    "the CUDA execution provider could not be registered ({}); \
                     every session stays on the CPU",
                    failure
                );
                (failure.recover(), false)
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (env, decision);
        (builder, false)
    }
}

/// Whether the CUDA execution provider can be registered at all here.
///
/// The answer is measured once per process by registering the provider on
/// a throwaway session builder with the failure flag set: that is the
/// moment ONNX Runtime loads `libonnxruntime_providers_cuda.so` and the
/// CUDA runtime behind it. Cached, because every later session would pay
/// the same failure again — and because a per-model failure is what the
/// caller already reports.
#[cfg(feature = "cuda")]
fn cuda_registers(env: &ort::environment::Environment) -> std::result::Result<(), String> {
    use std::sync::OnceLock;

    static VERDICT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    VERDICT
        .get_or_init(|| {
            let verdict = Session::builder(env)
                .map_err(|e| e.to_string())
                .and_then(|builder| {
                    builder
                        .with_execution_providers([ort::ep::CUDA::default()
                            .build()
                            .error_on_failure()])
                        .map(|_| ())
                        .map_err(|failure| condense_reason(&failure.to_string()))
                });
            // Reported once, here: every later session that asks for the
            // GPU gets the same answer from the cache and only logs it at
            // debug level, so a batch of six models does not print six
            // identical warnings.
            if let Err(reason) = &verdict {
                tracing::warn!(
                    "the CUDA execution provider is unavailable on this machine ({reason}); \
                     every session stays on the CPU"
                );
            }
            verdict
        })
        .clone()
}

/// Reduce an ONNX Runtime provider error to the part that carries
/// information.
///
/// ORT frames its provider failures with a source path from the machine it
/// was built on and wraps them over several lines; neither belongs in a log
/// line or in a doctor report. What matters is the marker and what follows
/// it: which library could not be loaded, and why.
#[cfg(feature = "cuda")]
fn condense_reason(reason: &str) -> String {
    let condensed = reason
        .split("FAIL : ")
        .last()
        .unwrap_or(reason)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let shortened: String = condensed.chars().take(180).collect();
    if shortened.len() == condensed.len() {
        shortened
    } else {
        format!("{shortened}…")
    }
}

/// Commit a session, and retry it on the CPU when the GPU placement fails.
///
/// A placement is a bet: the arena cap is derived from a VRAM reading, not
/// from a run, and the first allocation of a cuDNN workspace can still
/// exceed it. When that happens the model is far more useful on the CPU
/// than as an error, so the session is rebuilt without any provider and
/// committed again.
fn commit_with_cpu_retry<T, F>(
    builder: ort::session::builder::SessionBuilder,
    gpu_requested: bool,
    what: &str,
    commit: F,
) -> Result<T>
where
    F: Fn(ort::session::builder::SessionBuilder) -> ort::Result<T>,
{
    match commit(builder) {
        Ok(session) => Ok(session),
        Err(failure) if gpu_requested => {
            tracing::warn!("{what} could not run on the GPU ({failure}); retrying on the CPU");
            let env = environment()?;
            let builder =
                configure_session(Session::builder(&env).map_err(|e| {
                    Error::Onnx(format!("could not create a session builder: {e}"))
                })?)?;
            commit(builder).map_err(|e| Error::Onnx(format!("could not load {what}: {e}")))
        }
        Err(failure) => Err(Error::Onnx(format!("could not load {what}: {failure}"))),
    }
}

/// Load an ONNX graph from disk.
pub fn load_session(path: impl AsRef<Path>) -> Result<Session> {
    let path = path.as_ref();
    let env = environment()?;
    let decision = crate::gpu::plan(checkpoint_bytes(path));
    let builder = configure_session(
        Session::builder(&env)
            .map_err(|e| Error::Onnx(format!("could not create a session builder: {e}")))?,
    )?;
    let (builder, gpu_requested) = attach_execution_providers(&env, builder, &decision);
    let what = path.display().to_string();
    commit_with_cpu_retry(builder, gpu_requested, &what, |mut builder| {
        builder.commit_from_file(path)
    })
}

/// Load an ONNX graph from memory.
pub fn load_session_from_memory(bytes: &[u8]) -> Result<Session> {
    let env = environment()?;
    let decision = crate::gpu::plan(bytes.len() as u64);
    let builder = configure_session(
        Session::builder(&env)
            .map_err(|e| Error::Onnx(format!("could not create a session builder: {e}")))?,
    )?;
    let (builder, gpu_requested) = attach_execution_providers(&env, builder, &decision);
    commit_with_cpu_retry(
        builder,
        gpu_requested,
        "the embedded graph",
        |mut builder| builder.commit_from_memory(bytes),
    )
}

/// Convert an ONNX Runtime error into our error type.
pub fn onnx_error(context: &str, err: &ort::Error) -> Error {
    Error::Onnx(format!("{context}: {err}"))
}

/// The state of the CUDA execution provider on this machine.
///
/// Reported by `lc-rs doctor` so that a run which stayed on the CPU can be
/// explained: a card that is busy is a placement decision (see
/// [`crate::gpu`]), a provider that cannot register is a missing runtime,
/// and the two look identical from the outside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaSupport {
    /// The binary was built without the `cuda` cargo feature.
    NotCompiled,
    /// The provider is compiled in and registers on this machine.
    Available,
    /// The provider is compiled in and refuses to register here, with the
    /// reason ONNX Runtime gave.
    Unavailable(String),
}

impl CudaSupport {
    /// A one-line description.
    pub fn detail(&self) -> String {
        match self {
            Self::NotCompiled => "not compiled in (rebuild with `--features cuda`)".to_string(),
            Self::Available => "available".to_string(),
            Self::Unavailable(reason) => format!("unavailable: {reason}"),
        }
    }
}

/// Whether this binary can place a session on the GPU at all.
///
/// On a build without the `cuda` feature the answer is a compile-time
/// fact. Otherwise it is the cached verdict of registering the provider,
/// which is a `nvidia-smi`-free way of asking "is the CUDA runtime
/// actually usable here?".
pub fn cuda_support() -> CudaSupport {
    #[cfg(feature = "cuda")]
    {
        match environment() {
            Ok(env) => match cuda_registers(&env) {
                Ok(()) => CudaSupport::Available,
                Err(reason) => CudaSupport::Unavailable(reason),
            },
            Err(_) => CudaSupport::Unavailable(
                "the ONNX Runtime environment could not be created".to_string(),
            ),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        CudaSupport::NotCompiled
    }
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

    /// The committed session of a checkpoint is measured by the checkpoint's
    /// whole directory, not by the one graph file being loaded: Qwen keeps
    /// three gigabytes of decoder weights in a sibling `.data` file.
    #[test]
    fn the_budget_counts_the_whole_checkpoint() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        std::fs::write(dir.path().join("decoder_init.onnx"), [0_u8; 4]).expect("a graph");
        std::fs::write(dir.path().join("decoder_weights.data"), [0_u8; 12]).expect("weights");
        std::fs::write(dir.path().join("tokenizer.json"), [0_u8; 3]).expect("a vocabulary");

        assert_eq!(
            checkpoint_bytes(&dir.path().join("decoder_init.onnx")),
            19,
            "every file of the checkpoint counts towards the VRAM budget"
        );
        assert_eq!(
            checkpoint_bytes(&dir.path().join("missing.onnx")),
            19,
            "a graph that is not loaded yet still reports the checkpoint it belongs to"
        );
    }

    /// The mock failure: the checkpoint does not exist, which is a real
    /// ONNX Runtime error rather than a fabricated one.
    fn commit_a_missing_checkpoint(
        mut builder: ort::session::builder::SessionBuilder,
    ) -> ort::Result<usize> {
        builder
            .commit_from_file("/nonexistent-checkpoint.onnx")
            .map(|_| 0)
    }

    /// A placement is a bet: when the session cannot be created on the GPU
    /// the model is rebuilt on the CPU instead of failing the run.
    #[test]
    fn a_failed_gpu_session_is_retried_on_the_cpu() {
        let env = environment().expect("the shared environment");
        let builder = configure_session(Session::builder(&env).expect("a session builder"))
            .expect("the shared options");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let spy = Arc::clone(&calls);

        let outcome: Result<usize> =
            commit_with_cpu_retry(builder, true, "the probe checkpoint", move |builder| {
                spy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                commit_a_missing_checkpoint(builder)
            });

        assert!(outcome.is_err(), "a missing checkpoint cannot load");
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the GPU failure must be retried on the CPU"
        );
    }

    /// Without a GPU placement there is nothing to fall back to, so a
    /// failure is reported as it is rather than paid for twice.
    #[test]
    fn a_failure_on_the_cpu_is_not_retried() {
        let env = environment().expect("the shared environment");
        let builder = configure_session(Session::builder(&env).expect("a session builder"))
            .expect("the shared options");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let spy = Arc::clone(&calls);

        let outcome: Result<usize> =
            commit_with_cpu_retry(builder, false, "the probe checkpoint", move |builder| {
                spy.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                commit_a_missing_checkpoint(builder)
            });

        assert!(outcome.is_err(), "a missing checkpoint cannot load");
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "a CPU failure must not be attempted twice"
        );
    }
}
