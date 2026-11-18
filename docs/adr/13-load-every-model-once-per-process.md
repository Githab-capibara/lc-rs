# 13. Load every model once per process

- **Status:** Accepted
- **Date:** 2026-11-19
- **Deciders:** @Githab-capibara
- **Related:** [adr/01-run-models-in-process-through-onnx-runtime.md](01-run-models-in-process-through-onnx-runtime.md), [adr/03-adopt-per-model-onnx-exports.md](03-adopt-per-model-onnx-exports.md), [adr/16-reject-caching-the-redb-connection.md](16-reject-caching-the-redb-connection.md), [design/01-pipeline-stages.md](../design/01-pipeline-stages.md)

## Context

`src/ml/asr.rs` loaded its checkpoint at the top of every transcription
call, and `src/ml/langid.rs` parsed the embedded ECAPA graph at the top of
every language detection. One call means one file, so a batch of N files
paid the same load N times:

| Load | Measured (release, this machine) | Weights held |
|---|---|---|
| Parakeet fp32 | 3.3 – 5.0 s | 2.4 GB on disk, ~2.6 GiB resident |
| Qwen fp32 | 5.1 – 9.2 s | 3.8 GB on disk |
| Nemotron int8 | not measured, the same order | ~1.0 GB on disk |
| language-ID graph | ~0.23 s | 139 KB, compiled into the binary |

The Qwen load was the single most expensive thing the crate did that was not
inference: a two-file batch spent nine seconds of it on a checkpoint that
never changed between the two files. The load also holds gigabytes of
resident memory per call, which the previous call had just released.

Two facts about the runtime made a per-process cache possible:

- `ort::Session` is `Send + Sync`, but it is not `Clone` and `Session::run`
  takes `&mut self`, because ONNX Runtime's internals are not safe under
  concurrent inference on one session;
- a batch transcribes strictly one file at a time (`--batch` is sequential),
  so a lock around a session is never contended — it exists to satisfy the
  borrow checker, not to serialise work.

## Decision

`src/ml/mod.rs` gains `cached_model`, a process-wide cache keyed by the
canonicalised checkpoint directory (or by an explicit name for the embedded
graph), holding `Arc<Mutex<T>>`:

- the first caller with a given key builds the model through the closure it
  passes, and stores the result;
- every later caller receives the same `Arc` and takes the `Mutex` for the
  duration of its inference, through `ml::lock_model`;
- loading happens **outside** the cache lock, so two threads racing on an
  unseen checkpoint both load, the first to store wins, and the loser uses
  the winner's model — one model, and neither call fails;
- a load that fails is **not** cached, so a transient error (a checkpoint
  still being written, a busy disk) is retried on the next call instead of
  poisoning the rest of the run;
- the entries are type-erased (`dyn Any`) so that one map serves every model
  type, and a slot is only handed out after a successful downcast, so two
  different models can never be confused for one another because they share
  a key.

`langid`, `asr` (Parakeet, Qwen, Nemotron) and the two model structs behind
them are unchanged otherwise: the call sites became
`cached_model(dir, || Model::load(dir))?` plus `lock_model(&cached)?`.

`src/pipeline.rs` additionally moves language detection **and** the whole
ASR inference into `tokio::task::spawn_blocking`. ONNX Runtime runs its
graphs on its own thread pool and never yields to an async executor, so
running minutes of inference directly on the runtime starved the
`yt-dlp`/`ffmpeg` timeout futures, the progress output and Ctrl-C handling
for the whole inference. The blocking pool exists for exactly this.

## Consequences

- **Easier:** a batch no longer pays the load per file. Measured on the CLI:
  the first file of a two-file Qwen batch took 28.8 s wall clock, the second
  one added 6 s — the entire 5 – 9 s checkpoint load, and its ~4 GiB of
  resident memory, disappeared from the second file. The warm cache lookup
  itself costs 47 µs.
- **Easier:** the language identifier, whose first call dominated the
  detection stage, keeps its parsed graph between the pipeline's calls, so a
  run that detects twice (a batch, or a retry) parses once.
- **Easier:** the async runtime stays responsive during inference — the
  `ffmpeg` timeout future and Ctrl-C are no longer blocked behind it.
- **Harder:** a loaded model stays resident for the lifetime of the process.
  For a CLI that is exactly one run's worth of memory, and it is memory the
  per-file reload pattern was already paying — but a long-lived embedding of
  the library would now hold every checkpoint it ever touched.
- **Harder:** the models are shared, so `Qwen::set_language` and
  `Nemotron::set_language` mutate a value that outlives their call. That is
  safe because inference happens while the `Mutex` is held and every caller
  re-applies its own language before it infers, exactly like the old
  per-call load did implicitly.
- **Harder:** a poisoned `Mutex` (an inference that panicked) is reported as
  an error rather than silently recovered, because a panicked session leaves
  ONNX Runtime's internal state undefined.
- **Given up:** the ability to swap a checkpoint on disk under a running
  process and have the next call pick it up. `lc-rs model download` followed
  by another transcription in the same process is not a supported pattern;
  each `lc-rs` invocation is a fresh process.
- **Migration:** none. The cache is invisible to callers and to the
  command-line surface.

## Alternatives considered

- **Cache the redb connection the same way.** Rejected, with evidence: see
  [adr/16-reject-caching-the-redb-connection.md](16-reject-caching-the-redb-connection.md).
  ONNX Runtime sessions are sharable within a process; a redb writer is not.
- **Keep loading per call.** Rejected: it is the largest avoidable cost in
  the crate, and it scales with the batch size rather than being paid once.
- **Reload on a TTL or on a checksum change.** Rejected: the CLI is one run
  per process, so there is nothing to expire; it would add machinery for a
  case that does not exist and make the supported case slower.
- **Share through `Arc<Session>` without a `Mutex`.** Rejected: `Session::run`
  takes `&mut self`, so it does not type-check — and it would be wrong if it
  did, since ONNX Runtime forbids concurrent inference on one session.
- **Run each file on its own thread with its own loaded model.** Rejected:
  it multiplies the resident memory by the parallelism, which is the cost the
  cache exists to remove, and a batch is sequential anyway.