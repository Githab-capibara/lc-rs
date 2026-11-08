# 02. Pin the ONNX graph optimisation level to `ORT_ENABLE_ALL`

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [adr/01-run-models-in-process-through-onnx-runtime.md](01-run-models-in-process-through-onnx-runtime.md), [design/05-nemotron-streaming-backend.md](../design/05-nemotron-streaming-backend.md)

## Context

ONNX Runtime rewrites a graph before executing it. The rewrite level is a
trade-off: a lower level keeps the graph closer to the published export and
runs faster to load, a higher level fuses operators and produces faster
inference — at the price of changing the numerics of the result.

The `ort` crate exposes the levels through `GraphOptimizationLevel`, and the
naming is a trap. The crate's variants are `Disable`, `Level1`, `Level2`,
`Level3` and `All`, and a reader naturally assumes `Level3` is "the highest
level". It is not: the crate maps

| `ort` variant | ONNX Runtime constant |
|---|---|
| `Level1` | `ORT_ENABLE_BASIC` |
| `Level2` | `ORT_ENABLE_EXTENDED` |
| `Level3` | `ORT_ENABLE_LAYOUT` |
| `All` | `ORT_ENABLE_ALL` |

`ORT_ENABLE_LAYOUT` is the *layout* optimisation — it rewrites the graph for
the memory layout of the target device — and it is strictly *less* work than
`ORT_ENABLE_ALL`, which is the extended optimisation plus the layout pass.

That distinction is not academic. On the quantised Conformer graphs of the
Nemotron export, running with `Level3` produced transcripts that differed
from the reference implementation in their punctuation and in a handful of
words: the layout rewrite changed the accumulation order of the int8
matmuls, and the greedy RNN-T search amplified a single differing logit
into a different token. The same graphs, run with `All`, matched the
reference transcript.

## Decision

Every session is created with the optimisation level pinned explicitly to
`ort::session::builder::GraphOptimizationLevel::All`, in
`ml::configure_session` (`src/ml/mod.rs`):

```rust
builder.with_optimization_level(
    ort::session::builder::GraphOptimizationLevel::All,
)
```

The value is also ONNX Runtime's own default, so pinning it changes no
behaviour — it documents the choice and prevents a future "optimisation" from
silently downgrading the level to `Level3` in the name of faster loading.

The same helper pins the intra-op thread count to
`config::inference_threads()`, which defaults to the number of available
logical CPUs.

## Consequences

- **Easier:** the transcripts match the reference implementation on the
  quantised graphs, because the runtime performs the same rewrites the
  reference runtime performed.
- **Easier:** the choice is visible in one place. Every model — language
  identifier, Parakeet, Nemotron, Qwen — is created through the same helper,
  so no backend can drift to a different level.
- **Harder:** session creation takes slightly longer, because the extended
  optimisation pass rewrites more of the graph before the first inference.
  The cost is paid once per session, not per window.
- **Harder:** a reader who assumes `Level3` is the strongest level will be
  wrong. The comment in `src/ml/mod.rs` records the mapping so the next
  reader does not have to rediscover it.
- **Given up:** the marginally faster graph-loading time of
  `ORT_ENABLE_LAYOUT`. Load time is not the bottleneck of a transcription
  run; transcript fidelity is.
- **Migration:** none. The level is a build-time constant, and the sessions
  are created per process.

## Alternatives considered

- **Leave the level at the runtime default.** Rejected because "the
  default" is not visible at the call site, and the failure mode of
  choosing the wrong level is a *silent* transcript difference rather than a
  crash. Pinning it makes the decision reviewable.
- **Use `Level3` for faster session creation.** Rejected empirically: it
  changes the numerics of the int8 Conformer graphs and therefore the
  transcripts.
- **Pin `Disable` for bit-exact reproducibility.** Rejected because it
  forgoes the operator fusions that make the graphs fast, and it still would
  not reproduce the reference runtime bit for bit — the reference ran
  `torch`, not ONNX Runtime.
