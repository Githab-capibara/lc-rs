# Architectural decision records

Every decision that shaped `lc-rs`, in the Michael Nygard format: the
context that forced the choice, the choice itself, and its consequences.

| # | Title | Status | Date |
|---|---|---|---|
| [01-run-models-in-process-through-onnx-runtime](01-run-models-in-process-through-onnx-runtime.md) | Run every model in-process through ONNX Runtime | Accepted | 2026-11-09 |
| [02-pin-the-onnx-graph-optimisation-level](02-pin-the-onnx-graph-optimisation-level.md) | Pin the ONNX graph optimisation level to `ORT_ENABLE_ALL` | Accepted | 2026-11-09 |
| [03-adopt-per-model-onnx-exports](03-adopt-per-model-onnx-exports.md) | Adopt per-model ONNX exports instead of a single unified runtime | Accepted | 2026-11-09 |
| [04-route-asr-by-the-detected-language](04-route-asr-by-the-detected-language.md) | Route ASR by the detected language | Accepted | 2026-11-09 |
| [05-keep-the-command-line-surface-one-to-one](05-keep-the-command-line-surface-one-to-one.md) | Keep the command-line surface one-to-one with the Python implementation | Accepted | 2026-11-09 |
| [06-store-the-transcript-search-index-in-redb](06-store-the-transcript-search-index-in-redb.md) | Store the transcript search index in redb | Accepted | 2026-11-09 |
| [07-route-every-outbound-request-through-the-proxy](07-route-every-outbound-request-through-the-proxy.md) | Route every outbound request through the SOCKS5 proxy | Accepted | 2026-11-09 |
| [08-compile-cuda-support-as-an-opt-in-feature](08-compile-cuda-support-as-an-opt-in-feature.md) | Compile CUDA support as an opt-in cargo feature | Accepted | 2026-11-09 |
| [09-reproduce-the-reference-front-end-quirks](09-reproduce-the-reference-front-end-quirks.md) | Reproduce the reference implementation's front-end quirks | Accepted | 2026-11-09 |
| [10-ship-the-fp32-qwen-decoder](10-ship-the-fp32-qwen-decoder.md) | Ship the fp32 Qwen decoder and keep the int4 one as an option | Accepted | 2026-11-09 |
| [11-report-the-language-the-way-the-reference-does](11-report-the-language-the-way-the-reference-does.md) | Report the language the way the reference does | Accepted | 2026-11-10 |
| [12-ship-the-fp32-parakeet-export](12-ship-the-fp32-parakeet-export.md) | Ship the fp32 Parakeet export and keep the int8 one as an option | Accepted | 2026-11-10 |
| [13-load-every-model-once-per-process](13-load-every-model-once-per-process.md) | Load every model once per process | Accepted | 2026-11-19 |
| [14-fuse-the-mel-front-end-into-the-fft-loop](14-fuse-the-mel-front-end-into-the-fft-loop.md) | Fuse the mel front-end into the FFT loop | Accepted | 2026-11-19 |
| [15-prune-the-dependency-tree](15-prune-the-dependency-tree.md) | Prune the dependency tree and drop the oniguruma regex engine | Accepted | 2026-11-19 |
| [16-reject-caching-the-redb-connection](16-reject-caching-the-redb-connection.md) | Reject caching the redb connection | Rejected | 2026-11-19 |
| [17-place-inference-by-the-measured-vram-budget](17-place-inference-by-the-measured-vram-budget.md) | Place inference by the measured VRAM budget, CPU first as the fallback | Accepted | 2026-11-19 |

## The decisions in one paragraph

The Python implementation ran each model family in its own virtualenv, with
a `torch` process per transcript and a JSON document crossing the process
boundary. `lc-rs` runs every model in-process through ONNX Runtime, using
the published ONNX exports of the very same checkpoints, so that the weights
and the graph semantics are unchanged while the process boundary, the
virtualenvs and the start-up cost disappear. The transcripts therefore stay
comparable with the reference ones — down to the quirks of its front-ends,
which are reproduced deliberately and documented rather than silently
diverging from.

Three later decisions are about paying less for the same output, and they
share one rule: **the numbers do not move, only the work does.** A
checkpoint is loaded once per process rather than once per file
([13](13-load-every-model-once-per-process.md)); the mel front-end is fused
into the FFT loop and its tables cached, which makes a 30-second window 4.7×
cheaper without changing a single output bit
([14](14-fuse-the-mel-front-end-into-the-fft-loop.md)); and the dependency
tree loses the Oniguruma-backed tokenizer build, which removes a C library
without changing a single token id ([15](15-prune-the-dependency-tree.md)).
Each is verified by a parity test rather than by inspection, because a
faster port that says something different is not a port.

The fourth record is a rejection, kept because the proposal was reasonable
and the reason it fails is not obvious from the code: the search index is
*not* cached the way the models are, because a redb writer is exclusive and
a session is not ([16](16-reject-caching-the-redb-connection.md)).

The fifth moves the inference off the CPU when the workstation can host it,
and it starts from the fact that "the GPU" is not one state on a shared
machine: a card with room runs the whole checkpoint, a card with part of a
checkpoint's room runs what fits and leaves the rest on the CPU, and a card
with none runs nothing at all. The budget is measured with `nvidia-smi` —
a child process, because the crate forbids unsafe code and takes no NVML
binding — and the CPU is never further away than one failed session
([17](17-place-inference-by-the-measured-vram-budget.md)).
