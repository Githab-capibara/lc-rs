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
