# 01. Run every model in-process through ONNX Runtime

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [design/01-pipeline-stages.md](../design/01-pipeline-stages.md), [design/02-model-selection.md](../design/02-model-selection.md), [adr/03-adopt-per-model-onnx-exports.md](03-adopt-per-model-onnx-exports.md)

## Context

The Python implementation this port replaces ran each model family in its own
virtualenv: a `torch` environment for Qwen3-ASR, a `NeMo` environment for
Parakeet and Nemotron, and a `speechbrain` environment for the language
identifier. Every transcript therefore meant spawning a child process that
had to import a deep-learning framework, load a checkpoint from disk, decode
a WAV file, write a JSON document to a temporary file and exit again.

That design has three costs that a Rust port does not have to pay:

- **Start-up.** Importing `torch` alone costs several seconds per run, and a
  batch of forty videos paid it forty times.
- **Memory.** Each virtualenv carries its own copy of the framework; the
  checkpoints had to be duplicated on disk for the PyTorch and the ONNX
  exports alike.
- **Surface area.** The JSON protocol between the orchestrator and the
  runners — the payload keys, the segment shape, the error strings — was an
  implicit API with no type checking.

ONNX Runtime is a C++ inference engine with a stable C API. It runs the
published ONNX exports of all three model families, it needs no Python at
all, and it exposes the same API to every language.

## Decision

Every model runs **in-process**, inside the `lc-rs` binary, through the
[`ort`](https://crates.io/crates/ort) crate (ONNX Runtime 1.31):

- the language identifier, Parakeet, Nemotron and Qwen are all plain ONNX
  graphs loaded with `Session::builder(...).commit_from_file(...)`;
- the feature extractors that the exports do *not* ship (the Whisper mel
  filter bank for Qwen, the NeMo mel front-end for Nemotron) are implemented
  in Rust on top of `rustfft` — see [src/ml/features.rs](../../src/ml/features.rs);
- the tokenizers come from the [`tokenizers`](https://crates.io/crates/tokenizers)
  crate, which is a port of the same Rust library the Python stack uses, so
  the BPE merges are identical;
- the ONNX Runtime environment and its thread pool are created once per
  process and shared by every session (`src/ml/mod.rs`).

No child process is spawned for inference, and no JSON payload crosses a
process boundary.

## Consequences

- **Easier:** one binary, one dependency tree, one install step. There is no
  virtualenv to build, no per-model `requirements.txt`, and nothing that can
  drift out of sync with the Rust code.
- **Easier:** a run has no start-up penalty. The first model loads in a
  fraction of a second; the rest reuse the shared environment.
- **Easier:** the transcript is a typed value (`ml::asr::AsrResult`) instead
  of a JSON document parsed at runtime, so a malformed runner payload can no
  longer break the pipeline.
- **Easier:** the whole inference stack is testable from Rust — the golden
  tests in `tests/asr_golden.rs` drive the models directly.
- **Harder:** front-ends that the reference implementation got for free from
  `torch` or `transformers` have to be reimplemented. The Whisper mel filter
  bank, the NeMo pre-emphasis/dither front-end and the streaming chunk
  layout of the Nemotron encoder all live in `src/ml/features.rs` and
  `src/ml/nemotron.rs` now.
- **Harder:** the numerical behaviour is only as good as the ONNX exports.
  Where a graph was not published — the Nemotron export ships no feature
  extractor — the front-end had to be reconstructed from the model card and
  validated against the reference implementation.
- **Given up:** the ability to fall back to a `torch` runtime when an ONNX
  export is missing or broken. Every model must have a usable ONNX export.
- **Migration:** transcripts written by the Python implementation stay
  readable; the search index is a different format and is rebuilt on the
  first run (see [adr/06-store-the-transcript-search-index-in-redb.md](06-store-the-transcript-search-index-in-redb.md)).

## Alternatives considered

- **Keep the per-model virtualenvs and drive them as child processes.**
  Rejected because it preserves the start-up cost, the duplicated
  checkpoints and the untyped JSON protocol — the exact costs this port
  exists to remove.
- **Use `candle` instead of ONNX Runtime.** Rejected because `candle` would
  require re-implementing the model architectures from their papers, while
  ONNX Runtime executes the *published* exports of the very checkpoints the
  Python implementation used. Fidelity wins over re-derivation.
- **Call a Python sidecar through a socket.** Rejected: it keeps the
  dependency tree and the protocol, only moving the process boundary.
