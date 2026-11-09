# 03. Adopt per-model ONNX exports instead of a single unified runtime

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [adr/01-run-models-in-process-through-onnx-runtime.md](01-run-models-in-process-through-onnx-runtime.md), [models/01-model-registry.md](../models/01-model-registry.md), [design/02-model-selection.md](../design/02-model-selection.md)

## Context

The Python implementation installed one virtualenv per model family and ran
a PyTorch checkpoint in each. Its quality expectations were tied to those
checkpoints: the Russian model was Qwen3-ASR 0.6B in `bfloat16`, the English
model was NVIDIA Parakeet TDT 0.6B v3, the multilingual model was NVIDIA
Nemotron 3.5 ASR Streaming 0.6B.

A Rust port has to choose what to run. The obvious answer — reimplement the
three architectures in Rust and run them on a tensor framework — is a
multi-month project per model and would silently change the outputs, because
every reimplementation chooses its own numerics for attention, normalisation
and decoding.

The alternative is to run the *published ONNX exports* of the same
checkpoints. That keeps the model weights and the graph semantics identical
to what the Python implementation produced, and moves the work to the
front-end: whatever the export does not ship has to be reconstructed.

The exports are not uniform, and that shapes the whole design:

| Model | Export | What it ships | What we had to write |
|---|---|---|---|
| Language ID | self-contained ECAPA-TDNN graph | the whole front-end | the windowing and the vote tally |
| Parakeet | `nemo128` + encoder + decoder/joint | the mel front-end as its own graph | the TDT greedy search |
| Nemotron | encoder + decoder + joint, int8 | *no* feature extractor | the NeMo mel front-end and the streaming chunk layout |
| Qwen3-ASR | encoder + `decoder_init` + `decoder_step`, fp32 | *no* feature extractor, embeddings in a side file | the Whisper mel front-end, the prompt builder, the KV-cache loop |

Two of the four exports ship their own front-end; two do not. The Nemotron
and Qwen front-ends therefore had to be reconstructed from the model cards
and the reference implementations, and validated numerically.

## Decision

Each model is run from **its own published ONNX export**, and each export is
registered as a set of individually pinned artefacts (exact size and SHA-256)
in `src/models.rs`:

- `langid-ecapa` — the ECAPA-TDNN classifier, shipped *inside the binary*
  (`assets/langid/langid_inline.onnx`), so the routing stage needs no
  download at all;
- `parakeet-tdt-0.6b-v3` — `istupakov/parakeet-tdt-0.6b-v3-onnx`;
- `nemotron-3.5-asr-streaming-0.6b` — `DimQ1/nemotron-3.5-asr-streaming-0.6b-onnx-int8-cpu`;
- `qwen3-asr-0.6b` — `andrewleech/qwen3-asr-0.6b-onnx`, with the fp32
  decoder graphs plus `decoder_weights.data` and `embed_tokens.bin`; the
  quantised export is registered separately, under
  `qwen3-asr-0.6b-int4`, and is documented in
  [adr/10-ship-the-fp32-qwen-decoder.md](10-ship-the-fp32-qwen-decoder.md).

The front-ends that the exports omit are implemented in Rust on top of
`rustfft` (`src/ml/features.rs`), and each one is validated against the
reference implementation: the Whisper mel filter bank against
`transformers`' `WhisperFeatureExtractor` (max deviation 1.1e-4, see
`tests/features_parity.rs`), the NeMo front-end against the reference
transcript of the Nemotron export.

## Consequences

- **Easier:** the transcripts come from the same weights the Python
  implementation used, so a transcript produced by `lc-rs` is comparable to
  one produced by `localcaption` line by line.
- **Easier:** no model architecture is reimplemented. The port's risk is
  concentrated in the front-ends, which are small, testable and
  numerically comparable.
- **Easier:** every artefact is pinned by size and SHA-256, so a truncated
  or substituted download is detected instead of being loaded.
- **Harder:** the front-ends of Nemotron and Qwen are reconstructions. The
  Whisper front-end, for example, uses the Slaney mel scale *and* the
  Slaney filter normalisation — the HTK variant silently produces a
  different encoder input and therefore a different transcript.
- **Harder:** each export has its own quirks. The Nemotron encoder consumes
  a fixed 65-row mel window (9 cached rows plus 56 new ones) per chunk; the
  Qwen encoder consumes a fixed 3000-frame mel; the Parakeet encoder
  consumes the features of the whole utterance. These are contracts, not
  conventions, and they are documented per backend in `docs/design/`.
- **Harder:** the ONNX decoder is not the PyTorch checkpoint the reference
  implementation ran, so the transcripts are not character-identical. With
  the fp32 decoder the word error rate against the reference's own
  transcript is 0.267; with the quantised one it is 0.533, which is why the
  fp32 export is the default.
- **Given up:** the ability to run the exact `bfloat16` PyTorch weights
  without an ONNX export. No export was available for the language
  identifier in its original form, so the graph is embedded instead.
- **Migration:** the model registry of the Python implementation (PyTorch
  repositories) is replaced by the ONNX repositories above. Existing
  transcripts are unaffected; the checkpoints have to be downloaded again.

## Alternatives considered

- **Reimplement the architectures in Rust with `candle`.** Rejected: the
  work per model is enormous and the numerics would differ from the
  reference, so the parity goal would be unreachable.
- **Convert the PyTorch checkpoints to ONNX ourselves.** Rejected: the
  conversions would have to be reproduced and re-validated on every release
  of the upstream model, whereas the published exports are maintained by
  their authors and are pinned by checksum here.
- **Use one ONNX export per architecture family, published centrally.**
  Rejected: the per-model exports exist and are maintained; a central
  re-export would add a trust dependency without adding fidelity.
