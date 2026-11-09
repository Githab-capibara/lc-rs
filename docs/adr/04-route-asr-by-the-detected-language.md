# 04. Route ASR by the detected language

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [design/02-model-selection.md](../design/02-model-selection.md), [models/02-language-identification.md](../models/02-language-identification.md), [adr/03-adopt-per-model-onnx-exports.md](03-adopt-per-model-onnx-exports.md)

## Context

No 0.6B-class checkpoint wins on every language. The best-known
specialised models are language-specific: Russian is dominated by
Qwen3-ASR 0.6B (WER 5.13 on the Russian test set), English by NVIDIA
Parakeet TDT 0.6B v3 (Open ASR #1 at the time of writing). A multilingual
model has to spend its capacity on every language it supports, and pays for
it in every language.

At the same time the tool must work on a plain CPU, must not require an API
key, and must not upload the audio anywhere. That rules out cloud
transcription and leaves exactly three checkpoints that fit the budget.

The user must nevertheless be able to override the choice: sometimes the
detector is wrong, sometimes the user knows the language and does not want
to wait for the detection stage.

## Decision

Detect the spoken language first, then route:

| Detected language | Model key | Role |
|---|---|---|
| `ru` | `qwen3-asr-0.6b` | Russian |
| `en` | `parakeet-tdt-0.6b-v3` | English |
| anything else, or unclear | `nemotron-3.5-asr-streaming-0.6b` | multilingual catch-all |

Detection uses the ECAPA-TDNN classifier (`langid-ecapa`, 45 languages,
embedded in the binary). A result is *confident* when the posterior of the
winning class is at least `0.6` **and** the winning class holds at least
`0.5` of the summed votes across the analysed windows; below that the
multilingual Nemotron model is used instead of guessing
(`src/ml/langid.rs`, `src/pipeline.rs::resolve_model`).

The user can bypass the routing in two ways:

- `--language CODE` (or a language name) forces the routing decision;
- `--model KEY` forces the model itself, and then the language code reported
  in the artefacts follows the model's role.

## Consequences

- **Easier:** the best-known WER per language family at 0.6B scale, while
  the multilingual fallback still covers any of the 45 detected languages.
- **Easier:** fully offline — the language-ID graph is embedded in the
  binary, the ASR checkpoints live under the data directory, and no API key
  exists anywhere.
- **Easier:** a deterministic, testable routing contract
  (`ml::asr::model_for_language`, `pipeline::resolve_model`).
- **Harder:** four checkpoints, ≈ 5.8 GB, on the first install.
- **Harder:** a low-confidence detection quietly uses the slower
  multilingual model. Accuracy is preferred over speed.
- **Harder:** the routing is only as good as the classifier. A recording in
  a language outside the 45-language set is routed to the multilingual
  model, which may transcribe it poorly.
- **Given up:** single-backend simplicity — one model for everything — since
  no 0.6B checkpoint wins on both Russian and English at once.
- **Migration:** none for stored transcripts; fresh runs pick up the routing
  automatically.

## Alternatives considered

- **A single multilingual model for everything.** Rejected: worse Russian
  and English WER at the same size class.
- **Route by the file name or by the user's flag only.** Rejected: it makes
  the default path depend on the user knowing the language, and the whole
  point of the classifier is that it does not.
- **Let the multilingual model report its own language and route on that.**
  Rejected: the language has to be known *before* the transcription, not
  after it.
