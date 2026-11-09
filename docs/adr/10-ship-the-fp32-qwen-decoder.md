# 10. Ship the fp32 Qwen decoder and keep the int4 one as an option

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [models/01-model-registry.md](../models/01-model-registry.md), [design/03-qwen3-asr-backend.md](../design/03-qwen3-asr-backend.md), [adr/03-adopt-per-model-onnx-exports.md](03-adopt-per-model-onnx-exports.md)

## Context

The upstream export of Qwen3-ASR 0.6B publishes the decoder twice: once in
fp32 (`decoder_init.onnx`, `decoder_step.onnx`, `decoder_weights.data` —
3.0 GB) and once quantised to int4 (`decoder_init.int4.onnx`,
`decoder_step.int4.onnx`, `decoder_weights.int4.data` — 962 MB). The
encoder, the embedding table and the tokenizer are shared; upstream does not
quantise the encoder at all.

The registry has to pick one of them, and the choice was not obvious: the
int4 export is a third of the size, while the fp32 export is the
counterpart of what the reference implementation actually computes.

The reference implementation loads the checkpoint with
`dtype=torch.float32` — its runner asks for fp32 explicitly, even though
the published weights are bf16. The fp32 ONNX graphs are therefore the
numerical counterpart of the reference, and the int4 graphs are not.

Rather than argue from first principles, both exports were downloaded, the
digests were checked, and the same fixture was transcribed three times: by
the reference implementation in PyTorch, by the fp32 ONNX decoder and by the
int4 ONNX decoder.

The fixture is `tests/fixtures/nemotron_uk.wav` — twelve seconds of
Ukrainian speech — forced to the Russian language, on the CPU.

| | reference (PyTorch fp32) | ONNX fp32 | ONNX int4 |
|---|---|---|---|
| wall clock | 24.9 s | 18.2 s | 11.4 s |
| peak RSS | 6 373 MiB | 7 246 MiB | 3 359 MiB |
| decoder weights | — (3.0 GB in fp32) | 3 006 MB | 962 MB |
| whole checkpoint | 1 876 MB (bf16 on disk) | 4 074 MB | 2 038 MB |

The transcripts:

- **reference** — «Ты пытаешься твоя краина может zrobić для тебя.
  Попытаешься ты можешь zrobić для своей краины.»
- **ONNX fp32** — «Ты пытаешься, что твоя краина может сделать для тебя, а
  пытаешься, что ты можешь сделать для своей краины.»
- **ONNX int4** — «ты пытаешься, что твоя страна может сработать для тебя,
  ты пытаешься, что ты можешь сработать для своей страны.»

Counted against the reference's own transcript, the word error rate is
**0.267 for fp32** and **0.533 for int4**. The int4 decoder does not merely
rephrase: it replaces *краина* with *страна* and *сделать* with
*сработать* — a substitution of content, not of style.

The same ordering held on the other fixtures. On the German one the two
exports agree with each other and with the reference; on the Spanish one
the fp32 export reproduces the second sentence of the speaker
(«pregunta qué puedes hacer tú por tu país») where the int4 export drops
the object.

Two observations matter for the decision:

1. The reference implementation itself needs **6.4 GiB** of resident memory
   on the CPU. The fp32 port needs 7.2 GiB — it is *not* out of line with
   the thing it replaces.
2. The fp32 port is already **1.37× faster** than the reference, because
   ONNX Runtime has no Python in the loop.

## Decision

The registry ships the **fp32 decoder** under the key `qwen3-asr-0.6b`,
which is the key the Russian routing resolves to.

The quantised export stays available under the separate key
`qwen3-asr-0.6b-int4`, marked **optional** in the registry:

- `lc-rs model download --all` does not fetch it;
- `lc-rs doctor` does not fail when it is absent;
- `lc-rs model list` shows it, so that the choice is visible;
- `lc-rs --model qwen3-asr-0.6b-int4` selects it.

No flag is added. The command-line surface stays exactly as the reference
implementation has it; a variant is a registry key, and `--model` already
takes a key.

The loader accepts both layouts: a directory holding `*.int4.onnx` graphs
is loaded with them, a directory holding the plain graphs is loaded with
those.

## Consequences

- **Easier:** the Russian path is faithful to the reference: half the word
  errors of the quantised decoder, and the transcript of the flagship
  language matches the reference's own output.
- **Easier:** the default install is not slower and not much heavier than
  the reference implementation it replaces.
- **Harder:** the default download grows from 2.0 GB to 4.0 GB. Over Tor,
  with the rate limits of the CDN, that is hours rather than minutes.
- **Harder:** the resident memory of the Russian path grows from 3.4 GiB to
  7.2 GiB. A machine with 8 GB of RAM has to swap, and the quantised
  export is the answer for it.
- **Given up:** the smallest possible footprint. The int4 export is the
  only way to run the Russian model in 3.4 GiB, and it is kept for that.
- **Migration:** an installation that already holds the int4 files under
  `qwen3-asr-0.6b` must download the fp32 decoder into the same directory
  and move the int4 files to `qwen3-asr-0.6b-int4`. `lc-rs model download
  qwen3-asr-0.6b` does the first half; the second half is a move.

## Alternatives considered

- **Ship the int4 decoder only.** Rejected: it doubles the word error rate
  against the reference and substitutes content words. The saving is real,
  but it is not worth changing what the tool says.
- **Ship the fp32 decoder only, and drop the int4 export.** Rejected: a
  machine that cannot afford 7.2 GiB would then have no Russian model at
  all, while the artefact exists and works.
- **Add a `--precision` flag.** Rejected: it would add a flag the
  reference implementation does not have, and the registry already has a
  vocabulary for variants.
- **Export the decoder in bf16, which is what the published weights are.**
  Rejected: upstream publishes no bf16 export, and producing one would mean
  maintaining an export pipeline of our own — a dependency the port does
  not have.
