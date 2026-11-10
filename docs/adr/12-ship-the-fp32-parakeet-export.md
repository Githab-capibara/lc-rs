# 12. Ship the fp32 Parakeet export and keep the int8 one as an option

- **Status:** Accepted
- **Date:** 2026-11-10
- **Deciders:** @Githab-capibara
- **Related:** [adr/10-ship-the-fp32-qwen-decoder.md](10-ship-the-fp32-qwen-decoder.md), [models/01-model-registry.md](../models/01-model-registry.md), [design/04-parakeet-tdt-backend.md](../design/04-parakeet-tdt-backend.md)

## Context

The upstream export of NVIDIA Parakeet TDT 0.6B v3
(`istupakov/parakeet-tdt-0.6b-v3-onnx`) publishes the checkpoint twice, the
same way the Qwen export does:

| Export | Files | Size |
|---|---|---|
| fp32 | `encoder-model.onnx` + `encoder-model.onnx.data`, `decoder_joint-model.onnx` | 2 550 MB |
| int8 | `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx` | 670 MB |

The registry shipped the int8 export, because it was the smaller one and the
port had no measurement yet that would justify the other four fifths of the
download.

The English path is the one the reference implementation serves with
`nvidia/parakeet-tdt-0.6b-v3` in PyTorch, so the reference transcript of the
fixture is known exactly:

> The birch canoe slid on the smooth planks. Glue the sheet to the dark blue
> background. It is easy to tell the depth of a well. These days a chicken
> leg is a rare dish. Rice is often served in round bowls. The juice of
> lemons makes fine punch. The box was thrown beside the park truck. The
> hogs were fed chopped corn and garbage. Four hours of steady work faced
> us. A large size in stockings is hard to sell.

Both exports were downloaded, their digests checked and the same fixture
transcribed by all three implementations on the CPU:

| | reference (PyTorch) | ONNX fp32 | ONNX int8 |
|---|---|---|---|
| wall clock | 29.9 s | 8.5 s | 7.5 s |
| peak RSS | 3 317 MiB | 2 687 MiB | 1 522 MiB |
| checkpoint on disk | 2 393 MB | 2 432 MB | 639 MB |
| word error rate against the reference | — | **0.000** | 0.469 |

The fp32 export reproduces the reference transcript **word for word and
punctuation for punctuation**. The int8 export does not merely rephrase:

- «The birch canoe slid on the smooth planks» becomes «Birch can do it on
  the food plane»;
- «a chicken leg is a rare dish» is dropped entirely;
- «the hogs were fed chopped corn and garbage» becomes «The chocolate corn
  and garbage».

That is a substitution of content, the same failure mode that made the int4
Qwen decoder unusable as a default (ADR 10).

The port is not slower because of the fp32 weights: it is 3.5× faster than
the reference implementation and needs 630 MiB *less* resident memory, since
ONNX Runtime has no Python interpreter and no autograd state in the loop.

## Decision

The registry ships the **fp32 export** under the key `parakeet-tdt-0.6b-v3`,
which is the key the English routing resolves to.

The quantised export stays available under the separate key
`parakeet-tdt-0.6b-v3-int8`, marked **optional** in the registry:

- `lc-rs model download --all` does not fetch it;
- `lc-rs doctor` does not fail when it is absent;
- `lc-rs model list` shows it, so that the choice is visible;
- `lc-rs --model parakeet-tdt-0.6b-v3-int8` selects it.

No flag is added, for the same reason as in ADR 10: the command-line surface
stays exactly as the reference implementation has it, and a variant is a
registry key.

The loader accepts both layouts: a directory holding `encoder-model.onnx` and
`decoder_joint-model.onnx` is loaded with them, a directory holding the
`*.int8.onnx` graphs is loaded with those.

## Consequences

- **Easier:** the English path is faithful to the reference: the transcript
  of the fixture is character-identical to the one the Python implementation
  produced.
- **Easier:** the English path is the cheapest of the three in wall clock
  and in memory — the fp32 weights are offset by the absence of Python.
- **Harder:** the default download grows from 670 MB to 2.5 GB. Over Tor,
  with the rate limits of the CDN, that is an order of magnitude longer.
- **Harder:** the resident memory of the English path grows from 1.5 GiB to
  2.6 GiB. A machine that cannot afford that has the int8 export.
- **Given up:** the smallest possible footprint. The int8 export is the only
  way to run the English model in 1.5 GiB, and it is kept for that.
- **Migration:** an installation that already holds the int8 files under
  `parakeet-tdt-0.6b-v3` must download the fp32 files into the same
  directory and move the int8 files to `parakeet-tdt-0.6b-v3-int8`.
  `lc-rs model download parakeet-tdt-0.6b-v3` does the first half; the
  second half is a move.

## Alternatives considered

- **Ship the int8 export only.** Rejected: it replaces half the words of the
  reference transcript and drops a whole sentence. The saving is real, but
  it is not worth changing what the tool says.
- **Ship the fp32 export only, and drop the int8 one.** Rejected: a machine
  that cannot afford 2.6 GiB would then have no English model at all, while
  the artefact exists and works.
- **Convert the PyTorch checkpoint to ONNX ourselves, in fp32.** Rejected:
  upstream already publishes the fp32 export, pinned by digest, with the
  same front-end graph (`nemo128.onnx`) that this port uses.
- **Add a `--precision` flag.** Rejected: it would add a flag the reference
  implementation does not have, and the registry already has a vocabulary
  for variants.
