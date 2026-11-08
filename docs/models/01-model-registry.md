# 01. Model registry

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Documents the four checkpoints, their provenance and how they are fetched.
- **Feeds into:** [adr/03-adopt-per-model-onnx-exports.md](../adr/03-adopt-per-model-onnx-exports.md), [design/02-model-selection.md](../design/02-model-selection.md)

## The registry

The registry lives in `src/models.rs`. Every entry names a Hugging Face
repository, a role, an approximate size and the list of artefacts with their
exact size and SHA-256.

| Key | Role | Repository | Size |
|---|---|---|---|
| `langid-ecapa` | language-id | embedded in the binary (exported from `speechbrain/lang-id-commonlanguage_ecapa`) | 84 MB |
| `qwen3-asr-0.6b` | russian | `andrewleech/qwen3-asr-0.6b-onnx` | 2 030 MB |
| `parakeet-tdt-0.6b-v3` | english | `istupakov/parakeet-tdt-0.6b-v3-onnx` | 670 MB |
| `nemotron-3.5-asr-streaming-0.6b` | multilingual | `DimQ1/nemotron-3.5-asr-streaming-0.6b-onnx-int8-cpu` | 1 021 MB |

## The artefacts

### `qwen3-asr-0.6b`

| File | Size | SHA-256 (first 16) |
|---|---|---|
| `encoder.onnx` | 745 762 694 | `3c027f880f677615` |
| `decoder_init.int4.onnx` | 355 390 | `6d54633bbb9e6b1e` |
| `decoder_step.int4.onnx` | 354 888 | `7eaa4f31d5eb6ae2` |
| `decoder_weights.int4.data` | 962 460 672 | `d68cd1c0695a7ba4` |
| `embed_tokens.bin` | 311 164 928 | `e80150119fa5f7e5` |
| `tokenizer.json` | 11 429 377 | `bd2a97b55c8f7f9c` |
| `config.json` | 1 250 | `df31c4689abe9d78` |
| `preprocessor_config.json` | 298 | `9c7c558a05f326fe` |

The full digests are in `src/models.rs`; they are verified on every
download.

### `parakeet-tdt-0.6b-v3`

| File | Size |
|---|---|
| `encoder-model.int8.onnx` | 652 183 999 |
| `decoder_joint-model.int8.onnx` | 18 202 004 |
| `nemo128.onnx` | 139 764 |
| `vocab.txt` | 93 939 |
| `config.json` | 97 |

### `nemotron-3.5-asr-streaming-0.6b`

| File | Size |
|---|---|
| `encoder.onnx` | 2 815 853 |
| `encoder.onnx.data` | 967 176 192 |
| `decoder.onnx` | 4 696 |
| `decoder.onnx.data` | 59 785 216 |
| `joint.onnx` | 2 136 |
| `joint.onnx.data` | 37 830 656 |
| `vocab.txt` | 77 112 |
| `genai_config.json` | 1 962 |

### `langid-ecapa`

The ECAPA-TDNN classifier is embedded in the binary
(`assets/langid/langid_inline.onnx`, 84 110 845 bytes), so the routing stage
needs no download at all. The graph was exported to ONNX from
`speechbrain/lang-id-commonlanguage_ecapa`; the class order is the one of
the shipped `label_encoder.txt`, which a unit test checks against the
compiled-in label table.

The registry entry exists so that `lc-rs model list` and `lc-rs doctor`
report it like every other model; `lc-rs model download langid-ecapa`
writes it into the model directory so the two paths agree.

## The downloader

`models::download_model` fetches every artefact of a checkpoint:

1. an artefact that is already present with the right size is skipped,
   unless `--force` was given;
2. the download goes to `<name>.part` in the model directory;
3. a resumed download sends a `Range` header; if the server ignores it, the
   partial file is discarded and the download starts over;
4. the size is checked, then the SHA-256 is computed and compared;
5. a mismatch or a truncation triggers a retry — four attempts in total;
6. the verified file is moved into place with an atomic rename.

A failed download therefore never leaves a half-written checkpoint behind,
and a corrupted artefact is detected rather than loaded.

The `langid-ecapa` graph is not downloaded: it is written from the embedded
bytes.

## Installation status

`models::missing_files` reports the artefacts that are absent or have the
wrong size. `lc-rs model list` uses it for the status column, and
`lc-rs doctor` uses it for the model checks.

## Failure modes

- **The checkpoint is incomplete.** `Error::Dependency` with the list of
  missing artefacts.
- **The checksum does not match.** The download is retried; after four
  attempts `Error::Install` names the file, the URL and the last error.
- **The network is unreachable.** `Error::Download` with the transport
  error.
- **The registry key is unknown.** `Error::Model` with the list of valid
  keys.
