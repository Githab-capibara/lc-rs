# 05. Nemotron streaming backend

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Establishes how the multilingual ASR checkpoint is executed.
- **Feeds into:** [adr/02-pin-the-onnx-graph-optimisation-level.md](../adr/02-pin-the-onnx-graph-optimisation-level.md), [adr/03-adopt-per-model-onnx-exports.md](../adr/03-adopt-per-model-onnx-exports.md)

## Flow

The checkpoint is NVIDIA's Nemotron 3.5 ASR Streaming 0.6B, exported as an
int8 CPU package:

| File | Role |
|---|---|
| `encoder.onnx` (+ `.data`) | 65-row mel chunk → encoder states |
| `decoder.onnx` (+ `.data`) | decoder state → token logits |
| `joint.onnx` (+ `.data`) | encoder frame + decoder state → token logits |
| `vocab.txt` | the sentencepiece vocabulary |
| `genai_config.json` | the manifest that documents the graph semantics |

The published export ships **no feature extractor**, so the front-end had
to be reconstructed. The backend lives in `src/ml/nemotron.rs` and runs in
four steps.

1. **Front-end** (`src/ml/nemotron.rs`). The waveform is converted to a
   NeMo-style log-mel spectrogram: pre-emphasis with a coefficient of 0.97,
   a 512-point FFT with a 400-sample window and a 160-sample hop, and a
   natural logarithm with an epsilon of `5.96e-8`. The FFT buffer is
   zero-padded to the transform width and the overlap is handled explicitly,
   because a 512-point transform of a 400-sample window needs the padding
   reset for every frame.
2. **Chunking.** The encoder is a *streaming* encoder: it consumes a fixed
   window of 65 mel rows, of which 9 are the cached rows of the previous
   chunk and 56 are new. Every chunk therefore advances the time axis by 56
   rows and produces 7 output frames.
3. **Decoding.** The greedy RNN-T search runs over the encoder frames: the
   joint network predicts a token and the blank symbol ends the utterance.
   The decoder state is carried between frames.
4. **Language prompt.** The model is conditioned on a language *token* — a
   single id from a 121-entry table (`nemotron::PROMPTS`). The id `101`
   (`PROMPT_AUTO`) means "detect". The prompt is dropped from the text
   assembly, so it never appears in the transcript.

## Constants

| Constant | Value | Owner |
|---|---|---|
| Mel bands | 128 | `nemotron::NUM_MELS` |
| Encoder hidden size | 1 024 | `nemotron::ENCODER_HIDDEN` |
| Decoder hidden size | 640 | `nemotron::DECODER_HIDDEN` |
| Chunk length | 8 960 samples (0.56 s) | `nemotron::CHUNK_SAMPLES` |
| Cached mel rows | 9 | `nemotron::PRE_CACHE_ROWS` |
| New mel rows per chunk | 56 | `nemotron::CHUNK_FRAMES` |
| Mel rows per encoder call | 65 | `nemotron::MEL_ROWS` |
| Encoder frames per chunk | 7 | `nemotron::CHUNK_OUTPUT_FRAMES` |
| FFT size | 512 | `nemotron::FFT_SIZE` |
| Window / hop | 400 / 160 | `nemotron::WIN_LENGTH`, `nemotron::HOP` |
| Pre-emphasis | 0.97 | `nemotron::PREEMPH` |
| Auto prompt id | 101 | `nemotron::PROMPT_AUTO` |

## Output shape

```rust
pub struct NemotronResult {
    pub text: String,
}
```

The model does not report a language, so the routed code is reported
unchanged.

## Failure modes

- **The audio is shorter than one chunk.** The chunk is zero-padded, so the
  encoder still produces its 7 frames.
- **The prompt id is unknown.** `nemotron::prompt_for_language` falls back
  to `PROMPT_AUTO`, which is the reference behaviour for an unknown code.
- **The graph was rewritten with a different optimisation level.** The
  transcripts differ in punctuation and in a few words. See
  [adr/02-pin-the-onnx-graph-optimisation-level.md](../adr/02-pin-the-onnx-graph-optimisation-level.md).

## Validation

The end-to-end result is checked in `tests/asr_golden.rs` for two languages:

```text
nemotron_transcribes_the_ukrainian_sample ok
nemotron_transcribes_the_german_sample     ok
```

The reference transcript of the Ukrainian fixture is:

```text
Питаю, що твоя країна може зробити для тебе. Я питаю, що ти можеш зробити для своєї країни.
```
