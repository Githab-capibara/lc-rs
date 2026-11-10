# 04. Parakeet TDT backend

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Establishes how the English ASR checkpoint is executed.
- **Feeds into:** [adr/03-adopt-per-model-onnx-exports.md](../adr/03-adopt-per-model-onnx-exports.md)

## Flow

The checkpoint is NVIDIA's Parakeet TDT 0.6B v3, exported as three graphs
plus the vocabulary:

| File | Role |
|---|---|
| `nemo128.onnx` | waveform → 128-band log-mel features, with NeMo's utterance normalisation baked in |
| `encoder-model.int8.onnx` | features → encoder states |
| `decoder_joint-model.int8.onnx` | encoder frame + decoder state → token logits and duration logits |
| `vocab.txt` | the sentencepiece vocabulary |
| `config.json` | the model geometry |

The backend lives in `src/ml/parakeet.rs`.

1. **Front-end.** The waveform is passed to `nemo128.onnx` as a single
   `[1, samples]` tensor. The graph performs the pre-emphasis, the STFT and
   the log-mel compression itself, including the utterance-level
   normalisation, so the Rust side does not have to reproduce it.
2. **Encoder.** The features are encoded into a sequence of frames.
3. **Greedy TDT search.** The joint network is evaluated frame by frame. At
   each frame it predicts a token *and* a duration; the duration says how
   many frames to skip before the next prediction. A frame whose token is
   the blank symbol ends the utterance.

   The durations are `[0, 1, 2, 3, 4]` — the TDT head predicts a duration
   class, not a frame count — and the decoding is capped at 10 tokens per
   step, as in the reference implementation shipped with `onnx-asr`.
4. **Text assembly.** The token pieces are concatenated and the spaces are
   normalised (`parakeet::normalize_spaces`), which removes the artefacts of
   the sentencepiece segmentation.

## Constants

| Constant | Value | Owner |
|---|---|---|
| Duration classes | `[0, 1, 2, 3, 4]` | `parakeet::DURATIONS` |
| Maximum tokens per step | 10 | `parakeet::MAX_TOKENS_PER_STEP` |
| Frame step | 0.08 s | `parakeet::FRAME_STEP_SECONDS` |
| Vocabulary size | 8 193 (8 192 + blank) | the `vocab.txt` |

## Output shape

```rust
pub struct ParakeetResult {
    pub text: String,
}
```

The English model does not detect a language, and the reference
implementation reports `en` for this checkpoint, so the artefacts carry `en`
and `English` — whatever the routed code was.

## Failure modes

- **The vocabulary does not match the graph.** The token ids in `vocab.txt`
  are used to index the sentencepiece pieces; a mismatch produces
  `Error::Model` when the vocabulary is loaded.
- **The joint network predicts a duration beyond the encoder length.** The
  search stops at the last frame and the partial transcript is returned.
- **The utterance is silent.** The search ends immediately and the result is
  an empty text; the pipeline skips empty windows.

## Validation

The joint network and the decoding loop are bit-faithful to the reference
implementation, and the end-to-end result is checked in
`tests/asr_golden.rs`:

```text
parakeet_transcribes_the_reference_english_sample ok
```

The expectations are substring checks on stable phrases rather than exact
matches: the fixture is an 8 kHz recording, and the interpolation filter of
the resampler changes which words the model hears.
