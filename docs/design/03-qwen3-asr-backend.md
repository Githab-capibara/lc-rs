# 03. Qwen3-ASR backend

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Establishes how the Russian ASR checkpoint is executed, graph by graph.
- **Feeds into:** [adr/03-adopt-per-model-onnx-exports.md](../adr/03-adopt-per-model-onnx-exports.md), [adr/10-ship-the-fp32-qwen-decoder.md](../adr/10-ship-the-fp32-qwen-decoder.md), [adr/09-reproduce-the-reference-front-end-quirks.md](../adr/09-reproduce-the-reference-front-end-quirks.md)

## Flow

The checkpoint is an audio encoder plus an autoregressive language model,
exported as three graphs plus two side files:

| File | Role |
|---|---|
| `encoder.onnx` | 128-band log-mel features → 1024-dimensional audio embeddings |
| `decoder_init.onnx` | prompt + audio embeddings → the first token and the KV cache |
| `decoder_step.onnx` | one token + the KV cache → the next token and the updated cache |
| `decoder_weights.data` | the fp32 weights of both decoder graphs (external data) |
| `embed_tokens.bin` | the embedding table, raw little-endian `float16`, `[151936, 1024]` |
| `tokenizer.json` | the BPE tokenizer |
| `config.json`, `preprocessor_config.json` | the geometry and the front-end parameters |

The quantised export has the same shape with `decoder_init.int4.onnx`,
`decoder_step.int4.onnx` and `decoder_weights.int4.data`. The loader takes
the quantised graphs when the directory holds them and the fp32 ones
otherwise, so the same code path serves both registry keys. The fp32
decoder is the default; see
[adr/10-ship-the-fp32-qwen-decoder.md](../adr/10-ship-the-fp32-qwen-decoder.md).

The backend lives in `src/ml/qwen.rs` and runs in five steps.

1. **Front-end** (`compute_features`). The window is truncated to
   `N_SAMPLES = 480_000` samples (30 s at 16 kHz) and zero-padded to that
   length. The log-mel spectrogram is computed with the Whisper
   configuration — 128 bands, `n_fft = 400`, `hop = 160`, the **Slaney** mel
   scale, the **Slaney** filter normalisation, a `log10` with a dynamic-range
   clamp of 8 and the affine rescaling `(x + 4) / 4`. The first
   `NB_MAX_FRAMES = 3000` frames are kept.

   The Slaney scale *and* the Slaney normalisation are not interchangeable
   with the HTK variants: `transformers`' `WhisperFeatureExtractor` builds
   its filter bank with `mel_filter_bank(..., norm="slaney",
   mel_scale="slaney")`, and an HTK filter bank produces a different encoder
   input and therefore a different transcript.

2. **Encoder** (`encode`). The mel matrix is fed in band-major layout,
   `[1, 128, 3000]`, matching the graph's input contract. The output is
   `[1, 375, 1024]` — the encoder downsamples the time axis by 8.

3. **Prompt** (`build_prompt`). The prompt is assembled as text and encoded
   with the tokenizer:

   ```text
   <|im_start|>user\n<|audio_start|><|audio_pad|>×375<|audio_end|><|im_end|>\n<|im_start|>assistant\n
   ```

   The audio placeholder count equals the number of audio embeddings, so
   the `ScatterElements` node of the decoder graph writes the audio
   embeddings exactly onto the `<|audio_pad|>` positions. The `audio_offset`
   input carries the index of the first `<|audio_pad|>` token (3).

   When a language is forced, the assistant turn continues with
   `language {Name}<asr_text>`; otherwise the model emits that metadata
   itself and the transcript follows the tag.

4. **Prefill and decode** (`prefill`, `step`). `decoder_init` consumes the
   whole prompt at once and returns the logits of the last position plus
   the 28-layer KV cache. The first token is the arg-max of those logits.
   Every subsequent token is produced by `decoder_step`, which takes the
   embedding of the previous token (read from `embed_tokens.bin`),
   the next position and the cache.

   The graphs compute the M-RoPE internally from the two-dimensional
   `position_ids`, so the positions are simply `0, 1, 2, …`.

5. **Parsing** (`parse_output`). The generated ids are decoded with the
   tokenizer, the degenerate repetitions are collapsed, and the metadata
   before `<asr_text>` is separated from the transcript after it. The
   language is taken from the metadata — `language Russian` → `Russian` —
   and normalised to a code by `qwen::language_code`.

## Constants

| Constant | Value | Owner |
|---|---|---|
| Front-end samples | 480 000 (30 s) | `qwen::N_SAMPLES` |
| Mel frames kept | 3 000 | `qwen::NB_MAX_FRAMES` |
| Mel bands | 128 | `qwen::NUM_MELS` |
| Maximum new tokens | 1 024 | `qwen::MAX_NEW_TOKENS` |
| Decoder hidden size | 1 024 | `config.json` → `decoder.hidden_size` |
| KV cache | 28 layers × 8 heads × 128 dims | `config.json` |
| Repetition threshold | 20 | `qwen::repair_repetitions` |

## Output shape

```rust
pub struct QwenResult {
    pub text: String,          // the transcript
    pub language: Option<String>, // "Russian", …
}
```

`ml::asr::transcribe_qwen` turns the name into a code (`ru`) and falls back
to the forced language, and then to `undetermined`, when the model stated no
language — the same chain as the reference runner.

## Failure modes

- **A window is longer than 30 s.** The tail is dropped, exactly as in the
  reference implementation. See
  [adr/09-reproduce-the-reference-front-end-quirks.md](../adr/09-reproduce-the-reference-front-end-quirks.md).
- **The model emits a repetition loop.** The repetition repair collapses
  runs of more than 20 identical characters and patterns repeated at least
  20 times.
- **`embed_tokens.bin` has the wrong size.** `Error::Model` — the file must
  be exactly `vocab_size × hidden_size × 2` bytes.
- **The configuration disagrees with itself** (the encoder claims a
  different number of mel bands than the mel section). `Error::Model`.
- **The model reaches the token limit.** The generation stops after 1 024
  tokens and the transcript is returned as it stands.

## Validation

The front-end is validated numerically against `transformers` 4.57.6: the
maximum absolute deviation over a 6-second fixture is `1.1e-4` and the mean
deviation is `5.2e-7`, which is the `float32` rounding of the reference
computation. The check lives in `tests/features_parity.rs` and compares
against a fixture captured from the reference extractor.

The decoder is validated against the reference implementation's own
transcript of the same fixture: the word error rate is 0.267 for the fp32
export and 0.533 for the int4 one. The measurements, the method and the
decision are in
[adr/10-ship-the-fp32-qwen-decoder.md](../adr/10-ship-the-fp32-qwen-decoder.md).

The end-to-end behaviour is checked in `tests/asr_golden.rs`:

```text
qwen_transcribes_the_ukrainian_sample          ok
qwen_detects_the_language_when_it_is_not_forced ok
```
