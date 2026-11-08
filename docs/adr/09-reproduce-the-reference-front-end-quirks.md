# 09. Reproduce the reference implementation's front-end quirks

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [design/03-qwen3-asr-backend.md](../design/03-qwen3-asr-backend.md), [adr/03-adopt-per-model-onnx-exports.md](03-adopt-per-model-onnx-exports.md)

## Context

A port that aims at parity has to decide what to do with the *accidents* of
the reference implementation — the behaviours that are not obviously
intended, but that the users' existing transcripts were produced with.

The Qwen3-ASR path has the sharpest one. The reference implementation feeds
the model 45-second windows, but the Whisper front-end it uses truncates
the waveform to 30 seconds (`n_samples = 480000`) and pads the mel
spectrogram to 3000 frames. The consequence is that **15 seconds of every
45-second window are silently discarded**: the audio between 30 s and 45 s
of a window never reaches the encoder.

Verified empirically against `transformers` 4.57.6:

```text
mel shape for 45 s of audio: (1, 128, 3000)
mel shape for  6 s of audio: (1, 128, 3000)
max diff on the common part: 0.0
```

The 45-second input and the 6-second input produce *identical* mel
spectra over their common range — the tail is cut, not merely padded.

There are two defensible positions:

1. **Reproduce the quirk.** The tool behaves exactly like the reference:
   same windows, same truncation, same transcripts.
2. **Fix the quirk.** Feed the whole 45-second window to the encoder, so
   that no audio is lost.

The second is objectively better *as a recogniser*. The first is what a
parity port does.

## Decision

Reproduce the reference behaviour exactly, and document it.

`src/ml/qwen.rs` truncates the window to `N_SAMPLES = 480_000` samples
(30 s at 16 kHz), pads it with zeros to that length, computes the mel
spectrogram and keeps the first `NB_MAX_FRAMES = 3000` frames — the same
sequence of operations as `WhisperFeatureExtractor.__call__` with
`padding="max_length", truncation=True`.

The same principle governs the other quirks that were identified:

- the repetition repair (`detect_and_fix_repetitions`) that the reference
  applies to the raw model output is reimplemented, including the
  threshold of 20 and the pattern-collapse rule;
- the language metadata is parsed with the same precedence rules, and an
  unparseable language is reported as `undetermined`;
- the `txt`-points-at-`md` artefact behaviour is preserved.

The quirk is called out explicitly in the module documentation of
`src/ml/qwen.rs` and in [design/03-qwen3-asr-backend.md](../design/03-qwen3-asr-backend.md),
so that a future maintainer can find it and change it deliberately.

## Consequences

- **Easier:** the transcripts are directly comparable with the ones the
  Python implementation produced. A transcript can be checked against the
  reference run without accounting for a systematic difference.
- **Easier:** the quirk is documented in one place, so it can be changed
  deliberately rather than by accident.
- **Harder:** the tool loses 15 seconds of every 45-second window when it
  uses Qwen. The lost audio is *not* silent — it is simply absent from the
  transcript. This is a defect of the reference implementation, inherited
  on purpose.
- **Harder:** the front-end has to reproduce the exact truncation and
  padding sequence, not merely "produce 30 seconds of mel features".
- **Given up:** 15 seconds of audio per window, and the chance to silently
  improve on the reference.
- **Migration:** none. The behaviour matches the reference from the first
  release.

## Alternatives considered

- **Feed the full 45-second window to the encoder.** Rejected: it changes
  the transcript in a way that cannot be attributed to the port, and the
  encoder's positional embeddings are tied to the 30-second geometry of its
  training data — a 45-second mel would also be out of distribution.
- **Reproduce the quirk but warn the user.** Rejected: the warning would
  appear on every Russian run and would not be actionable.
- **Ignore the difference and hope it does not matter.** Rejected: it is
  exactly the kind of silent divergence that makes a port untrustworthy.
