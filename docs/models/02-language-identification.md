# 02. Language identification

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Documents how the spoken language is detected, and how the detection feeds the routing.
- **Feeds into:** [design/02-model-selection.md](../design/02-model-selection.md), [adr/04-route-asr-by-the-detected-language.md](../adr/04-route-asr-by-the-detected-language.md)

## The model

The classifier is `speechbrain/lang-id-commonlanguage_ecapa`, exported as a
self-contained ONNX graph: the feature extraction is baked in, so the graph
takes raw waveform samples and returns the 45-way posterior. The graph is
embedded in the binary (`assets/langid/langid_inline.onnx`), so the
detection stage needs no download and no installation step.

## The flow

`ml::langid::detect_language` (`src/ml/langid.rs`):

1. **Read the WAV.** The normalised 16 kHz mono WAV of the pipeline is
   parsed.
2. **Window.** The waveform is split into 6-second windows. At most 16
   windows are analysed, so a long recording does not multiply the
   inference cost without bound.
3. **Infer.** Each window is passed through the graph as a `[1, samples]`
   tensor; the output is the posterior over the 45 classes.
4. **Average.** The posteriors of the non-silent windows are averaged into
   one distribution.
5. **Vote.** Each window also contributes a vote for its best class; the
   votes are tallied per class.
6. **Decide.** The winner is *confident* when both of the following hold:
   - the mean posterior of the winning class is at least `0.6`;
   - the winning class holds at least `0.5` of the summed votes.

The class labels are the 45 display names of the classifier, mapped to the
internal codes by `src/languages.rs` — including the region-qualified
pseudo-codes for the Chinese variants (`cnh` for Hakka Chin, and so on).

## Constants

| Constant | Value |
|---|---|
| Classes | 45 |
| Window | 6 s |
| Maximum windows | 16 |
| Confidence threshold | 0.6 |
| Vote-share threshold | 0.5 |

## Output shape

```rust
pub struct LangIdResult {
    pub name: String,   // "Ukrainian"
    pub code: String,   // "uk"
    pub score: f32,     // the mean posterior of the winner
    pub votes: BTreeMap<String, f32>, // the vote tally
}
```

`LangIdResult::confident()` reports whether the two thresholds were met.

## Failure modes

- **The detection is not confident.** The multilingual model is used and a
  warning is logged. The run succeeds.
- **The audio is shorter than one window.** The window is zero-padded; the
  graph still produces a posterior.
- **The audio is silent.** Every posterior is flat; the winner is
  arbitrary, the confidence is low, and the multilingual model is used.
- **The graph fails to load.** `Error::Onnx`. This cannot happen in
  practice, because the graph is embedded in the binary.
