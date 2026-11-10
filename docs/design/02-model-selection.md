# 02. Model selection and routing

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Establishes which model serves which language, and how the decision is made.
- **Feeds into:** [adr/04-route-asr-by-the-detected-language.md](../adr/04-route-asr-by-the-detected-language.md), [models/01-model-registry.md](../models/01-model-registry.md)

## Flow

The routing decision lives in two places: the table that maps a language code
to a model key, and the resolution logic that decides whether the language
is known at all.

1. **Normalise the hint.** `pipeline::normalize_language` turns the
   `--language` value into either `"auto"` or an ISO-ish code. `"auto"`,
   `"detect"` and the empty string all mean "detect"; a language *name*
   ("Russian") is resolved through the 45-entry table in
   `src/languages.rs`.
2. **Resolve the model.** `pipeline::resolve_model`:
   - with `--model KEY`, that model is used and the reported language code
     follows the model's role (`ru`, `en` or `und`);
   - with a concrete language code, `ml::asr::model_for_language` maps it to
     a model key;
   - with `"auto"`, the classifier votes.
3. **Vote.** `ml::langid::detect_language` splits the waveform into
   6-second windows, runs the embedded graph on each, averages the
   posteriors over the non-silent windows, and tallies the votes per class.
   The winner is *confident* when its mean posterior is at least `0.6` and
   it holds at least `0.5` of the summed votes.
4. **Fall back.** An unconfident result routes to the multilingual model.
   The detected code is still reported, but the transcription is done by
   Nemotron.

The routing table:

| Code | Model key | Role | Why |
|---|---|---|---|
| `ru` | `qwen3-asr-0.6b` | Russian | WER 5.13 on the Russian test set — the best 0.6B result |
| `en` | `parakeet-tdt-0.6b-v3` | English | Open ASR #1 at 0.6B scale |
| everything else | `nemotron-3.5-asr-streaming-0.6b` | multilingual | 40 languages, int8 graphs, one model for the tail |

The routing never resolves to `qwen3-asr-0.6b-int4`: that key is the
optional quantised variant of the Russian model, reachable only through an
explicit `--model`.

## Constants

| Constant | Value | Owner |
|---|---|---|
| Language-ID confidence threshold | 0.6 | `ml::langid::LANGID_CONFIDENCE_THRESHOLD` |
| Language-ID vote-share threshold | 0.5 | `ml::langid::LANGID_VOTE_THRESHOLD` |
| Language-ID window | 6 s | `ml::langid::LANGID_WINDOW_SECONDS` |
| Maximum windows analysed | 16 | `ml::langid::LANGID_MAX_WINDOWS` |
| Classifier classes | 45 | `ml::langid::LABELS` |

## Output shape

`resolve_model` returns the pair `(model_key, language_code)`. The code
decides *which* model runs; the language that the artefacts carry is the one
the reference implementation reports — see
[adr/11-report-the-language-the-way-the-reference-does.md](../adr/11-report-the-language-the-way-the-reference-does.md):

| Backend | `language` | `language_name` |
|---|---|---|
| `parakeet-tdt-0.6b-v3` | `en` | `English` |
| `nemotron-3.5-asr-streaming-0.6b` | `auto` | `auto` |
| `qwen3-asr-0.6b` | the code the model announced, else the forced code, else `undetermined` | the matching name |

A Nemotron run therefore writes:

```json
{
  "model": "nemotron-3.5-asr-streaming-0.6b",
  "role": "",
  "language": "auto",
  "language_name": "auto",
  "text": "…",
  "segments": [ … ]
}
```

The Qwen backend is the only one that says something about the audio: it
*detects* the language itself and reports it, so `transcribe_qwen` overrides
the routed code with the model's own answer (or with the forced code, or
with `undetermined` when neither is available).

## Failure modes

- **The classifier is unconfident.** The multilingual model is used and a
  warning is logged. The run succeeds.
- **`--model` names an unknown key.** `Error::Model` with the list of valid
  keys. The check happens *before* the download, so a typo does not cost a
  media download.
- **The classifier graph is missing.** It is embedded in the binary, so this
  cannot happen; the `langid-ecapa` entry in the registry exists only to
  report the model consistently in `lc-rs model list`.
