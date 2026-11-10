# 11. Report the language the way the reference does

- **Status:** Accepted
- **Date:** 2026-11-10
- **Deciders:** @Githab-capibara
- **Related:** [adr/09-reproduce-the-reference-front-end-quirks.md](09-reproduce-the-reference-front-end-quirks.md), [adr/04-route-asr-by-the-detected-language.md](04-route-asr-by-the-detected-language.md), [design/02-model-selection.md](../design/02-model-selection.md)

## Context

The reference implementation writes the language into every transcript
artefact and into the closing log line:

```python
detected = str(payload.get("language") or detected_language or code)
...
log.info(f"done ({name_for_code(transcripts.language)}, {transcripts.model})")
```

The value that ends up there is a property of the *runner*, not of the audio.
`runners/nvidia.py` hard-codes it:

```python
if args.model_key == PARAKEET:
    decode = _load_parakeet(args.model)
    lang = "en"
else:
    decode = _load_nemotron(args.model)
    lang = "auto"
```

So the Parakeet run always reports `en` — the checkpoint is English-only —
and the Nemotron run always reports `auto`, even when the routing stage
knew the language and even when `--language` named one. The Nemotron runner
does not even pass the hint to the model: the processor call hard-codes
`language="auto"`, so the checkpoint picks the language itself.

Only the Qwen runner reports something about the audio. It keeps the forced
language as its starting value, lets the model override it, normalises the
name through its own twelve-language table and falls back to `undetermined`
when the model says nothing at all.

The first implementation of this port reported the *routing* code instead:
`uk` for a Ukrainian file, `Ukrainian` in the log line. That is more
informative — and it is exactly the kind of silent divergence ADR 09 exists
to prevent. A transcript produced by `lc-rs` could not be compared with the
reference artefact field by field, and the `done (…)` line differed even
when the transcript itself was identical.

## Decision

Reproduce the reference values exactly.

| Backend | Reported language | Language used by the model |
|---|---|---|
| Parakeet | `en` | English-only checkpoint |
| Nemotron | `auto` | automatic detection — the prompt is always `PROMPT_AUTO` |
| Qwen | the code the model announced, else the forced code, else `undetermined` | the forced language, if the checkpoint knows it |

`src/ml/asr.rs` owns the two constants (`PARAKEET_LANGUAGE`, `NEMOTRON_LANGUAGE`)
and the Qwen fallback chain; the routing code that `resolve_model` computes
keeps deciding *which* model runs, but it no longer leaks into the
artefacts.

## Consequences

- **Easier:** the artefacts of the two implementations are comparable field
  by field. `language`, `language_name` and the `done (…)` log line are
  identical for the same input, whatever the backend is.
- **Easier:** the JSON schema carries no second, port-specific meaning, so
  a consumer of the artefacts cannot tell which implementation wrote them.
- **Harder:** the artefacts no longer say which language the routing stage
  detected. A Nemotron transcript of a Ukrainian file says `auto`, which is
  true of the model but says nothing about the audio.
- **Harder:** `--language` no longer conditions the Nemotron checkpoint. The
  hint still decides *which* model runs — that is the routing — but the
  multilingual checkpoint always detects the language itself, exactly as in
  the reference.
- **Given up:** the informative routing code in the artefacts, and the
  ability to force the language of the multilingual checkpoint.
- **Migration:** none. The values were never released.

## Alternatives considered

- **Keep the informative routing code.** Rejected: it is a divergence from
  the reference in the exact place where a divergence is hardest to notice
  — a field that looks like it means the same thing in both tools.
- **Report the routing code in an additional field.** Rejected: it changes
  the artefact schema, which is part of the contract the port reproduces.
- **Report the language Nemotron detected.** Rejected: the checkpoint does
  not expose the detected language through this export, and inferring it
  from the transcript would be guesswork.
- **Force the language on Nemotron and report `auto` anyway.** Rejected: the
  artefacts would claim a mode the model was not running in.
