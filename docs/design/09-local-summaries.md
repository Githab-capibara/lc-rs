# 09. Local summaries

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Establishes how the optional Markdown summary is produced.
- **Feeds into:** [design/01-pipeline-stages.md](01-pipeline-stages.md), [api/02-configuration.md](../api/02-configuration.md)

## Flow

`--summary` asks a local Ollama instance to summarise the transcript. The
whole feature is optional and best-effort: the transcript pipeline succeeds
whether or not the summary is produced.

1. **Read the transcript.** The `.txt` artefact is read; in the default
   single-file mode the Markdown artefact is used instead, because that is
   the file that exists.
2. **Build the prompt.** The built-in template (`assets/summary_prompt.txt`)
   receives the transcript. `--summary-prompt FILE` replaces the template.
3. **POST to Ollama.** The request goes to
   `http://localhost:11434/api/generate` with the model name from
   `--summary-model` (default `llama3.1:8b`). The request has a hard
   timeout of 120 seconds.
4. **Write the summary.** The response is written next to the transcript as
   `<stem>.summary.md`.

## Constants

| Constant | Value | Owner |
|---|---|---|
| Endpoint | `http://localhost:11434/api/generate` | `config::DEFAULT_SUMMARY_ENDPOINT` |
| Model | `llama3.1:8b` | `config::DEFAULT_SUMMARY_MODEL` |
| Timeout | 120 s | `config::DEFAULT_SUMMARY_TIMEOUT_SECS` |
| Built-in prompt | `assets/summary_prompt.txt` | `summary::built_in_prompt` |

## Output shape

```markdown
The speaker discusses how the country can support its citizens, and what
each person can do in return.
```

The file is written next to the transcript, in the output directory.

## Failure modes

Every failure of this stage is a **warning**, never an error. The transcript
has already been written at this point, and losing it because Ollama is
down would be absurd.

| Failure | What the user sees |
|---|---|
| Ollama is not running | a warning; the run succeeds |
| The model is not pulled | a warning naming the missing model |
| The request times out | a warning; the run succeeds |
| The transcript is too long for the model's context | a warning; the run succeeds |

## Privacy

The summary is produced by a *local* Ollama instance, and the request
never leaves the machine — the loopback host is explicitly excluded from
the proxy (`config::NO_PROXY_HOSTS`). No transcript is sent to a cloud
service, which is the same promise the rest of the tool makes.
