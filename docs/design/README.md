# Design documents

How the mechanisms of `lc-rs` actually work: the flow, the constants, the
artefacts they produce and the way they fail.

| # | Title | Purpose |
|---|---|---|
| [01-pipeline-stages.md](01-pipeline-stages.md) | The end-to-end five-stage flow and the files it produces |
| [02-model-selection.md](02-model-selection.md) | Which model serves which language, and how the decision is made |
| [03-qwen3-asr-backend.md](03-qwen3-asr-backend.md) | The Russian checkpoint: front-end, encoder, prompt, KV-cache loop |
| [04-parakeet-tdt-backend.md](04-parakeet-tdt-backend.md) | The English checkpoint: the greedy TDT transducer search |
| [05-nemotron-streaming-backend.md](05-nemotron-streaming-backend.md) | The multilingual checkpoint: the streaming chunk layout |
| [06-output-formats.md](06-output-formats.md) | How a transcript is serialised, and how chapters are folded in |
| [07-batch-and-playlists.md](07-batch-and-playlists.md) | How several sources are transcribed in one invocation |
| [08-chapters-and-search.md](08-chapters-and-search.md) | How chapters are persisted and how past transcripts are searched |
| [09-local-summaries.md](09-local-summaries.md) | The optional Markdown summary from a local Ollama instance |

Each document follows the same shape: **Flow** (the numbered steps, naming
the module that owns each one), **Constants** (the exact values), **Output
shape** (the files and schemas) and **Failure modes** (what breaks and how
it is signalled).

The decisions behind the mechanisms live in [../adr](../adr/README.md).
