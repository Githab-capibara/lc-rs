# lc-rs documentation

The authoritative documentation for `lc-rs` — a fully-local
YouTube/file → transcript orchestrator written in Rust. `yt-dlp` fetches the
media, `ffmpeg` normalises it, an embedded ECAPA-TDNN classifier picks one
of three specialist ASR models, and everything runs in-process through ONNX
Runtime. All traffic goes through a SOCKS5 proxy, and no API key exists
anywhere.

## Start here

| Guide | Purpose |
|---|---|
| [design/01-pipeline-stages.md](design/01-pipeline-stages.md) | The end-to-end five-stage pipeline and the files it produces |
| [api/01-command-line-interface.md](api/01-command-line-interface.md) | Every CLI flag and subcommand |
| [api/02-configuration.md](api/02-configuration.md) | Every environment variable and the filesystem layout |
| [adr/01-run-models-in-process-through-onnx-runtime.md](adr/01-run-models-in-process-through-onnx-runtime.md) | Why the Python runtimes are gone |
| [repo root README.md](../README.md) | User-facing overview and install |

## Directory map

| Folder | What lives here | README |
|---|---|---|
| [`adr/`](adr/) | Architectural Decision Records (Michael Nygard format) | [adr/README.md](adr/README.md) |
| [`design/`](design/) | Design documents: pipeline, model selection, the three ASR backends, output formats, batch, chapters, summaries | [design/README.md](design/README.md) |
| [`api/`](api/) | CLI reference, configuration, error model | [api/README.md](api/README.md) |
| [`models/`](models/) | Model registry and language identification | [models/README.md](models/README.md) |
| [`operations/`](operations/) | Installation and `doctor --fix` | [operations/README.md](operations/README.md) |

## Key entry points

| Page | What it answers |
|---|---|
| [design/01-pipeline-stages.md](design/01-pipeline-stages.md) | What does a run actually do, and what files come out? |
| [design/02-model-selection.md](design/02-model-selection.md) | Which model serves which language? |
| [design/03-qwen3-asr-backend.md](design/03-qwen3-asr-backend.md) | How does the Russian model actually work? |
| [adr/02-pin-the-onnx-graph-optimisation-level.md](adr/02-pin-the-onnx-graph-optimisation-level.md) | Why is the graph optimisation level pinned to `ORT_ENABLE_ALL`? |
| [adr/09-reproduce-the-reference-front-end-quirks.md](adr/09-reproduce-the-reference-front-end-quirks.md) | Which reference behaviours are reproduced on purpose? |
| [api/03-error-handling.md](api/03-error-handling.md) | What exit codes and error variants exist? |
| [operations/02-doctor-self-healing.md](operations/02-doctor-self-healing.md) | What does `lc-rs doctor --fix` do? |

## The module map

The crate is organised as follows; the documentation above follows the same
division.

| Module | Responsibility |
|---|---|
| `src/cli.rs` | the command-line definitions and the dispatch |
| `src/config.rs` | the environment-driven configuration |
| `src/pipeline.rs` | the end-to-end orchestration |
| `src/downloader.rs` | the `yt-dlp` driver |
| `src/audio.rs` | the `ffmpeg` driver and the 16 kHz mono normalisation |
| `src/ml/` | the ONNX Runtime backends: `langid`, `parakeet`, `nemotron`, `qwen`, and the shared feature extractors |
| `src/formatter.rs` | the transcript serialisers |
| `src/chapters.rs` | the chapter markers and the transcript slicing |
| `src/search.rs` | the redb-backed search index |
| `src/summary.rs` | the optional local Ollama summary |
| `src/models.rs` | the model registry and the downloader |
| `src/doctor.rs` | the diagnostics |
| `src/errors.rs` | the error taxonomy |

## Related project

`lc-rs` is a from-scratch Rust port of the Python project `localcaption`.
The command surface, the artefacts and the routing decisions are one-to-one;
the Python virtualenvs and the `torch` runtimes are gone. The decisions
that differ from the Python implementation are recorded in
[adr/](adr/README.md).
