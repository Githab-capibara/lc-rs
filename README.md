<div align="center">

# lc-rs

**Fully-local transcription for YouTube and local files.**
Language-ID routing, three specialist ASR models, SRT/VTT/JSON out.
No API key. No Python. One binary.

<!-- Package -->
<p>
  <a href="https://crates.io/crates/lc-rs"><img src="https://img.shields.io/crates/v/lc-rs?logo=rust&logoColor=white&color=%23dea584&cacheSeconds=300" alt="crates.io version"></a>
  <a href="https://github.com/Githab-capibara/lc-rs/actions/workflows/ci.yml"><img src="https://github.com/Githab-capibara/lc-rs/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Githab-capibara/lc-rs?color=yellow" alt="License: MIT"></a>
</p>

<!-- Community -->
<p>
  <a href="https://github.com/Githab-capibara/lc-rs"><img src="https://img.shields.io/badge/website-Githab--capibara/lc--rs-8A5CF5?logo=github" alt="Website"></a>
  <a href="https://github.com/Githab-capibara/lc-rs/stargazers"><img src="https://img.shields.io/github/stars/Githab-capibara/lc-rs?style=flat&logo=github" alt="GitHub stars"></a>
  <a href="https://github.com/Githab-capibara/lc-rs/issues"><img src="https://img.shields.io/github/issues/Githab-capibara/lc-rs" alt="Open issues"></a>
  <a href="https://github.com/Githab-capibara/lc-rs/commits/main"><img src="https://img.shields.io/github/last-commit/Githab-capibara/lc-rs?logo=github" alt="Last commit"></a>
</p>

```bash
cargo install --path . && lc-rs doctor --fix
```

</div>

> [!TIP]
> Local, offline transcription for YouTube, Vimeo, Twitch and [1000+ other
> sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)
> via `yt-dlp`, plus any video or audio file on disk. Paste a URL or a
> path; get a `.md` transcript by default (`--output-format` adds `.txt`,
> `.srt`, `.vtt` or `.json`) without an API key and without uploading audio
> to the cloud. An ECAPA-TDNN language-ID model picks between
> [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B),
> [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and
> [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

`lc-rs` is a tiny orchestrator over four stages:

| Stage | Tool |
|---|---|
| Download the best audio | [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) (YouTube, Vimeo, Twitch, 1000+ sites) |
| Re-encode to 16 kHz mono WAV | [`ffmpeg`](https://ffmpeg.org/) |
| Detect the language | ECAPA-TDNN classifier (45 languages), embedded in the binary |
| Transcribe locally | Qwen3-ASR 0.6B · Parakeet TDT 0.6B v3 · Nemotron 3.5 ASR 0.6B |

Nothing is uploaded to a third-party service. No OpenAI / Google / DeepL
keys required. Unlike pasting a clip into ChatGPT or calling the Whisper
API, transcription stays on your laptop after the download.

This is a **from-scratch Rust port** of the Python project
`localcaption`. The command surface, the artefacts and the routing
decisions are one-to-one; the Python virtualenvs and the `torch` runtimes
are gone.

## Installation

```bash
git clone https://github.com/Githab-capibara/lc-rs
cd lc-rs
cargo build --release
install -Dm755 target/release/lc-rs ~/.local/bin/lc-rs

lc-rs doctor --fix          # install ffmpeg / yt-dlp, fetch the checkpoints
```

Requirements: Rust 1.98+, `ffmpeg`, `yt-dlp`, and ~3.8 GB of disk for the
four checkpoints. See [docs/operations/01-installation.md](docs/operations/01-installation.md).

### With CUDA

```bash
cargo build --release --features cuda
```

CUDA is opt-in; the default build has no CUDA dependency and cannot fail
because of a missing toolkit.

## Usage

```bash
# A YouTube video, language detected automatically.
lc-rs https://www.youtube.com/watch?v=dQw4w9WgXcQ

# A local file, Russian forced, every output format.
lc-rs lecture.mp4 -l ru --output-format all -o ./out

# A batch of videos, resumable after an interruption.
lc-rs --batch urls.txt -o ./out

# Search what you have transcribed so far.
lc-rs search "climate policy"
```

The transcript lands in `./transcripts/<stem>.md`:

```markdown
Питаю, що твоя країна може зробити для тебе. Я питаю, що ти можеш зробити для своєї країни.

## Timestamps

- `00:00:00,000` — Питаю, що твоя країна може зробити для тебе. Я питаю, що ти можеш зробити для своєї країни.
```

### Commands

| Command | Purpose |
|---|---|
| `lc-rs <url-or-file>` | transcribe a video (the default command) |
| `lc-rs --batch FILE` | transcribe every URL in a list, sequentially |
| `lc-rs doctor [--fix]` | diagnose the install and repair what is missing |
| `lc-rs model list\|download\|rm\|info` | manage the local checkpoints |
| `lc-rs search <term>` | search the past transcripts |

Every flag is documented in
[docs/api/01-command-line-interface.md](docs/api/01-command-line-interface.md).

## The models

| Model | Role | Size | Source |
|---|---|---|---|
| `langid-ecapa` | language identification, 45 languages | 84 MB | embedded in the binary |
| `qwen3-asr-0.6b` | Russian | 2.0 GB | [`andrewleech/qwen3-asr-0.6b-onnx`](https://huggingface.co/andrewleech/qwen3-asr-0.6b-onnx) |
| `parakeet-tdt-0.6b-v3` | English | 670 MB | [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) |
| `nemotron-3.5-asr-streaming-0.6b` | multilingual, 40 languages | 1.0 GB | [`DimQ1/nemotron-3.5-asr-streaming-0.6b-onnx-int8-cpu`](https://huggingface.co/DimQ1/nemotron-3.5-asr-streaming-0.6b-onnx-int8-cpu) |

```bash
lc-rs model download --all
```

Every artefact is pinned by size *and* SHA-256, so a truncated or
substituted download is detected instead of being loaded. The downloads are
resumable.

## Architecture

```text
             ┌──────────────┐
  URL ──────▶│    yt-dlp    │────▶ media file
             └──────────────┘
                    │
             ┌──────────────┐
             │    ffmpeg    │────▶ 16 kHz mono WAV
             └──────────────┘
                    │
             ┌──────────────┐
             │  langid      │────▶ language code
             │ (ECAPA-TDNN) │      ru ─▶ Qwen3-ASR
             └──────────────┘      en ─▶ Parakeet TDT
                    │             else ─▶ Nemotron 3.5
             ┌──────────────┐
             │  ONNX Runtime│────▶ segments + text
             └──────────────┘
                    │
             ┌──────────────┐
             │  formatter   │────▶ .md .txt .srt .vtt .json
             └──────────────┘
                    │
             ┌──────────────┐
             │  redb index  │────▶ lc-rs search TERM
             └──────────────┘
```

Every model runs **in-process** through ONNX Runtime: no virtualenv, no
`torch`, no child process, no JSON payload crossing a process boundary. See
[adr/01-run-models-in-process-through-onnx-runtime.md](docs/adr/01-run-models-in-process-through-onnx-runtime.md).

The pipeline stage by stage:
[docs/design/01-pipeline-stages.md](docs/design/01-pipeline-stages.md).

## Configuration

Every setting is an environment variable. The `LCRS_` prefix wins over the
original `LOCALCAPTION_` one, so an existing setup keeps working.

| Variable | Default | Purpose |
|---|---|---|
| `LCRS_PROXY` | `socks5h://127.0.0.1:9050` | the SOCKS5 proxy for every outbound request |
| `LCRS_MODELS_DIR` | `$XDG_DATA_HOME/lc-rs/models` | where the checkpoints live |
| `LCRS_INDEX_PATH` | `$XDG_DATA_HOME/lc-rs/index.redb` | where the search index lives |
| `LCRS_THREADS` | the number of logical CPUs | the ONNX Runtime thread count |
| `LCRS_CUDA` | — | force CPU (`0`) or GPU (`1`) in a CUDA build |

The full list is in
[docs/api/02-configuration.md](docs/api/02-configuration.md).

## Documentation

Everything lives in [`docs/`](docs) (master index:
[docs/README.md](docs/README.md)):

| Folder | Contents |
|---|---|
| [docs/adr](docs/adr/README.md) | Architecture Decision Records: in-process ONNX Runtime, the graph optimisation level, per-model exports, language routing, CLI parity, the search index, the proxy policy, CUDA, and the reproduced reference quirks |
| [docs/design](docs/design/README.md) | The mechanisms: pipeline stages, model selection, the three ASR backends, output formats, batch and playlists, chapters and search, local summaries |
| [docs/api](docs/api/README.md) | The CLI reference, the configuration and the error model |
| [docs/models](docs/models/README.md) | The model registry and the language identification |
| [docs/operations](docs/operations/README.md) | Installation and `doctor --fix` |

### Key entry points

| Page | What it answers |
|---|---|
| [docs/design/01-pipeline-stages.md](docs/design/01-pipeline-stages.md) | What does a run actually do, and what files come out? |
| [docs/api/01-command-line-interface.md](docs/api/01-command-line-interface.md) | Every flag and subcommand |
| [docs/adr/01-run-models-in-process-through-onnx-runtime.md](docs/adr/01-run-models-in-process-through-onnx-runtime.md) | Why is there no Python in this port? |
| [docs/adr/02-pin-the-onnx-graph-optimisation-level.md](docs/adr/02-pin-the-onnx-graph-optimisation-level.md) | Why is the optimisation level pinned? |
| [docs/design/03-qwen3-asr-backend.md](docs/design/03-qwen3-asr-backend.md) | How does the Russian model actually work? |
| [docs/adr/09-reproduce-the-reference-front-end-quirks.md](docs/adr/09-reproduce-the-reference-front-end-quirks.md) | Which reference behaviours are reproduced on purpose? |

## Development

```bash
cargo build --release
cargo test --all-features
cargo clippy --all-targets --all-features
cargo fmt --all
```

The crate denies warnings and forbids unsafe code:

```toml
[lints.rust]
unsafe_code = "forbid"
missing_debug_implementations = "deny"

[lints.clippy]
all = { level = "deny", priority = -1 }
pedantic = { level = "deny", priority = -1 }
```

The unit tests run without any model. The end-to-end tests need the
checkpoints and are marked `#[ignore]`:

```bash
lc-rs model download --all
cargo test --release -- --ignored
```

The mel front-end is checked numerically against the reference
`transformers` extractor (`tests/features_parity.rs`), and the transcripts
against the reference wording (`tests/asr_golden.rs`).

## Licence

MIT — see [LICENSE](LICENSE).

The model checkpoints are distributed by their respective authors under
their own licences.
