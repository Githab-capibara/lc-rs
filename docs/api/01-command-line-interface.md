# 01. Command-line interface

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Documents every command, flag and exit code of the `lc-rs` binary.
- **Feeds into:** [adr/05-keep-the-command-line-surface-one-to-one.md](../adr/05-keep-the-command-line-surface-one-to-one.md), [design/01-pipeline-stages.md](../design/01-pipeline-stages.md)

## Commands

```text
lc-rs <url-or-file> [options]    transcribe a video (default)
lc-rs --batch FILE [options]     transcribe a list of URLs/files
lc-rs doctor [--fix]             diagnose a local install
lc-rs model <subcommand>         list / download / remove models
lc-rs search <term>              search past transcripts
```

The transcription command is the default. `lc-rs transcribe URL` is accepted
as well and behaves identically — the subcommands are dispatched by hand in
`src/cli.rs` so that the URL form *is* the transcribe form.

## Transcription flags

| Flag | Default | Purpose |
|---|---|---|
| `<URL>` | — | the source: a `YouTube` URL, any URL `yt-dlp` supports, or a path to a local video/audio file |
| `--batch <FILE>` | — | transcribe every non-empty, non-`#` line of `FILE` sequentially; mutually exclusive with `<URL>` |
| `--model <KEY>` | — | force a specific ASR model key, skipping the language routing |
| `-o, --out <DIR>` | `transcripts` | the output directory |
| `-l, --language <LANG>` | `auto` | an ISO code or a language name, or `auto` to detect |
| `--output-format <FORMAT>` | `md` | `md`, `txt`, `srt`, `vtt`, `json`, `all`, `full`, `complete` |
| `--keep-audio` | off | keep the downloaded audio and the intermediate WAV under `<out>/.work/` |
| `--no-print` | off | do not echo the transcript to stdout |
| `--cookies <BROWSER_OR_FILE>` | — | a browser name (`firefox`, `chrome`, `edge`, `brave`, `vivaldi`) or a Netscape cookie file |
| `--no-proxy` | off | bypass the configured proxy for this run |
| `--rotate-tor` | off | request a fresh Tor exit circuit before downloading |
| `--no-playlist` | off | transcribe only the first video of a playlist URL |
| `--playlist-limit <N>` | — | transcribe only the first `N` videos of a playlist URL |
| `--auto-download` | off | download a missing checkpoint without asking |
| `--summary` | off | write a Markdown summary via a local Ollama instance |
| `--summary-model <MODEL>` | `llama3.1:8b` | the Ollama model used by `--summary` |
| `--summary-prompt <FILE>` | built-in | a prompt template for `--summary` |
| `-V, --version` | — | print the version |

Exactly one of `<URL>` and `--batch` must be given. Giving both, or
neither, is a usage error:

```console
$ lc-rs --batch urls.txt https://example.com/v
lc-rs: error: provide a URL/file or --batch FILE, not both
```

### Examples

```bash
# Transcribe a YouTube video, detecting the language.
lc-rs https://www.youtube.com/watch?v=dQw4w9WgXcQ

# Russian audio with a forced language, all output formats.
lc-rs lecture.mp4 -l ru --output-format all -o ./out

# A batch of videos, resuming after an interruption.
lc-rs --batch urls.txt -o ./out

# Force a specific model.
lc-rs talk.mp3 --model qwen3-asr-0.6b

# Every format plus a local summary.
lc-rs talk.mp3 --output-format all --summary --summary-model llama3.1:8b
```

## `lc-rs doctor`

Diagnoses the local installation: the binary, the external tools, the
network path, the ONNX Runtime and the checkpoints.

| Flag | Purpose |
|---|---|
| `--fix` | install the missing system tools and download the missing checkpoints |

The output is one row per check, marked with ✅ or ❌:

```text
lc-rs 0.1.0

Tools:
  ✅ ffmpeg  (/usr/bin/ffmpeg)
  ✅ yt-dlp  (/usr/bin/yt-dlp)

Network:
  ✅ proxy  (socks5h://127.0.0.1:9050 (reachable))

Model runtimes:
  ✅ onnxruntime  (in-process (CPU))

Models:
  ✅ qwen3-asr-0.6b (~2.0 GB)  (Qwen3-ASR 0.6B — Russian (WER 5.13))
```

Exit code `0` when every check passed, `1` otherwise.

## `lc-rs model`

| Subcommand | Purpose |
|---|---|
| `list` | every supported model, its role, its size and its install status |
| `download <KEY>` | download one model; `--all` downloads every model, `--force` re-downloads, `-y` skips the confirmation |
| `rm <KEY>` | remove an installed model; the aliases `remove` and `delete` work too, `-y` skips the confirmation |
| `info <KEY>` | the details of one model: role, description, size, source, local path, missing artefacts |

```console
$ lc-rs model list
Models used by lc-rs:

  Key                             Role                Size   Status
  --------------------------------------------------------   -----------
  langid-ecapa                    language-id        84 MB   ✅ installed
  qwen3-asr-0.6b                  russian           2.0 GB   ✅ installed
  parakeet-tdt-0.6b-v3            english             670 MB   ✅ installed
  nemotron-3.5-asr-streaming-0.6b multilingual     1021 MB   ✅ installed
```

## `lc-rs search`

```bash
lc-rs search "climate policy"
```

Searches the indexed transcripts for a case-insensitive substring. The
results are ranked by the number of occurrences, and carry the timestamp of
the matched segment when the transcript has timed segments:

```text
dQw4w9WgXcQ  00:01:23  and that is the climate policy we discussed
        The talk — Chapter 3
```

Exit code `1` when nothing matched.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | the run failed — a missing dependency, a failed download, a failed transcription |
| `2` | a usage error — an unknown flag, an invalid value, both `<URL>` and `--batch` |
| `130` | the run was interrupted (Ctrl-C) |
