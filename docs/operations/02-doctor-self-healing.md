# 02. `doctor` and self-healing

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Documents what `lc-rs doctor` checks, and what `--fix` repairs.
- **Feeds into:** [operations/01-installation.md](01-installation.md), [api/01-command-line-interface.md](../api/01-command-line-interface.md)

## The checks

`lc-rs doctor` runs five groups of checks and prints one row per check.

### The binary

The version of the running binary. This check cannot fail, but it anchors
the output.

### The tools

`ffmpeg` and `yt-dlp` are located on `PATH` and their absolute path is
printed. `git` and `curl` are checked as well, because the package manager
and the downloader use them.

```text
Tools:
  ✅ ffmpeg  (/usr/bin/ffmpeg)
  ✅ yt-dlp  (/usr/bin/yt-dlp)
```

### The downloader

`yt-dlp` is invoked once with `--version` to make sure it actually runs.

### The network

The configured proxy is probed with a three-second timeout. The row reports
the proxy URL and whether it answered.

```text
Network:
  ✅ proxy  (socks5h://127.0.0.1:9050 (reachable))
```

### The bot-check mitigations

The cookie source and the Tor circuit rotation are checked:

```text
YouTube / Tor bot-check mitigations:
  ✅ cookies  (browser: chrome)
  ✅ tor-rotate  (ok (rotated the current circuit))
```

### The model runtimes

There is no separate runtime to build — ONNX Runtime is linked into the
binary — so the check reports the runtime instead of a virtualenv:

```text
Model runtimes:
  ✅ onnxruntime  (in-process (CPU))
```

A CUDA-enabled build reports `(CUDA)` here.

### The models

Every registered checkpoint is checked for completeness, and the row
reports the size and the description.

```text
Models:
  ✅ qwen3-asr-0.6b (~2.0 GB)  (Qwen3-ASR 0.6B — Russian (WER 5.13))
```

## `--fix`

`lc-rs doctor --fix` repairs what it can:

| Gap | Repair |
|---|---|
| `ffmpeg` is missing | installed through the host's package manager |
| `yt-dlp` is missing | installed through the host's package manager |
| a checkpoint is missing | downloaded through the project's own downloader |

The package-manager invocation has a ten-minute timeout and needs root, so
it may fail on a locked-down machine — in which case the command says so
and suggests the manual command.

After the repairs, the diagnostics are **re-run**, so the output shows the
state after the fix, not before.

## The exit code

| Code | Meaning |
|---|---|
| `0` | every check passed |
| `1` | at least one check failed |

Without `--fix`, a failed run also prints a "How to fix" section with the
manual command for every gap.
