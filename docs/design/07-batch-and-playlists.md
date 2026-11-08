# 07. Batch runs and playlists

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Establishes how several sources are transcribed in one invocation.
- **Feeds into:** [design/01-pipeline-stages.md](01-pipeline-stages.md), [api/01-command-line-interface.md](../api/01-command-line-interface.md)

## Flow

Two mechanisms transcribe more than one source.

**`--batch FILE`** reads a text file with one source per line. Empty lines
and lines starting with `#` are ignored (`src/batch.rs`). Every other line
is a URL or a local path.

**A playlist URL** is detected by its query parameters: a `list=` or
`playlist=` parameter means the URL points at a playlist rather than at a
video (`src/playlist.rs`). The playlist is expanded with a metadata-only
`yt-dlp --flat-playlist` round-trip — fast even over a Tor proxy — and every
entry becomes a source of a batch run.

Both paths converge on `batch::transcribe_urls`, which processes the sources
**sequentially**. A single ASR model already saturates the CPU, so a worker
pool would only add contention.

Each source is processed independently:

1. the output directory for the source is `<out>/<video-id>/`;
2. a source whose transcript already exists is *skipped*, so an interrupted
   batch can be resumed by re-running the same command;
3. a source that fails does not stop the batch — it is recorded as `failed`
   and the run continues.

## The summary table

At the end, the batch prints one row per source:

```text
── batch ──────────────────────────────────────────
 ✅ dQw4w9WgXcQ   03:31   00:12   talk.md
 ❌ abc123XYZ     01:02   00:04   download error: HTTP 429
 ⏭ def456UVW     02:10   skipped
```

The columns are the video id, the media duration, the elapsed time and the
outcome. The exit code is `1` when at least one source failed, `0`
otherwise.

## The playlist flags

| Flag | Effect |
|---|---|
| `--no-playlist` | do not expand a playlist URL; treat it as a single video |
| `--playlist-limit N` | transcribe only the first `N` entries of the expanded playlist |

Without either flag, a playlist URL is expanded in full and every entry is
transcribed — the reference behaviour.

## Constants

| Constant | Value | Owner |
|---|---|---|
| Playlist expansion timeout | 10 min | `playlist::EXPANSION_TIMEOUT` |

## Failure modes

- **The batch file does not exist.** `lc-rs: batch file not found: …`, exit
  code `1`.
- **The batch file is empty.** `lc-rs: no URLs in …`, exit code `1`.
- **A source fails.** The failure is recorded in the summary table and the
  remaining sources are still processed.
- **The playlist cannot be expanded.** `Error::Playlist`; the whole run
  stops, because there is nothing to transcribe.
- **`--playlist-limit 0`.** The limit is ignored — zero means "no limit",
  which is the reference behaviour.
