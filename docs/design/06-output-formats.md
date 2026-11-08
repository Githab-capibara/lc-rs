# 06. Output formats and chapter markers

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Establishes how a transcript is serialised, and how chapter markers are folded in.
- **Feeds into:** [design/01-pipeline-stages.md](01-pipeline-stages.md), [api/01-command-line-interface.md](../api/01-command-line-interface.md)

## Flow

`--output-format` selects which artefacts are written. The value is
resolved by `pipeline::output_formats`:

| Value | Artefacts |
|---|---|
| `md` | `<stem>.md` — the default |
| `txt` | `<stem>.txt` |
| `srt` | `<stem>.srt` |
| `vtt` | `<stem>.vtt` |
| `json` | `<stem>.json` |
| `all`, `full`, `complete` | all five |

`src/formatter.rs` owns the serialisation of each format, and mirrors
`src/localcaption/formats.py` of the reference implementation byte for
byte, including the clock formats and the trailing-newline rules.

## The formats

**Markdown** (`write_md`) — the transcript, a blank line, then the
timestamps section:

```markdown
Питаю, що твоя країна може зробити для тебе.

## Timestamps

- `00:00:00,000` — Питаю, що твоя країна може зробити для тебе.
```

**Plain text** (`write_txt`) — the transcript with the trailing whitespace
stripped and exactly one trailing newline.

**SRT** (`write_srt`) — one block per segment, `HH:MM:SS,mmm` clocks:

```text
1
00:00:00,000 --> 00:00:06,000
Питаю, що твоя країна може зробити для тебе.
```

**WebVTT** (`write_vtt`) — the same blocks with a `WEBVTT` header and
`HH:MM:SS.mmm` clocks.

**JSON** (`write_json`) — the structured document:

```json
{
  "model": "nemotron-3.5-asr-streaming-0.6b",
  "role": "",
  "language": "uk",
  "language_name": "Ukrainian",
  "text": "Питаю, що твоя країна може зробити для тебе.",
  "segments": [
    { "start": 0.0, "end": 6.0, "text": "Питаю, що твоя країна може зробити для тебе." }
  ]
}
```

The file name is built by `formatter::output_file`, which appends the suffix
to the *stem* rather than replacing an existing extension — a media file
called `lecture.2024` yields `lecture.2024.md`, not `lecture.md`.

## Chapter markers

The chapter markers come from the yt-dlp metadata. `src/chapters.rs` holds
them and slices the transcript by time.

- In the default single-file mode (`--output-format md`) the chaptered
  transcript is folded **into** the Markdown file, so exactly one artefact
  remains.
- In the multi-format mode the chapters are written separately, as
  `<stem>.chapters.json` and `<stem>.chaptered.md`.

## Constants

| Constant | Value | Owner |
|---|---|---|
| Supported formats | `md`, `txt`, `srt`, `vtt`, `json` | `pipeline::SUPPORTED_OUTPUT_FORMATS` |
| SRT clock separator | `,` | `formatter::clock` |
| VTT clock separator | `.` | `formatter::clock` |

## Failure modes

- **A segment has a negative timestamp.** The clock is clamped to zero.
- **The chapter list is empty.** No chapter artefacts are written; the
  Markdown file stays as it is.
- **The chapter artefacts cannot be written.** A warning is logged and the
  transcript itself is kept — the chapters are an enrichment, not a
  requirement.
- **The summary needs a `.txt` artefact.** In the single-file mode the
  Markdown file is used; in the multi-format mode the `.txt` file is used.
  If neither exists, the summary is skipped with a warning.
