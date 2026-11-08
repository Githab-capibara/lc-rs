# 01. Pipeline stages

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Establishes the end-to-end flow of a transcription run and the files it produces.
- **Feeds into:** [api/01-command-line-interface.md](../api/01-command-line-interface.md), [design/08-chapters-and-search.md](08-chapters-and-search.md)

## Flow

A run is five stages. Each one owns a module, and each one can fail
independently.

1. **Acquire the media.** `src/downloader.rs` spawns `yt-dlp` with an
   absolute path and a hard timeout, asking for the best audio-only stream.
   The metadata (`id`, `title`, `duration`, `chapters`, `webpage_url`) comes
   back as JSON on stdout and is parsed into a `serde_json::Value`. A local
   file skips this stage entirely and gets a synthetic metadata document
   built from its path.
2. **Normalise the audio.** `src/audio.rs` spawns `ffmpeg` to re-encode the
   media into a 16 kHz mono PCM WAV under `<out>/.work/`. The sample rate
   and the channel count are enforced regardless of the source, because
   every model expects exactly that format.
3. **Identify the language.** `src/ml/langid.rs` runs the embedded
   ECAPA-TDNN graph over 6-second windows and tallies the votes. The
   outcome decides which ASR model runs
   (`src/pipeline.rs::resolve_model`).
4. **Transcribe.** `src/ml/asr.rs` splits the waveform into 45-second
   windows and feeds them to the selected backend, collecting the timed
   segments. The three backends are described in
   [design/03-qwen3-asr-backend.md](03-qwen3-asr-backend.md),
   [design/04-parakeet-tdt-backend.md](04-parakeet-tdt-backend.md) and
   [design/05-nemotron-streaming-backend.md](05-nemotron-streaming-backend.md).
5. **Write the artefacts.** `src/formatter.rs` serialises the transcript into
   the requested formats, `src/chapters.rs` folds the chapter markers in,
   and `src/search.rs` adds the transcript to the search index. When
   `--summary` is given, `src/summary.rs` posts the transcript to a local
   Ollama instance.

## Constants

| Constant | Value | Owner |
|---|---|---|
| Target sample rate | 16 000 Hz | `audio::TARGET_SAMPLE_RATE` |
| Target channels | 1 | `audio::TARGET_CHANNELS` |
| ASR window | 45 s | `ml::asr::NVIDIA_WINDOW_SECONDS`, `QWEN_WINDOW_SECONDS` |
| Language-ID window | 6 s | `ml::langid::LANGID_WINDOW_SECONDS` |
| Language-ID windows (max) | 16 | `ml::langid::LANGID_MAX_WINDOWS` |
| Language-ID confidence | 0.6 | `ml::langid::LANGID_CONFIDENCE_THRESHOLD` |
| Language-ID vote share | 0.5 | `ml::langid::LANGID_VOTE_THRESHOLD` |
| Work directory | `<out>/.work/` | `pipeline::WORK_DIR_NAME` |
| Default output directory | `./transcripts` | `TranscribeArgs::out` |

## Output shape

A run writes into the output directory, named after the media stem:

| File | Written when | Content |
|---|---|---|
| `<stem>.md` | `--output-format md` (the default) | the transcript, then a `## Timestamps` section with one bullet per segment |
| `<stem>.txt` | `--output-format txt` | the plain transcript, one trailing newline |
| `<stem>.srt` | `--output-format srt` | `SubRip` subtitles, `HH:MM:SS,mmm` clocks |
| `<stem>.vtt` | `--output-format vtt` | `WebVTT` subtitles, `HH:MM:SS.mmm` clocks |
| `<stem>.json` | `--output-format json` | `{model, role, language, language_name, text, segments}` |
| `<stem>.chapters.json` | a source with chapters, format ≠ `md` | the chapter list |
| `<stem>.chaptered.md` | a source with chapters, format ≠ `md` | the transcript sliced by chapter |
| `<stem>.summary.md` | `--summary` and a reachable Ollama | the Markdown summary |
| `.work/` | `--keep-audio` | the downloaded media and the normalised WAV |

The Markdown transcript looks like this:

```markdown
Питаю, що твоя країна може зробити для тебе.

## Timestamps

- `00:00:00,000` — Питаю, що твоя країна може зробити для тебе.
```

When only the Markdown file was written, it is also the file that the
tool prints at the end of the run and the file that `--summary` reads. This
mirrors the reference implementation, where `txt` points at the Markdown
artefact in the single-file mode.

## Failure modes

| Failure | Signalled as | What the user sees |
|---|---|---|
| `yt-dlp` missing | `Error::Dependency` | `lc-rs: dependency error: …` |
| `yt-dlp` failed | `Error::Download` | the yt-dlp stderr, verbatim |
| `ffmpeg` failed | `Error::AudioConversion` | the ffmpeg stderr, verbatim |
| A checkpoint is missing | `Error::Dependency` | the list of missing artefacts and the `lc-rs model download` command |
| The classifier or the ASR failed | `Error::Transcription` / `Error::Onnx` | the ONNX Runtime message |
| The index could not be written | a warning | `could not update search index: …`; the run still succeeds |
| Ollama is unreachable | a warning | the run succeeds without a summary |

The intermediates in `.work/` are removed after the run unless
`--keep-audio` was given — including when the run failed, so a partial
download does not fill the disk.
