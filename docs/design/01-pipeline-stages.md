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

## Reuse and concurrency

Two of those stages are expensive, and they are expensive in a way that a
batch multiplies:

- **The models are loaded once per process, not once per file.**
  `ml::cached_model` in `src/ml/mod.rs` keys a process-wide cache by the
  canonicalised checkpoint directory (or by a name, for the embedded language
  identifier) and hands out `Arc<Mutex<T>>`. The first caller loads; every
  later caller locks the same model for the duration of its inference. ONNX
  Runtime forbids concurrent inference on one session, and a batch is
  sequential, so the lock never contends — it exists because
  `Session::run` takes `&mut self`. See
  [adr/13-load-every-model-once-per-process.md](../adr/13-load-every-model-once-per-process.md).
- **The front-end tables are cached the same way.** `src/ml/features.rs`
  keeps the mel filterbank and the Hann window per `MelConfig`, because both
  depend only on the configuration and were rebuilt on every window. See
  [adr/14-fuse-the-mel-front-end-into-the-fft-loop.md](../adr/14-fuse-the-mel-front-end-into-the-fft-loop.md).

The two stages that do the work also run where they should: `run_pipeline`
moves language detection and the transcription into
`tokio::task::spawn_blocking`, because ONNX Runtime runs its graphs on its
own thread pool and never yields to an async executor. Running minutes of
inference directly on the runtime would stall the `ffmpeg` timeout future,
the progress output and Ctrl-C for the whole inference.

The search index is deliberately *not* cached: a redb writer is exclusive, so
a process-wide connection would lock every other `lc-rs` invocation out of
the index for as long as the process lived. `src/search.rs` opens the
database per call, at the cost of one file open per file — see
[adr/16-reject-caching-the-redb-connection.md](../adr/16-reject-caching-the-redb-connection.md).

## Where the inference runs

Stage 3 and stage 4 are the two that spend the time, and on a build made
with `--features cuda` each of them is placed per checkpoint by
`src/gpu.rs`. The card is not dedicated to this tool, so "the GPU" is not
one state but four, and the decision is a measurement rather than a flag:

| Free VRAM (measured with `nvidia-smi`) | Placement |
|---|---|
| at least what the checkpoint needs | the whole model on the GPU, arena unbounded |
| between 256 MiB and what it needs | the GPU with the arena capped at the free space; whatever does not fit runs on the CPU |
| below 256 MiB | the CPU — a placement that cannot warm up is worse than none |
| unmeasurable | the GPU is tried, and a session that cannot be created falls back to the CPU |

The budget of a checkpoint is every regular file of its directory — weights
are often external to the graph (Qwen keeps 3 GB in `decoder_weights.data`),
and the three Parakeet graphs of one directory are resident together —
multiplied by 1.5 with a flat 256 MiB on top, to cover activations, the
cuDNN algorithm-search workspace and arena rounding.

Two safety nets make a wrong budget survivable: the provider is registered
with `error_on_failure()`, so a missing CUDA runtime is an error we recover
from rather than ORT's silent downgrade, and a session that then fails to
*commit* on the GPU is rebuilt with no provider and committed again. The
retry is what turns "the arena was capped one MiB too aggressively" into a
slower run instead of a dead one.

A build **without** the `cuda` feature never registers the provider at all,
so it reports the CPU placement and does not even probe the card — the
measurement would be a child process whose result nothing could act on.
`LCRS_CUDA=0` forces the same for one run of a CUDA build, and
`LCRS_GPU_FREE_MIB=<n>` replaces the measurement, which is how the four rows
above are exercised on one machine.

Everything else in the pipeline stays where it was, and for reasons that are
not preferences: `ffmpeg` has no GPU path for discarding video and decoding
audio, `yt-dlp` is bound by the network, and the DSP front-end would need
its own CUDA or Vulkan kernels, which the crate cannot take (no `unsafe`, no
new dependency). The models are what moves. The full argument is
[adr/17-place-inference-by-the-measured-vram-budget.md](../adr/17-place-inference-by-the-measured-vram-budget.md).

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
| Card busy below | 256 MiB free | `gpu::MIN_USEFUL_FREE_MIB` |
| Checkpoint VRAM budget | weights × 3/2 + 256 MiB | `gpu::Decision::required_mib` |
| `nvidia-smi` probe timeout | 2 s | `gpu::PROBE_TIMEOUT` |

## Output shape

A run writes into the output directory, named after the media stem:

| File | Written when | Content |
|---|---|---|
| `<stem>.md` | `--output-format md` (the default) | the transcript, then a `## Timestamps` section with one bullet per segment |
| `<stem>.txt` | `--output-format txt` | the plain transcript, one trailing newline |
| `<stem>.srt` | `--output-format srt` | `SubRip` subtitles, `HH:MM:SS,mmm` clocks |
| `<stem>.vtt` | `--output-format vtt` | `WebVTT` subtitles, `HH:MM:SS.mmm` clocks |
| `<stem>.json` | `--output-format json` | `{model, role, language, language_name, text, segments}`; the language is the one the reference reports (`auto`, `en` or the code Qwen detected) |
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
| The GPU placement cannot run | a warning | `<model> could not run on the GPU (…); retrying on the CPU`, and the run succeeds |
| The index could not be written | a warning | `could not update search index: …`; the run still succeeds |
| Ollama is unreachable | a warning | the run succeeds without a summary |

The intermediates in `.work/` are removed after the run unless
`--keep-audio` was given — including when the run failed, so a partial
download does not fill the disk.
