# 03. Error model

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Documents the failure taxonomy of the crate and how the failures reach the user.
- **Feeds into:** [api/01-command-line-interface.md](01-command-line-interface.md), [design/01-pipeline-stages.md](../design/01-pipeline-stages.md)

## The error type

Every failure in `lc-rs` is a variant of `errors::Error`
(`src/errors.rs`). The variants mirror the exception classes of the
reference implementation, so a caller can match on a specific failure
instead of scraping a message.

| Variant | Raised when |
|---|---|
| `Dependency` | a required external tool or checkpoint is missing |
| `Download` | `yt-dlp` or an HTTP fetch failed |
| `AudioConversion` | `ffmpeg` failed, or the WAV has the wrong shape |
| `Transcription` | the ASR or the classifier failed to produce a transcript |
| `Install` | an automated install step failed |
| `Model` | the model registry rejected a key, or a checkpoint is inconsistent |
| `Playlist` | the playlist expansion failed |
| `Usage` | a local capability check failed |
| `Io` | a filesystem operation failed; carries the path and the source error |
| `Json` | a JSON (de)serialisation failed |
| `Onnx` | an ONNX Runtime call failed |
| `Tokenizer` | the tokenizer could not be loaded or used |
| `Interrupted` | the user interrupted the run |

## The rendering

`Error` derives `thiserror::Error`, so the `Display` implementation is the
user-facing message. The CLI prints it as:

```console
$ lc-rs talk.mp3
lc-rs: dependency error: Missing or incomplete model checkpoints:
  - qwen3-asr-0.6b: missing encoder.onnx

Download them with:  lc-rs model download qwen3-asr-0.6b
Or download everything:  lc-rs model download --all
```

The `Io` variant is special: it renders the path *and* the underlying
error, so the user knows which file failed.

## The exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | a runtime failure — any `Error` variant that is not a usage error |
| `2` | a usage error — clap rejected the arguments, or the CLI detected an invalid combination |
| `130` | `Error::Interrupted` |

## Warnings versus errors

Some failures are deliberately *not* fatal, because the work they affect is
an enrichment of an artefact that has already been written:

| Failure | Behaviour |
|---|---|
| the search index cannot be updated | a warning; the run succeeds |
| the chapter artefacts cannot be written | a warning; the transcript is kept |
| the summary fails (Ollama down, timeout, unknown model) | a warning; the run succeeds |
| the language detection is not confident | a warning; the multilingual model is used |
| a source of a batch fails | recorded as `failed`; the remaining sources are processed |

The rule: **an error means the requested artefact was not produced; a
warning means it was produced but something around it was not.**

## The library API

`lc-rs` is also a library (`lc_rs`), and the error type is part of its
public surface:

```rust
use lc_rs::errors::{Error, Result};

fn transcribe(path: &std::path::Path) -> Result<String> {
    let spec = lc_rs::models::get_model("qwen3-asr-0.6b")?;
    // … the error type carries the path, the source error and the context.
    Ok(String::new())
}
```

Callers that need to distinguish the failures match on the variant:

```rust
match lc_rs::models::get_model(key) {
    Ok(spec) => { /* … */ }
    Err(Error::Model(message)) => eprintln!("unknown model: {message}"),
    Err(other) => return Err(other),
}
```
