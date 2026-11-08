# 02. Configuration

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Documents every environment variable and the filesystem layout.
- **Feeds into:** [adr/07-route-every-outbound-request-through-the-proxy.md](../adr/07-route-every-outbound-request-through-the-proxy.md), [models/01-model-registry.md](../models/01-model-registry.md)

## Environment variables

Every variable of the reference implementation is honoured under both its
original `LOCALCAPTION_` prefix and the Rust-native `LCRS_` prefix. When
both are set, `LCRS_` wins.

| Variable | Default | Purpose |
|---|---|---|
| `LCRS_PROXY` / `LOCALCAPTION_PROXY` | `socks5h://127.0.0.1:9050` | the SOCKS5 proxy for every outbound request; an empty value disables the proxy |
| `LCRS_TOR_CONTROL` / `LOCALCAPTION_TOR_CONTROL` | — | the Tor control endpoint (`host:port`) used by `--rotate-tor` |
| `LCRS_TOR_DATADIR` / `LOCALCAPTION_TOR_DATADIR` | — | the directory holding Tor's control cookie |
| `LCRS_YTDLP_COOKIES` / `LOCALCAPTION_YTDLP_COOKIES` | — | a Netscape cookie jar for `yt-dlp` |
| `LCRS_YTDLP_COOKIE_BROWSERS` / `LOCALCAPTION_YTDLP_COOKIE_BROWSERS` | — | a comma-separated list of browsers to harvest cookies from |
| `LCRS_MODELS_DIR` / `LOCALCAPTION_MODELS_DIR` | `$XDG_DATA_HOME/lc-rs/models` | where the checkpoints live |
| `LCRS_INDEX_PATH` / `LOCALCAPTION_INDEX_PATH` | `$XDG_DATA_HOME/lc-rs/index.redb` | where the search index lives |
| `LCRS_SUMMARY_ENDPOINT` / `LOCALCAPTION_SUMMARY_ENDPOINT` | `http://localhost:11434/api/generate` | the Ollama endpoint used by `--summary` |
| `LCRS_SUMMARY_MODEL` / `LOCALCAPTION_SUMMARY_MODEL` | `llama3.1:8b` | the default Ollama model |
| `LCRS_THREADS` / `LOCALCAPTION_THREADS` | the number of logical CPUs | the ONNX Runtime intra-op thread count |
| `LCRS_CUDA` / `LOCALCAPTION_CUDA` | — | force CPU (`0`) or GPU (`1`) execution in a CUDA-enabled build |
| `XDG_DATA_HOME` | `~/.local/share` on Linux | the platform data root |
| `XDG_CONFIG_HOME` | `~/.config` on Linux | the platform config root |
| `XDG_CACHE_HOME` | `~/.cache` on Linux | the platform cache root |
| `RUST_LOG` | `info` | the `tracing` filter; `RUST_LOG=debug` shows the download progress, `RUST_LOG=warn` silences the progress lines |

The diagnostics go to **stderr**, so that stdout carries only the
artefacts — the transcript and the reports the user asked for:

```text
 INFO language: Ukrainian (uk, p=0.45)
 INFO done (Ukrainian, nemotron-3.5-asr-streaming-0.6b)
```

The proxy URL is validated before use. Only the `http`, `https`, `socks5`
and `socks5h` schemes are accepted.

## Cookie detection

The cookies for `yt-dlp` are resolved in this order:

1. `LCRS_YTDLP_COOKIES` — a Netscape cookie jar;
2. `~/.localcaption/yt-dlp-cookies.txt` — the default jar;
3. the browsers of `LCRS_YTDLP_COOKIE_BROWSERS`, or `chrome,firefox` when
   the variable is not set.

A browser is only accepted when its **cookie database** exists — `Cookies`
for the Chromium family, `cookies.sqlite` for Firefox — in the directory
`yt-dlp` itself searches for that browser. An empty profile directory is
not enough: `yt-dlp --cookies-from-browser` then aborts the download with
"could not find chrome cookies database".

The Chromium family is matched strictly, so a Chromium profile in
`~/.config/chromium` never makes the tool claim that *Chrome* cookies are
available. To use the Chromium profile, name it explicitly:

```sh
LCRS_YTDLP_COOKIE_BROWSERS=chromium lc-rs <url>
```

## Filesystem layout

```text
$XDG_DATA_HOME/lc-rs/
├── models/
│   ├── langid-ecapa/
│   │   └── langid_inline.onnx
│   ├── qwen3-asr-0.6b/
│   │   ├── config.json
│   │   ├── decoder_init.int4.onnx
│   │   ├── decoder_step.int4.onnx
│   │   ├── decoder_weights.int4.data
│   │   ├── embed_tokens.bin
│   │   ├── encoder.onnx
│   │   ├── preprocessor_config.json
│   │   └── tokenizer.json
│   ├── parakeet-tdt-0.6b-v3/
│   │   ├── config.json
│   │   ├── decoder_joint-model.int8.onnx
│   │   ├── encoder-model.int8.onnx
│   │   ├── nemo128.onnx
│   │   └── vocab.txt
│   └── nemotron-3.5-asr-streaming-0.6b/
│       ├── decoder.onnx
│       ├── decoder.onnx.data
│       ├── encoder.onnx
│       ├── encoder.onnx.data
│       ├── genai_config.json
│       ├── joint.onnx
│       ├── joint.onnx.data
│       └── vocab.txt
└── index.redb
```

On macOS the data root is `~/Library/Application Support`; on Linux it is
`~/.local/share`.

The output directory is chosen by the user (`-o`, default `./transcripts`)
and holds the transcripts plus the `.work/` directory of the intermediates.

## The proxy and the child processes

`yt-dlp` is spawned with the proxy exported into its environment, in both
the upper-case and the lower-case spelling, because different tools honour
different ones:

```text
ALL_PROXY=socks5h://127.0.0.1:9050
HTTP_PROXY=socks5h://127.0.0.1:9050
HTTPS_PROXY=socks5h://127.0.0.1:9050
all_proxy=…
http_proxy=…
https_proxy=…
NO_PROXY=localhost,127.0.0.1,::1
no_proxy=localhost,127.0.0.1,::1
```

The loopback hosts are always excluded, so a local Ollama instance and the
proxy itself stay reachable.

## Build-time configuration

| Cargo feature | Default | Effect |
|---|---|---|
| `cuda` | off | compiles the ONNX Runtime CUDA execution provider |

```bash
cargo build --release --features cuda
```

A CUDA-enabled build needs the CUDA and cuDNN shared libraries on
`LD_LIBRARY_PATH` at run time; without them the library fails to load.
`LCRS_CUDA=0` forces CPU execution in such a build.
