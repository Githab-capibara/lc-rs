# 01. Installation

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Documents how to install `lc-rs` and its dependencies.
- **Feeds into:** [api/02-configuration.md](../api/02-configuration.md), [adr/08-compile-cuda-support-as-an-opt-in-feature.md](../adr/08-compile-cuda-support-as-an-opt-in-feature.md)

## Requirements

| Dependency | Why |
|---|---|
| Rust 1.98 or newer | the crate uses edition 2024 |
| `ffmpeg` | re-encodes the media into 16 kHz mono WAV |
| `yt-dlp` | downloads the media from a URL |
| a SOCKS5 proxy (usually Tor) | every outbound request goes through it |
| a C compiler and `cmake` | the TLS provider `reqwest` selects (`aws-lc-rs`) and ONNX Runtime's build-time downloader (`ring`, through `ort`) compile C while the crate builds |
| ~7.7 GB of disk | the four required checkpoints |

`ffmpeg` and `yt-dlp` are found on `PATH`; `lc-rs doctor --fix` can install
them through the host's package manager.

No Python, no `torch`, no virtualenv and no model server is needed: the
inference runs in-process through ONNX Runtime. The C toolchain entry above
is the only non-Rust build requirement, and it comes from the two crates
that have no pure-Rust option today — see
[adr/15-prune-the-dependency-tree.md](../adr/15-prune-the-dependency-tree.md).

## Building from source

```bash
git clone https://github.com/Githab-capibara/lc-rs
cd lc-rs
cargo build --release
```

The binary lands in `target/release/lc-rs`. Install it wherever you keep
your binaries:

```bash
install -Dm755 target/release/lc-rs ~/.local/bin/lc-rs
```

### With CUDA

```bash
cargo build --release --features cuda
```

The feature links the CUDA execution provider into ONNX Runtime and
symlinks the provider libraries next to the binary — ONNX Runtime finds
them relative to the executable, and without them a CUDA build would run on
the CPU with a library path that says nothing about the actual problem.

A CUDA build needs the CUDA 13 runtime on the machine at run time:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

The runtime is `libcudart`, `libcublas` and `libcurand` of the 13.x line.
Without them the provider refuses to register: the binary still starts, the
run prints one warning with the reason and stays on the CPU, and
`lc-rs doctor` reports the same thing under "GPU placement". Nothing has to
be configured for the GPU itself — each checkpoint is placed against the
card's free VRAM, measured with `nvidia-smi`, and the placement falls back
to the CPU on its own. See
[design/01-pipeline-stages.md](../design/01-pipeline-stages.md).

`LCRS_CUDA=0` forces the CPU for one run of a CUDA build, and
`LCRS_GPU_FREE_MIB=<MiB>` replaces the measurement, which is how the
placement policy is exercised without a second card.

## First run

```bash
lc-rs doctor --fix
```

The command checks the tools, the proxy, the ONNX Runtime and the
checkpoints, and it installs what is missing. Then:

```bash
lc-rs model download --all
```

Downloads are resumable and verified against the pinned SHA-256 digests, so
an interrupted download can simply be repeated.

## The models on their own

| Model | Size |
|---|---|
| `langid-ecapa` | 84 MB (embedded in the binary) |
| `qwen3-asr-0.6b` | 4 070 MB |
| `qwen3-asr-0.6b-int4` | 2 030 MB (optional) |
| `parakeet-tdt-0.6b-v3` | 2 550 MB |
| `parakeet-tdt-0.6b-v3-int8` | 670 MB (optional) |
| `nemotron-3.5-asr-streaming-0.6b` | 1 021 MB |

A run only needs the model it routes to — unless the language is `auto`, in
which case any of the three ASR models may be needed, so all of them
should be installed.

The optional `qwen3-asr-0.6b-int4` is the same model with a quantised
decoder: it downloads a third of the decoder weights and runs in half the
memory, at the cost of the fidelity. It is never fetched by `--all`; ask
for it by name when the machine cannot afford the 7.2 GiB the fp32
decoder needs:

```bash
lc-rs model download qwen3-asr-0.6b-int4
lc-rs talk.mp3 --model qwen3-asr-0.6b-int4
```

The optional `parakeet-tdt-0.6b-v3-int8` is the English model with quantised
encoder and joint graphs: an eighth of the encoder weights and 1.6 GiB
instead of 2.7 GiB of memory, at the cost of the fidelity — the word error
rate against the reference transcript rises from 0 to 0.469:

```bash
lc-rs model download parakeet-tdt-0.6b-v3-int8
lc-rs talk.mp3 --model parakeet-tdt-0.6b-v3-int8
```

## Tor

The default proxy is a local Tor instance:

```bash
sudo systemctl enable --now tor
```

`lc-rs doctor` reports whether the proxy is reachable. A different SOCKS5
endpoint is configured with `LCRS_PROXY`, and `--no-proxy` bypasses the
proxy for a single run.
