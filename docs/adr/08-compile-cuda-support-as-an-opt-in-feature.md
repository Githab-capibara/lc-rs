# 08. Compile CUDA support as an opt-in cargo feature

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [api/02-configuration.md](../api/02-configuration.md), [operations/01-installation.md](../operations/01-installation.md)

## Context

ONNX Runtime ships a CPU execution provider in every build and a CUDA
execution provider only in a build that was compiled against the CUDA
toolkit and cuDNN. The `ort` crate mirrors that split: the `cuda` feature
enables the CUDA execution provider, and a build without it simply cannot
use a GPU.

A GPU makes the difference between "usable" and "not usable" on a long
recording: the CPU path decodes a 45-second window of the 0.6B models in a
few seconds, which is comfortable, but a batch of forty videos spends most
of its time in inference.

The complication is that a CUDA-enabled ONNX Runtime needs a matching CUDA
and cuDNN at run time. On a machine where they are not installed, the
library fails to load — and it fails at *session creation*, deep inside the
run, long after the user has already downloaded the media.

## Decision

CUDA is an **opt-in cargo feature**, off by default:

```toml
[features]
default = []
cuda = ["ort/cuda"]
```

A CPU build has no reference to the CUDA provider at all and therefore no
way to fail on a missing toolkit. A CUDA build registers the provider
through `ort`'s `ExecutionProvider` list when the environment is created.

The runtime behaviour is controlled separately, by the environment:

- `LCRS_CUDA=0` (or `false`, `no`, `off`) forces CPU execution even in a
  CUDA-enabled build;
- `LCRS_CUDA=1` enables it in a CUDA-enabled build.

This split means a single binary can be built once and used on both a
laptop without a toolkit and a workstation with one — as long as the
toolkit's `libonnxruntime_providers_cuda.so` and the cuDNN libraries are on
`LD_LIBRARY_PATH` at run time.

## Consequences

- **Easier:** the default build has no CUDA dependency and cannot fail
  because of a missing toolkit. `cargo install lc-rs` works on a machine
  with no GPU whatsoever.
- **Easier:** the decision to use the GPU is a build-time choice, visible
  in `Cargo.toml` and in the `--features cuda` flag, instead of a runtime
  environment variable that silently changes performance.
- **Easier:** a CUDA build can still be used on CPU, so one artefact covers
  both cases.
- **Harder:** a user with a GPU has to build with `--features cuda` and
  provide the CUDA and cuDNN libraries at run time. The documentation and
  `lc-rs doctor` both state this.
- **Harder:** the CUDA path cannot be exercised in CI without a runner with
  a GPU, so it is compile-checked (`cargo clippy --all-features`) but not
  executed there.
- **Given up:** a single binary that uses the GPU automatically. The
  alternative — bundling the CUDA provider — would make the default
  installation fragile.
- **Migration:** none. The default behaviour is unchanged.

## Alternatives considered

- **Enable CUDA by default.** Rejected: the default build would then fail
  to load on every machine without the CUDA toolkit.
- **Detect the GPU at run time and fail gracefully.** Rejected as
  insufficient: the failure happens inside ONNX Runtime's loader, which
  cannot be recovered from in-process.
- **Ship two binaries.** Rejected: one feature-flagged binary is simpler to
  distribute and to document.
