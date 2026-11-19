# 17. Place the inference on the measured VRAM budget, CPU first as the fallback

- **Status:** Accepted
- **Date:** 2026-11-19
- **Deciders:** @Githab-capibara
- **Related:** [adr/01-run-models-in-process-through-onnx-runtime.md](01-run-models-in-process-through-onnx-runtime.md), [adr/08-compile-cuda-support-as-an-opt-in-feature.md](08-compile-cuda-support-as-an-opt-in-feature.md), [adr/13-load-every-model-once-per-process.md](13-load-every-model-once-per-process.md), [design/01-pipeline-stages.md](../design/01-pipeline-stages.md)

## Context

[adr/01](01-run-models-in-process-through-onnx-runtime.md) put every model on
the CPU through ONNX Runtime, and
[adr/08](08-compile-cuda-support-as-an-opt-in-feature.md) kept CUDA as an
opt-in cargo feature. That left the tool with a yes/no split the workstation
it runs on does not have: the card is shared with whatever else the user is
doing, so "the GPU" is not one state but three.

The machine this decision was made on states the problem exactly. A Quadro
M4000M with 4 GiB of VRAM, of which a local `llama-server` holds most. At the
time of writing it reported 96–1319 MiB free, and the number moves every
minute. Neither "always GPU" nor "never GPU" is right for it:

| Card state | What the run should do |
|---|---|
| Room for the whole checkpoint | run the whole model on the GPU |
| Room for part of it | use the space that exists, the rest on the CPU |
| No room left at all | stay on the CPU |
| Not measurable | try the GPU; fall back if it does not work |

Two facts about the crate bound the solution before it could be designed.
`#![forbid(unsafe_code)]` rules out talking to the NVIDIA driver from Rust
(a CUDA binding is unsafe FFI), and the standing rule against new
dependencies rules out pulling in an NVML wrapper. So the memory picture has
to come from a child process, not from a library — `nvidia-smi` is already
on every machine with an NVIDIA driver, it reports VRAM in MiB, and it is
the same tool the user has open anyway.

## Decision

**The placement is a per-model decision, computed from a measured VRAM
budget, and the CPU is always reachable.** `src/gpu.rs` holds the policy,
`src/ml/mod.rs` applies it to every session it creates.

1. `gpu::probe_memory()` reads `nvidia-smi
   --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits`
   under a two-second timeout and parses the first line. A missing tool, a
   timeout and an unparseable answer all mean "unmeasured" rather than
   "no GPU", because those are different failures.
2. `gpu::Decision::required_mib(model_bytes)` budgets a checkpoint at its
   weights × 1.5 + 256 MiB — the multiplier covers activations, the cuDNN
   algorithm-search workspace and arena rounding for a 45 s window. The
   budget is the whole checkpoint directory, not the one graph file being
   loaded: Qwen keeps 3 GB of decoder weights in `decoder_weights.data`
   next to a 285 KB `decoder_init.onnx`, and Parakeet is three graphs out
   of one directory that are resident together.
3. `gpu::Decision::for_budget(free, bytes)` applies the table above. A free
   VRAM below `MIN_USEFUL_FREE_MIB` (256 MiB) counts as "the card is busy":
   that is less than one cuDNN workspace, and a placement that cannot warm
   up is worse than no placement.
4. The decision is applied at session creation in
   `attach_execution_providers`, per session, so the 640 MB int8 encoder and
   the 3.8 GB fp32 decoder are each judged on their own budget. A placement
   that fits gets `ep::CUDA` with an unbounded arena; a placement that only
   partly fits gets the same provider with `gpu_mem_limit` set to the free
   VRAM and `ArenaExtendStrategy::SameAsRequested` (rounding the cap up to
   the next power of two would overshoot the card).
5. Two safety nets, because a budget is a bet rather than a measurement of
   the run. The provider is registered with `error_on_failure()`, so a
   missing CUDA runtime produces an error we recover from instead of ORT's
   silent downgrade to CPU; and a session that then fails to *commit* on the
   GPU is rebuilt without any provider and committed again. The retry is
   what turns "the arena was capped one MiB too aggressively" into a slower
   run instead of a dead one.

The `cuda` cargo feature also enables `ort/copy-dylibs`, which symlinks the
provider libraries next to the built binary. ONNX Runtime resolves them
relative to the executable, and without that a CUDA-enabled build fails
with a library path that says nothing about the actual problem.

`LCRS_CUDA=0` forces the CPU for one run. `LCRS_GPU_FREE_MIB=<n>` forces the
budget, which makes all four rows of the table replayable without a second
GPU — it is how the policy was verified.

One gate sits in front of the table: `gpu::COMPILED` is a constant, `true`
only in a build with the `cuda` feature. Without it the CUDA provider cannot
be registered, so `plan` returns the CPU placement *without probing the
card* — a probe whose result could not be acted on would be a child process
spawned for nothing, and a report claiming a GPU placement on a binary that
has no provider is a lie.

## Consequences

- **Easier:** a card with room is used, a card without room is left alone,
  and nothing has to be configured for either: the decision is a
  measurement. `LCRS_CUDA` exists for the rare case where the user knows
  better.
- **Easier:** the answer to "why was my run on the CPU" is now a line of
  output. `lc-rs doctor` prints the card's memory picture, whether the CUDA
  provider registers and, for every installed checkpoint, the placement and
  the budget it was decided on.
- **Harder:** the crate now runs a child process per session creation
  (`nvidia-smi`, ~100–200 ms, bounded by the timeout) on the model load
  path. That is seconds of work, and the model cache means it happens once
  per checkpoint per process, not once per file.
- **Harder:** `--features cuda` now downloads the CUDA provider distribution
  instead of the CPU one — 229 MB on disk against 107 MB here — so a build
  for GPU use is a bigger download. The default build does not pay for it.
- **Given up:** moving the parts of the pipeline that are *not* the models
  onto the GPU. They cannot go there from this crate:
  - the DSP front-end (FFT, mel filterbank, dither, resampling) would need
    its own CUDA or Vulkan kernels, which means unsafe driver bindings or a
    new dependency — both forbidden;
  - folding the front-end into the ONNX graph so ONNX Runtime would run it
    on the GPU was considered and rejected: it changes the numerics that
    [adr/14](14-fuse-the-mel-front-end-into-the-fft-loop.md) pins down to the
    bit, and the front-end is already 13–17 ms per 30 s window — 0.2 % of a
    run;
  - `ffmpeg` has no GPU path for what we ask of it (we discard the video
    and decode audio, and NVIDIA video decode does not apply);
  - `yt-dlp` is I/O-bound on the network.
  What is left is the models themselves, and they now go where there is
  room.
- **Migration:** none. A CPU-only build behaves as before, and a CUDA build
  that cannot register degrades to the CPU path with a warning instead of
  failing.

## Verification

On the machine this was decided on, every branch of the table was exercised
end to end, and all six runs produced a byte-identical `.srt`
(`c95714cc93c11b31fc50af2bdb056e06`):

| Budget | Branch taken | Outcome |
|---|---|---|
| live (96 MiB free, 1216 MiB needed) | card busy → CPU | identical |
| `LCRS_GPU_FREE_MIB=1116` (1216 needed) | capped arena | identical |
| `LCRS_GPU_FREE_MIB=9000` | unbounded arena | identical |
| `LCRS_GPU_FREE_MIB=100` | card busy → CPU | identical |
| `LCRS_CUDA=0` on the `cuda` build | CPU, GPU disabled | identical |
| no `cuda` feature | CPU | identical |

The GPU placements on this machine land on the fallback: the ORT build we
ship is the CUDA 13 distribution, and this machine has neither the CUDA 13
runtime (`libcublasLt.so.13` is absent — the error the run reports is
`Failed to load library … libcublasLt.so.13: cannot open shared object
file`) nor a card in the architectures it targets: the Quadro M4000M is
Maxwell, and the CUDA 13 line dropped Maxwell support. Both facts are about
this box, so neither was measurable by a test here. The run reports exactly
that and continues on the CPU, which is the behaviour the record describes:
the placement code is what runs on every branch; the GPU execution is what
this particular box cannot provide.

Output identity was re-checked where it is checked: 152 unit tests, the
8 golden ASR transcripts and the mel parity test all pass with
`--features cuda`, i.e. through the new placement path, on both the MSRV
(1.98) and the current toolchain.

## Alternatives considered

- **Always register the CUDA provider and let ONNX Runtime fall back.**
  Rejected: that is what the code did before, and the fallback is silent.
  A user with a CUDA build and a busy card had no way to tell whether the
  run was on the GPU except by watching `nvidia-smi`.
- **Opt in through `LCRS_CUDA=1`, as before.** Rejected: the point of the
  policy is that a workstation with a mostly-free card should use it
  without being asked. The variable becomes the override, not the switch.
- **Bind libcuda through `unsafe` FFI and read VRAM without a child
  process.** Rejected: `#![forbid(unsafe_code)]` is a crate-level rule with
  a CI job guarding it, and `nvidia-smi` gives the same number.
- **Reserve VRAM per model and refuse to run when the total does not fit.**
  Rejected: the checkpoint cache means the models already share one process
  lifetime, and a hard refusal turns "this one file would be tight" into
  "this transcript does not exist". Falling back per session is the lesser
  harm, and the arena cap is what makes the partial case honest.