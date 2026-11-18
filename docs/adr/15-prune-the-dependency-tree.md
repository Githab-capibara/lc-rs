# 15. Prune the dependency tree and drop the oniguruma regex engine

- **Status:** Accepted
- **Date:** 2026-11-19
- **Deciders:** @Githab-capibara
- **Related:** [adr/01-run-models-in-process-through-onnx-runtime.md](01-run-models-in-process-through-onnx-runtime.md), [src/ml/features.rs](../../src/ml/features.rs), [src/models.rs](../../src/models.rs)

## Context

The dependency set this port started from was chosen for correctness, not for
size, and it carried four costs that had nothing to do with what the crate
actually does:

| Dependency | Why it was there | What it actually cost |
|---|---|---|
| `tokenizers` with the `onig` feature (0.21) | the BPE tokenizer of the Qwen checkpoint | links **Oniguruma**, a C library, and runs its regex engine inside the tokenizer's pre-tokenizer |
| `ndarray` (direct) | nothing in this crate — it came in as the `ort` `ndarray` feature's companion | a second array API next to `ort::value::Tensor`, and a transitive copy of the same code |
| `memchr` | nothing in this crate | a dependency maintained for a call that no longer exists |
| `futures-util` | the `StreamExt` loop of the checkpoint downloader | one trait import, for a loop `reqwest` can write itself |

Two of those were consequences of crate features rather than of direct use:
`ort` was built with its `ndarray` feature, and `reqwest` with its `stream`
feature. Both were enabled by default reflex; neither was needed — the crate
talks to ONNX Runtime through `Tensor`/`TensorRef` and streams a download
through `Response::chunk()`, which is reqwest's own API and needs no feature
flag at all.

`tokenizers` 0.21 is also the last line that still depends on Oniguruma: the
0.22+ line replaced the `onig` feature with `fancy-regex`, a pure-Rust,
Oniguruma-compatible engine, which removes both the C library and the
maintenance question of whether the pinned version is still built.

## Decision

The dependency set becomes the smallest one that still does the job:

- **`tokenizers` 0.21 → 0.23.2, `onig` → `fancy-regex`.** The tokenizer is
  the same BPE implementation over the same `tokenizer.json`; the regex engine
  behind the pre-tokenizer splits changes from Oniguruma to its pure-Rust
  port. The Cargo.lock contains no `onig` entry afterwards.
- **Drop the direct `ndarray`, `memchr` and `futures-util` dependencies**, and
  drop `ort`'s `ndarray` feature: the crate no longer builds an array type it
  never uses.
- **Drop `reqwest`'s `stream` feature.** The downloader in `src/models.rs`
  consumes `Response::chunk()` directly, with exactly the error mapping the
  old `StreamExt::next` loop had.
- **`tempfile` moves to `[dev-dependencies]`.** Every use is `#[cfg(test)]`,
  so the release binary stops carrying it.
- **`redb` 4.4 → 4.6.** A same-major refresh of the pure-Rust store; the API
  the index uses is unchanged.
- **`clap` stays at `"4.6"`.** It already resolves to 4.6.8; bumping the lock
  entry only would add diff noise without changing the build.

## Consequences

- **Easier:** the tokenizer no longer pulls a C library into the build or a
  C regex engine into the process. Oniguruma is gone from the lock file, and
  with it a second thing that can fail to build on a new platform.
- **Easier:** the dependency graph shrinks by four direct dependencies and
  their transitive tails; `cargo tree` is shorter and a `cargo audit` triage
  is shorter with it.
- **Easier:** the download path now depends on `reqwest` alone, so a future
  reqwest upgrade cannot break the loop through a feature it no longer uses.
- **Harder:** the tokenizer upgrade is a *behavioural* change dressed as a
  version bump, so it had to be verified rather than assumed: the prompt
  token ids of the Qwen backend and the transcript of the Ukrainian fixture
  were compared bit for bit before and after, and all eight golden tests in
  `tests/asr_golden.rs` still pass unchanged. They do — the anchors
  (`"user\n<|audio_start|>…"` → `[872, 198, 151669, 151676, 151670, 198,
  77091, 198]` and the two transcript anchors) reproduce exactly.
- **Harder:** `fancy-regex` is slower than Oniguruma on pathological patterns.
  It is not slower for this workload — the pre-tokenizer splits are short and
  linear, and the golden tests pin the result.
- **Given up:** the newest `tokenizers` is not adopted, only the newest line
  without Oniguruma. A future `tokenizers` that changes the BPE default would
  have to be re-verified the same way.
- **Given up:** a C-free build. That was the original motivation and it is
  **not** achieved: `reqwest`'s `rustls` feature is hard-wired to `aws-lc-rs`
  in reqwest's own manifest, which compiles C through `cc` and drives its
  build with `cmake`, and `ort`'s `download-binaries` feature makes `ort-sys`
  build-depend on `ureq`, which brings `ring`, also compiled with `cc`. What
  this decision removes is *our* regex-engine dependency; the C that remains
  belongs to the TLS provider and to ONNX Runtime's build-time downloader,
  and no configuration of those two crates avoids it today.
- **Migration:** `cargo update` semantics; no user-visible change and no
  on-disk format change.

## Alternatives considered

- **Keep `tokenizers` 0.21 with `onig`.** Rejected: it keeps a C library in
  the build and an upstream dependency that no longer moves.
- **Drop `download-binaries` from `ort`.** Rejected: it would remove `ring`
  from the graph, but it also removes the build-time fetch of the ONNX
  Runtime binaries, so every user would have to install a matching
  `libonnxruntime` themselves. A smaller build for a much larger install
  burden.
- **Select `ring` as the TLS provider instead of `aws-lc-rs`.** Rejected:
  reqwest's `rustls` feature enables `aws_lc_rs` unconditionally, so the
  choice is not available at this crate's layer — and `ring` compiles C too,
  so the saving would be `cmake`, not a C toolchain.
- **Vendor the tokenizer and drop `tokenizers` entirely.** Rejected: it would
  re-implement a BPE that is already correct, and the parity anchors exist
  precisely because the crate's contract is "the same tokens as the reference
  implementation".