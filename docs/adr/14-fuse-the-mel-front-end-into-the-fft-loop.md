# 14. Fuse the mel front-end into the FFT loop

- **Status:** Accepted
- **Date:** 2026-11-19
- **Deciders:** @Githab-capibara
- **Related:** [adr/09-reproduce-the-reference-front-end-quirks.md](09-reproduce-the-reference-front-end-quirks.md), [design/03-qwen3-asr-backend.md](../design/03-qwen3-asr-backend.md), [src/ml/features.rs](../../src/ml/features.rs)

## Context

`log_mel_spectrogram` in `src/ml/features.rs` is the front-end shared by the
Qwen encoder (Whisper mel) and the Nemotron encoder (NeMo mel). It ran one
45-second window per ASR step, and it did the work in the most literal way
possible:

1. it built the dense filterbank matrix (`Vec<Vec<f32>>`, 128 × 201) and the
   Hann window on **every call**, although both depend only on the
   `MelConfig`;
2. it allocated a fresh `Vec<Complex32>` per frame and let `Fft::process`
   allocate its scratch buffer internally on every call;
3. it materialised the full power spectrogram as `spectra: Vec<Vec<f32>>`
   (3001 frames × 201 bins ≈ 2.4 MB for a 30 s window) before projecting;
4. it walked all 201 bins of every band, testing each weight for zero.

Measured on a 30-second Whisper window in release mode: **79.99 ms**, of
which the projection and the allocation churn were most of it. That is
small next to a multi-second encoder, but it is pure overhead — the same
numbers came out of the reference implementation, so nothing about the
arithmetic had to change. Only the data movement did.

## Decision

The front-end computes exactly the same values, in the same order, with the
same data structures it needs per call and no more:

- **The filterbank is stored sparsely.** `sparse_front_end` converts the
  dense matrix into `Vec<Vec<(bin, weight)>>` holding only the non-zero
  entries (about 6 of the 201 bins per band for a 128-band Slaney bank).
  Each band's entries are kept in ascending bin order, which is the order the
  dense loop accumulated in after its zero test, so the floating-point
  result is bit-for-bit the one the dense loop produced.
- **The mel projection is fused into the FFT loop.** Instead of building the
  whole `spectra` matrix and then projecting, each frame's 201-bin power
  array is projected immediately. The power array stays L1-resident, and the
  multi-megabyte spectrogram is never allocated.
- **The frame buffer, the FFT scratch and the power array are allocated once
  per call** and reused for every frame through
  `Fft::process_with_scratch`.
- **The tables are cached per process.** `cached_filters_and_window` keeps
  `(MelConfig, Arc<FrontEnd>)` pairs in a `OnceLock<Mutex<Vec<…>>>` capped at
  `MAX_CACHED_CONFIGS = 8` entries, keyed by an exact `MelConfig` comparison.
  The build happens under the lock, so two threads racing on an unseen config
  cannot each build their own copy.
- **The result is transposed, not re-derived.** The fused loop produces
  frame-major rows; the public `MelFeatures` layout is band-major
  (`n_mels × n_frames`), so a blocked transpose (64 frames at a time) moves
  the values into place.

## Consequences

- **Easier:** the Whisper front-end costs **13.1 – 17.0 ms** per 30-second
  window instead of 79.99 ms — **4.7× faster**, with no change to a single
  output value.
- **Easier:** the peak memory of a call drops by the size of the spectrogram
  matrix (~2.4 MB per 30 s window, more for a longer one), and the
  per-frame allocations disappear.
- **Easier:** the parameters of the front-end are the same `MelConfig` the
  parity tests already pinned, so the optimisation is testable rather than
  assumed: `mel_spectrogram_is_bit_identical_to_the_dense_implementation`
  compares the bit patterns of the new path against the pre-optimisation
  implementation kept in the test module as a golden reference.
- **Harder:** the code is less obviously a transcription of the reference
  Python. The dense loop is gone from the hot path and lives in the test as
  the reference instead — a reader has to know that `reference_spectrogram`
  is the specification.
- **Harder:** the cache is process-wide and unbounded in *time*, bounded in
  *size*. A caller that invents a new config on every call can push out older
  entries; a caller that uses the same configs (which is what the crate
  does) pays one linear scan of one or two entries per call.
- **Given up:** the `spectra` matrix as an inspectable intermediate. Nothing
  in the crate consumed it; the parity fixture compares the final features.
- **Migration:** none. The output is bit-identical, which the golden
  transcripts (all three backends) confirm.

## Alternatives considered

- **Parallelise the front-end with `rayon`.** Rejected: it would add a
  dependency and a thread pool to save 60 ms on a path that runs once per
  45-second window, and it would complicate the deterministic dither.
- **Drop `rustfft` for a hand-written radix-2 FFT.** Rejected: the fused loop
  already made the FFT a minority of the cost, and a hand-written transform
  would be a new source of numerical drift the parity tests would have to
  re-pin.
- **Precompute the dense matrix once and keep the dense loop.** Rejected: it
  keeps the spectrogram allocation and the 201-bin walk per band, which is
  where most of the remaining time went.
- **Cache the features per input.** Rejected: the inputs are distinct audio
  windows; only a repeated transcription of the same file would hit, and the
  call is already cheaper than the encoder that follows it.