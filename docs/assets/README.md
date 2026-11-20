# Diagrams

- **Status:** Living document
- **Date:** 2026-11-20
- **Deciders:** @Githab-capibara
- **Purpose:** The hand-drawn SVG assets the README and this documentation embed.
- **Feeds into:** [../README.md](../README.md), [../design/01-pipeline-stages.md](../design/01-pipeline-stages.md), [../adr/17-place-inference-by-the-measured-vram-budget.md](../adr/17-place-inference-by-the-measured-vram-budget.md)

Every diagram is a static SVG: no scripts, no external references, one
palette (`#8250df` for the in-process stages, `#dea584` for inference,
`#1a7f37` for the front-end). They are committed so that a fork and a
mirror render the same picture as the repository.

| Asset | What it shows | Used by |
|---|---|---|
| [banner.svg](banner.svg) | the hero banner: one binary, five stages | the README's opening |
| [pipeline.svg](pipeline.svg) | the five stages end to end, and the three ASR backends the router picks between | the README, [design/01-pipeline-stages.md](../design/01-pipeline-stages.md) |
| [placement.svg](placement.svg) | the VRAM placement policy: four card states and what each one gets | the README, [design/01-pipeline-stages.md](../design/01-pipeline-stages.md), [ADR 17](../adr/17-place-inference-by-the-measured-vram-budget.md) |
| [benchmark.svg](benchmark.svg) | what 33.6 s of audio costs per model on this repository's machine, load against transcribe, plus the composition of one cold run | the README |

The numbers in `benchmark.svg` are measured, not modelled, and they are
dated by the machine they were taken on: an i7-6820HQ, 4 cores / 8 threads,
warmed page cache, one 45-second window, CPU-only. When a figure moves, the
diagram moves with it.