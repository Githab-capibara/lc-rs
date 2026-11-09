# Models

The four checkpoints `lc-rs` uses, and the language identifier that decides
which one of them runs — plus the optional quantised variant of the Russian
model.

| Guide | Purpose |
|---|---|
| [01-model-registry.md](01-model-registry.md) | The checkpoints, their artefacts, their provenance and the downloader |
| [02-language-identification.md](02-language-identification.md) | How the spoken language is detected and how it feeds the routing |

The routing decision itself is documented in
[../design/02-model-selection.md](../design/02-model-selection.md), and the
rationale in [../adr/04-route-asr-by-the-detected-language.md](../adr/04-route-asr-by-the-detected-language.md).
