# 05. Keep the command-line surface one-to-one with the Python implementation

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [api/01-command-line-interface.md](../api/01-command-line-interface.md), [design/01-pipeline-stages.md](../design/01-pipeline-stages.md)

## Context

`lc-rs` replaces `localcaption`. The two tools do the same job — download
media, transcribe it locally, write the artefacts — and users of the Python
tool have scripts, muscle memory and documentation that refer to its
commands and flags.

A rewrite is the ideal moment to clean up a CLI: fewer flags, clearer names,
better defaults. It is also the moment where a rewrite silently breaks every
script that used the old tool. In this case the requirement is explicit: the
command surface must match one-to-one, so that `localcaption …` can be
replaced by `lc-rs …` without touching anything else.

The Python CLI has some quirks that a fresh design would not choose:

- the transcription command is the *default* — `localcaption URL` works, and
  `localcaption transcribe URL` works too;
- `--output-format` accepts `all`, `full` and `complete` as synonyms;
- `--model rm` has the aliases `remove` and `delete`;
- the transcript echo prints the *Markdown* artefact even in the default
  single-file mode, because the Python implementation points `txt` at the
  Markdown path when it is the only file that was written;
- `model list` prints the model key in a 30-column column, exactly as the
  reference implementation does.

## Decision

The command surface of `lc-rs` mirrors the Python CLI one-to-one:

```text
lc-rs <url-or-file> [options]    transcribe a video (default)
lc-rs --batch FILE [options]     transcribe a list of URLs/files
lc-rs doctor [--fix]             diagnose a local install
lc-rs model <subcommand>         list / download / remove models
lc-rs search <term>              search past transcripts
```

Every flag of `TranscribeArgs` in `src/cli.rs` has the same name, the same
short form, the same default and the same accepted values as its Python
counterpart. The synonyms (`all`/`full`/`complete`, `rm`/`remove`/`delete`)
are preserved. The quirks are preserved too — including the `txt`-points-at-
`md` behaviour, which is implemented in
`pipeline::write_output_artefacts` with a comment naming the Python
behaviour it reproduces.

The subcommands are dispatched by hand rather than through clap's subcommand
machinery, because the transcription command is the default: `lc-rs URL` and
`lc-rs transcribe URL` must behave identically, and clap's
`args_conflicts_with_subcommands` cannot express "the URL form *is* the
transcribe form".

## Consequences

- **Easier:** an existing `localcaption` workflow keeps working by changing
  one word in the command line.
- **Easier:** the CLI is documented once; the reference implementation is the
  specification, and the parity is testable.
- **Easier:** the awkward parts of the Python surface are explicit and
  documented rather than accidental.
- **Harder:** some flags are redundant (`--no-playlist` versus
  `--playlist-limit`, `--no-proxy` versus `LOCALCAPTION_PROXY=""`). The
  redundancy is the price of the compatibility.
- **Harder:** the "several flags instead of one" design reads as unusual to
  someone who has not seen the Python tool, and the help text has to explain
  each flag individually.
- **Given up:** the chance to simplify the flag set. The comment in
  `src/cli.rs` records that the flat flag set is deliberate.
- **Migration:** none. `lc-rs` is a drop-in replacement at the command line.

## Alternatives considered

- **Redesign the CLI while porting.** Rejected: it would break every
  existing invocation and every script, and it was explicitly out of scope.
- **Use clap's subcommand machinery with a `transcribe` subcommand only.**
  Rejected: it removes the default form, which is the most used form.
- **Keep only the flags that the Rust implementation can honour.**
  Rejected: `--summary` and `--rotate-tor` are honoured; there is no flag
  that has to be dropped.
