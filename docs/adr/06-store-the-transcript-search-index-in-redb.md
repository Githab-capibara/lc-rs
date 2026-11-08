# 06. Store the transcript search index in redb

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [design/08-chapters-and-search.md](../design/08-chapters-and-search.md), [api/01-command-line-interface.md](../api/01-command-line-interface.md)

## Context

`lc-rs search TERM` searches the transcripts produced by earlier runs. The
Python implementation kept a JSONL file — one JSON document per line — and
on every search it re-read the whole file and re-parsed every document,
including the timed segments, only to match a substring against them.

That is quadratic in the number of transcripts and in their length. With a
few hundred transcripts it is noticeable; with a few thousand it is the
dominant cost of the command. The index is also never used for anything but
search, so all it needs is a key-value store with a substring scan.

`redb` is an embedded, pure-Rust, ACID key-value store: a single file, no
server, no background process, no recovery daemon. It supports range scans
and secondary tables, which is what a search index needs.

## Decision

The transcript index is an embedded `redb` database, stored at
`$XDG_DATA_HOME/lc-rs/index.redb` by default:

- one table `transcripts` maps the transcript id to its JSON document
  (id, url, title, duration, language, model, chapters, transcript path);
- one table `segments` maps the same id to the JSON array of its timed
  segments;
- the index is written after a successful run, and a failure to update it is
  a *warning*, never a failed run — a locked or corrupt index must not lose
  a transcript that has already been written to disk;
- the ranking rules are unchanged from the Python implementation: a hit
  scores the number of occurrences of the needle in the matched text plus
  the number of occurrences in the title, and a chapter hit gets a `+2`
  bonus.

The location is overridable with `LCRS_INDEX_PATH` (or
`LOCALCAPTION_INDEX_PATH`).

## Consequences

- **Easier:** searches do not touch the filesystem and do not re-parse the
  transcripts. The cost of a search is proportional to the number of
  transcripts, not to their total size.
- **Easier:** the index is a single file that can be deleted to start over;
  there is no schema migration to run.
- **Easier:** the index survives a crash — the write is transactional.
- **Harder:** the index format is binary, so it cannot be inspected with
  `cat` or `grep` the way a JSONL file could. The Python implementation's
  index is not readable by `lc-rs` and vice versa.
- **Harder:** `redb` is a new dependency, and it locks the database file
  while it is open. Two `lc-rs` processes writing the index at the same
  time cannot both succeed; the second one logs a warning and carries on.
- **Given up:** the human-readable, append-only JSONL index. It was a
  deliberate simplicity trade in the Python implementation, and it cost
  performance.
- **Migration:** the old JSONL index is not read. The first `lc-rs` run
  rebuilds the index from the transcript it just wrote; older transcripts
  have to be re-transcribed (or indexed manually) to become searchable.

## Alternatives considered

- **Keep the JSONL index.** Rejected: it re-reads and re-parses everything
  on every search.
- **SQLite through `rusqlite`.** Rejected: `rusqlite` needs a C library or
  a bundled build of SQLite, which conflicts with the "minimal, portable
  dependency set" constraint. `redb` is pure Rust.
- **Index the transcripts on disk (a directory of small files).** Rejected:
  it is slower than a key-value store and does not give transactions.
- **Write a proper inverted index.** Rejected: the corpora are small, and
  the substring scan over the stored segments is fast enough.
