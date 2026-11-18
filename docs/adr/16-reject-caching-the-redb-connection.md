# 16. Reject caching the redb connection

- **Status:** Rejected
- **Date:** 2026-11-19
- **Deciders:** @Githab-capibara
- **Related:** [adr/06-store-the-transcript-search-index-in-redb.md](06-store-the-transcript-search-index-in-redb.md), [adr/13-load-every-model-once-per-process.md](13-load-every-model-once-per-process.md), [design/08-chapters-and-search.md](../design/08-chapters-and-search.md)

## Context

[adr/13](13-load-every-model-once-per-process.md) caches the ONNX Runtime
sessions, because loading a checkpoint costs seconds and gigabytes. The
obvious follow-up proposal was to treat the search index the same way:
hold one `redb::Database` in a `OnceLock` for the lifetime of the process, so
that a batch of N files writes N transcripts through one open database
instead of opening and closing the file N times.

The proposal looked symmetric with the model cache, and it is not. The two
backing stores have opposite concurrency models:

| | ONNX Runtime session | `redb::Database` |
|---|---|---|
| Sharable within a process | yes, one `Mutex`, one inference at a time | **no** — a writer holds an exclusive lock |
| Sharable across processes | irrelevant (in-process only) | **no** — the next process gets `DatabaseAlreadyOpen` |
| Cost of re-opening | seconds and gigabytes | a file open plus a lock acquisition |

The cache would have moved a cost measured in microseconds onto a lock
measured in availability.

## Decision

**Rejected.** `src/search.rs` keeps opening the database per call, and no
redb connection is cached.

Two probes settled it, both run against the crate's own `search::open`:

1. **Two live handles in one process.** Holding the first `Database` and
   opening a second one on the same file returns
   `Err(Dependency("could not open the search index at …/index.redb:
   Database already open. Cannot acquire lock."))`. A `OnceLock` would create
   exactly that situation for every later caller that did not go through the
   cache.
2. **A second process.** Holding the database open in one `lc-rs` process and
   opening the same file from another produces the identical error. So the
   cache would not merely be redundant: it would make the index unreadable
   for any other `lc-rs` invocation — `lc-rs search` in another terminal, a
   batch running next to an interactive run — for as long as the process
   lived.

The per-call open is also what makes the index usable at all: `upsert`
returns its `Database` to the OS, so the next process can write.

## Consequences

- **Easier:** the search index stays available to any number of processes.
  `lc-rs transcribe` in one terminal and `lc-rs search` in another keep
  working, which is the behaviour users already have.
- **Easier:** no lifetime question about who owns the index, and no order in
  which a process must exit for the lock to be released.
- **Harder:** a batch pays one file open and one lock acquisition per file
  instead of one per run. That cost is microseconds and it is not on the hot
  path — the transcription is seconds to minutes.
- **Given up:** the symmetry with the model cache. Sessions are expensive to
  create and cheap to share; a redb writer is the opposite, and treating them
  alike was the mistake.
- **Migration:** none, because nothing changed.

## Alternatives considered

- **Cache the connection, fall back to per-call open when the lock is taken.**
  Rejected: it would make the first file of every batch fast and every later
  one contended, and it would reintroduce the cross-process failure as a
  *sometimes* rather than a *never* — the worst shape for a bug like that.
- **Open the database once per batch and pass it down.** Rejected: the
  pipeline's stages are independent functions that can each fail on their
  own, and threading a live writer through them buys microseconds while
  adding a lifetime to every signature.
- **Use redb's multi-writer mode.** Rejected: the crate writes one transcript
  at a time, so there is no concurrency to gain, and multi-writer mode has a
  different durability cost.
- **Cache the parsed transcript entries in memory instead.** Not needed: the
  index is read by `lc-rs search`, which is a separate short-lived process,
  and the read path is already a single file open.