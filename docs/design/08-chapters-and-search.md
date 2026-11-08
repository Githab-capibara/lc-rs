# 08. Chapters and the search index

- **Status:** Living document
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Researcher:** @Githab-capibara
- **Purpose:** Establishes how chapter markers are persisted and how past transcripts are searched.
- **Feeds into:** [adr/06-store-the-transcript-search-index-in-redb.md](../adr/06-store-the-transcript-search-index-in-redb.md), [design/06-output-formats.md](06-output-formats.md)

## Flow

### The index

After a successful run, `pipeline::build_index_entry` assembles the row
that describes the transcript:

| Field | Source |
|---|---|
| `id` | the yt-dlp video id, or the sanitised file stem for a local file |
| `url` | the `webpage_url` of the metadata, or the original source |
| `title` | the title of the metadata, or the id |
| `duration` | the duration of the metadata, when known |
| `language` | the routed or detected language code |
| `model` | the registry key of the model that produced the transcript |
| `chapters` | the chapter list, when the source has one |
| `transcript` | the path of the transcript file on disk |

The row and the timed segments are written into a `redb` database — see
[adr/06-store-the-transcript-search-index-in-redb.md](../adr/06-store-the-transcript-search-index-in-redb.md)
for the rationale. The write is transactional and happens *after* the
transcript artefacts are on disk, so a failed index update can never lose a
transcript.

### The search

`lc-rs search TERM` scans the index for the needle:

1. every transcript is matched against the needle, case-insensitively;
2. every timed segment is matched as well, so a hit carries a timestamp;
3. the chapter titles are matched, and a chapter hit gets a `+2` bonus.

The ranking rules are the ones of the reference implementation:

```text
score = occurrences of the needle in the matched text
      + occurrences of the needle in the title
      + 2 (when the hit came from a chapter title)
```

The output is one block per hit:

```text
dQw4w9WgXcQ  00:01:23  the exact matched segment
        Talk title — Chapter 3
```

A hit without a timestamp shows `--:--`.

### The chapter artefacts

`src/chapters.rs` persists the chapter markers and slices a transcript by
time. The chapter list is written as `<stem>.chapters.json`:

```json
[
  { "title": "Introduction", "start_time": 0.0, "end_time": 65.5 },
  { "title": "Main topic", "start_time": 65.5, "end_time": null }
]
```

The chaptered Markdown is the transcript sliced at the chapter boundaries,
each section headed by the chapter title.

## Constants

| Constant | Value | Owner |
|---|---|---|
| Index location | `$XDG_DATA_HOME/lc-rs/index.redb` | `paths::index_path` |
| Chapter bonus | 2 | `src/search.rs` |

## Failure modes

- **The index cannot be written** (locked by another process, corrupt
  file). A warning is logged; the run succeeds.
- **The index file is missing when searching.** `Error::Dependency` with a
  hint to run a transcription first.
- **A transcript file was deleted after it was indexed.** The index entry
  survives; the search still reports the hit and the timestamp.
- **The needle matches nothing.** `No matches for …`, exit code `1`.
