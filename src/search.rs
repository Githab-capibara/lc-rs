//! Transcript search index, backed by redb.
//!
//! The Python implementation kept a JSONL file and re-read every transcript
//! from disk on each search. This port uses an embedded key-value store
//! instead: transcripts and their timed segments are stored alongside each
//! other, so a search never touches the filesystem.
//!
//! The ranking rules are unchanged: a hit scores the number of occurrences
//! of the needle in the matched text, plus the number of occurrences in the
//! title, and chapter hits get a `+2` bonus.

use std::path::Path;

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::chapters::Segment;
use crate::errors::{Error, IoContext, Result};

/// Table holding one JSON document per transcript.
const TRANSCRIPTS: TableDefinition<&str, &str> = TableDefinition::new("transcripts");

/// Table holding the timed segments of each transcript.
const SEGMENTS: TableDefinition<&str, &str> = TableDefinition::new("segments");

/// One indexed transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Stable identifier (the video id, or a sanitised path).
    pub id: String,
    /// Original URL or path.
    pub url: String,
    /// Human-readable title.
    pub title: String,
    /// Media duration in seconds, when known.
    pub duration: Option<f64>,
    /// Detected language code.
    pub language: String,
    /// Model that produced the transcript.
    pub model: String,
    /// Chapter markers.
    pub chapters: Vec<ChapterEntry>,
    /// Absolute path of the transcript artefact.
    pub transcript: String,
}

/// One chapter marker as stored in the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterEntry {
    /// Chapter title.
    pub title: String,
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds, when known.
    pub end_time: Option<f64>,
}

/// One search result.
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    /// Identifier of the transcript.
    pub id: String,
    /// Original URL or path.
    pub url: String,
    /// Title of the transcript.
    pub title: String,
    /// Timestamp of the hit, when known.
    pub start: Option<f64>,
    /// The matched text.
    pub text: String,
    /// Relevance score (higher is better).
    pub score: i64,
    /// Absolute path of the transcript artefact.
    pub transcript: String,
}

/// Open (or create) the index database.
fn open(path: &Path) -> Result<Database> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_path(parent)?;
    }
    Database::create(path).map_err(|e| {
        Error::Dependency(format!(
            "could not open the search index at {}: {e}",
            path.display()
        ))
    })
}

/// Insert or replace the entry with the same id.
pub fn upsert(path: &Path, entry: &IndexEntry, segments: &[Segment]) -> Result<()> {
    let db = open(path)?;
    let write = db
        .begin_write()
        .map_err(|e| Error::Dependency(format!("could not lock the search index: {e}")))?;
    {
        let mut table = write
            .open_table(TRANSCRIPTS)
            .map_err(|e| Error::Dependency(format!("could not open the index table: {e}")))?;
        let payload = serde_json::to_string(entry)?;
        table
            .insert(entry.id.as_str(), payload.as_str())
            .map_err(|e| Error::Dependency(format!("could not write the index row: {e}")))?;
    }
    {
        let mut table = write
            .open_table(SEGMENTS)
            .map_err(|e| Error::Dependency(format!("could not open the segment table: {e}")))?;
        let payload = serde_json::to_string(segments)?;
        table
            .insert(entry.id.as_str(), payload.as_str())
            .map_err(|e| Error::Dependency(format!("could not write the index segments: {e}")))?;
    }
    write
        .commit()
        .map_err(|e| Error::Dependency(format!("could not commit the index update: {e}")))?;
    Ok(())
}

/// Every indexed transcript.
pub fn load(path: &Path) -> Result<Vec<IndexEntry>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let db = open(path)?;
    let read = db
        .begin_read()
        .map_err(|e| Error::Dependency(format!("could not lock the search index: {e}")))?;
    let table = read
        .open_table(TRANSCRIPTS)
        .map_err(|e| Error::Dependency(format!("could not open the index table: {e}")))?;
    let mut entries = Vec::new();
    for row in table
        .iter()
        .map_err(|e| Error::Dependency(format!("could not read the index: {e}")))?
    {
        let (_, value) =
            row.map_err(|e| Error::Dependency(format!("could not read the index: {e}")))?;
        if let Ok(entry) = serde_json::from_str::<IndexEntry>(value.value()) {
            entries.push(entry);
        }
    }
    Ok(entries)
}

/// Timed segments of one indexed transcript.
pub fn load_segments(path: &Path, id: &str) -> Result<Vec<Segment>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let db = open(path)?;
    let read = db
        .begin_read()
        .map_err(|e| Error::Dependency(format!("could not lock the search index: {e}")))?;
    let table = read
        .open_table(SEGMENTS)
        .map_err(|e| Error::Dependency(format!("could not open the segment table: {e}")))?;
    let Some(value) = table
        .get(id)
        .map_err(|e| Error::Dependency(format!("could not read the index segments: {e}")))?
    else {
        return Ok(Vec::new());
    };
    Ok(serde_json::from_str(value.value()).unwrap_or_default())
}

/// Remove one transcript from the index.
pub fn remove(path: &Path, id: &str) -> Result<()> {
    let db = open(path)?;
    let write = db
        .begin_write()
        .map_err(|e| Error::Dependency(format!("could not lock the search index: {e}")))?;
    {
        let mut table = write
            .open_table(TRANSCRIPTS)
            .map_err(|e| Error::Dependency(format!("could not open the index table: {e}")))?;
        table
            .remove(id)
            .map_err(|e| Error::Dependency(format!("could not remove the index row: {e}")))?;
    }
    {
        let mut table = write
            .open_table(SEGMENTS)
            .map_err(|e| Error::Dependency(format!("could not open the segment table: {e}")))?;
        table
            .remove(id)
            .map_err(|e| Error::Dependency(format!("could not remove the index segments: {e}")))?;
    }
    write
        .commit()
        .map_err(|e| Error::Dependency(format!("could not commit the index update: {e}")))?;
    Ok(())
}

/// Search every indexed transcript for `term` (case-insensitive substring).
pub fn search(path: &Path, term: &str) -> Result<Vec<SearchHit>> {
    let needle = term.trim().to_lowercase();
    if needle.is_empty() {
        return Ok(Vec::new());
    }
    let entries = load(path)?;
    let mut hits = Vec::new();
    for entry in entries {
        let segments = load_segments(path, &entry.id).unwrap_or_default();
        hits.extend(hits_for_entry(&entry, &segments, &needle));
    }
    hits.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.id.cmp(&b.id))
            .then_with(|| {
                let left = a.start.unwrap_or(0.0);
                let right = b.start.unwrap_or(0.0);
                left.partial_cmp(&right)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    Ok(hits)
}

/// The segments to search in, with the reference implementation's fallback.
///
/// The indexed segments are authoritative. When a transcript has none — it
/// was indexed without them, or it was never segmented — the sibling
/// `.json` or `.srt` is read, and failing that the transcript file itself
/// is split into one synthetic segment per non-empty line. Without that
/// fallback a transcript without timed markers would be unsearchable.
fn fallback_segments(entry: &IndexEntry, segments: &[Segment]) -> Vec<Segment> {
    if !segments.is_empty() {
        return segments.to_vec();
    }
    let path = Path::new(&entry.transcript);
    if path.as_os_str().is_empty() {
        return Vec::new();
    }
    let from_disk = crate::chapters::load_segments(path);
    if !from_disk.is_empty() {
        return from_disk;
    }
    let Ok(body) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    body.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| Segment {
            start: -1.0,
            end: -1.0,
            text: line.to_string(),
        })
        .collect()
}

/// Compute the hits of one transcript.
fn hits_for_entry(entry: &IndexEntry, segments: &[Segment], needle: &str) -> Vec<SearchHit> {
    let title_bonus = entry.title.to_lowercase().matches(needle).count() as i64;
    let mut hits = Vec::new();

    for chapter in &entry.chapters {
        let count = chapter.title.to_lowercase().matches(needle).count() as i64;
        if count == 0 {
            continue;
        }
        hits.push(SearchHit {
            id: entry.id.clone(),
            url: entry.url.clone(),
            title: entry.title.clone(),
            start: Some(chapter.start_time),
            text: format!("[chapter] {}", chapter.title),
            score: count + title_bonus + 2,
            transcript: entry.transcript.clone(),
        });
    }

    for segment in fallback_segments(entry, segments) {
        let count = segment.text.to_lowercase().matches(needle).count() as i64;
        if count == 0 {
            continue;
        }
        hits.push(SearchHit {
            id: entry.id.clone(),
            url: entry.url.clone(),
            title: entry.title.clone(),
            start: if segment.start >= 0.0 {
                Some(segment.start)
            } else {
                None
            },
            text: segment.text.trim().to_string(),
            score: count + title_bonus,
            transcript: entry.transcript.clone(),
        });
    }

    if hits.is_empty() && title_bonus > 0 {
        hits.push(SearchHit {
            id: entry.id.clone(),
            url: entry.url.clone(),
            title: entry.title.clone(),
            start: None,
            text: entry.title.clone(),
            score: title_bonus,
            transcript: entry.transcript.clone(),
        });
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn entry() -> IndexEntry {
        IndexEntry {
            id: "abc123".into(),
            url: "https://example.com/watch?v=abc123".into(),
            title: "Rust Conference 2024".into(),
            duration: Some(120.0),
            language: "en".into(),
            model: "parakeet-tdt-0.6b-v3".into(),
            chapters: vec![ChapterEntry {
                title: "Introduction".into(),
                start_time: 0.0,
                end_time: Some(10.0),
            }],
            transcript: "/tmp/abc123.md".into(),
        }
    }

    #[test]
    fn upsert_and_load_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.redb");
        let segments = vec![Segment {
            start: 1.0,
            end: 2.0,
            text: "hello rust".into(),
        }];
        upsert(&path, &entry(), &segments).unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "abc123");
        assert_eq!(load_segments(&path, "abc123").unwrap().len(), 1);
    }

    #[test]
    fn search_finds_segments_and_chapters() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.redb");
        let segments = vec![
            Segment {
                start: 1.0,
                end: 2.0,
                text: "hello rust".into(),
            },
            Segment {
                start: 5.0,
                end: 6.0,
                text: "nothing here".into(),
            },
        ];
        upsert(&path, &entry(), &segments).unwrap();
        let hits = search(&path, "rust").unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().any(|h| h.text.contains("hello rust")));
        // The chapter title does not contain "rust", but the title does, so
        // the title bonus applies.
        assert!(hits.iter().all(|h| h.score >= 1));
    }

    #[test]
    fn search_is_case_insensitive() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.redb");
        upsert(
            &path,
            &entry(),
            &[Segment {
                start: 0.0,
                end: 1.0,
                text: "Hello Rust".into(),
            }],
        )
        .unwrap();
        assert_eq!(search(&path, "hello").unwrap().len(), 1);
        assert_eq!(search(&path, "HELLO").unwrap().len(), 1);
        assert_eq!(search(&path, "absent").unwrap().len(), 0);
    }

    #[test]
    fn empty_term_returns_nothing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.redb");
        upsert(&path, &entry(), &[]).unwrap();
        assert!(search(&path, "   ").unwrap().is_empty());
    }

    #[test]
    fn missing_index_is_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("missing.redb");
        assert!(load(&path).unwrap().is_empty());
        assert!(search(&path, "anything").unwrap().is_empty());
    }

    /// A transcript without timed segments is still searchable.
    ///
    /// The reference implementation falls back to one synthetic segment per
    /// non-empty line of the transcript file, so a match in an unsegmented
    /// transcript is found instead of being silently dropped.
    #[test]
    fn an_unsegmented_transcript_is_still_searchable() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("index.redb");
        let transcript = dir.path().join("abc123.md");
        std::fs::write(&transcript, "First line\n\nSecond line with rust\n").unwrap();

        let mut entry = entry();
        entry.transcript = transcript.to_string_lossy().into_owned();
        // No segments are indexed, and no sibling `.json` or `.srt` exists.
        upsert(&path, &entry, &[]).unwrap();

        let hits = search(&path, "rust").unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("Second line with rust"));
        // A line without a timestamp carries no start.
        assert!(hits[0].start.is_none());
    }
}
