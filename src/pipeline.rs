//! End-to-end orchestration: source → transcript artefacts.
//!
//! The flow mirrors `src/localcaption/pipeline.py` of the Python
//! implementation:
//!
//! 1. `yt-dlp` downloads the best audio (or the source is a local file).
//! 2. `ffmpeg` re-encodes the audio to 16 kHz mono PCM WAV.
//! 3. The language identifier picks the ASR model.
//! 4. The chosen model transcribes the audio and the artefacts are written.
//! 5. Chapter markers (when the source has them) are folded into the
//!    transcript, and the transcript is added to the search index.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use crate::audio;
use crate::chapters::{self, Chapter, Segment};
use crate::downloader;
use crate::errors::{Error, IoContext, Result};
use crate::formatter;
use crate::languages;
use crate::ml;
use crate::models;
use crate::paths;
use crate::search;
use crate::summary;

/// Formats that `--output-format` understands.
pub const SUPPORTED_OUTPUT_FORMATS: [&str; 5] = ["md", "txt", "srt", "vtt", "json"];

/// Directory (relative to the output directory) holding the intermediates.
const WORK_DIR_NAME: &str = ".work";

/// Knobs of one transcription run.
#[derive(Debug, Clone)]
pub struct PipelineOptions {
    /// Directory the transcript artefacts are written to.
    pub out_dir: PathBuf,
    /// Language hint: `"auto"`, an ISO code or a language name.
    pub language: String,
    /// Registry key of an ASR model, overriding the language routing.
    pub force_model: Option<String>,
    /// Keep the downloaded audio and the intermediate WAV.
    pub keep_intermediate: bool,
    /// Basename of the transcript files; defaults to the media stem.
    pub stem: Option<String>,
    /// Write a summary through local Ollama.
    pub summary: bool,
    /// Ollama model used for the summary.
    pub summary_model: String,
    /// Prompt template used for the summary.
    pub summary_prompt: Option<PathBuf>,
    /// Requested output format(s).
    pub output_format: String,
    /// `yt-dlp` cookie source.
    pub cookies: Option<String>,
    /// Proxy override; an empty string disables the proxy.
    pub proxy: Option<String>,
    /// Request a fresh Tor exit circuit before downloading.
    pub force_rotate: bool,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("transcripts"),
            language: "auto".to_string(),
            force_model: None,
            keep_intermediate: false,
            stem: None,
            summary: false,
            summary_model: crate::config::summary_model(),
            summary_prompt: None,
            output_format: "md".to_string(),
            cookies: None,
            proxy: None,
            force_rotate: false,
        }
    }
}

/// Paths of the artefacts produced by one transcription run.
///
/// Every field is `None` when that format was not requested.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionResult {
    /// Plain-text transcript (or the Markdown file, when that was the only
    /// artefact — matching the Python implementation).
    pub txt: Option<PathBuf>,
    /// `SubRip` subtitles.
    pub srt: Option<PathBuf>,
    /// `WebVTT` subtitles.
    pub vtt: Option<PathBuf>,
    /// JSON transcript with the timed segments.
    pub json: Option<PathBuf>,
    /// Markdown transcript.
    pub md: Option<PathBuf>,
    /// Language of the transcript.
    pub language: String,
    /// Model that produced the transcript.
    pub model: String,
}

impl TranscriptionResult {
    /// The artefacts that exist on disk, in a stable order.
    pub fn existing(&self) -> Vec<(&'static str, PathBuf)> {
        let mut out = Vec::new();
        for (kind, path) in [
            ("txt", &self.txt),
            ("srt", &self.srt),
            ("vtt", &self.vtt),
            ("json", &self.json),
            ("md", &self.md),
        ] {
            if let Some(path) = path
                && path.is_file()
            {
                out.push((kind, path.clone()));
            }
        }
        out
    }
}

/// Everything one pipeline run produced.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The source that was transcribed.
    pub source_url: String,
    /// The downloaded media file, when it was kept.
    pub audio_path: Option<PathBuf>,
    /// The normalised 16 kHz WAV, when it was kept.
    pub wav_path: Option<PathBuf>,
    /// The transcript artefacts.
    pub transcripts: TranscriptionResult,
    /// Duration of the media in seconds, when known.
    pub duration_s: Option<f64>,
    /// The summary file, when one was written.
    pub summary: Option<PathBuf>,
    /// The chapter list, when it was written separately.
    pub chapters_json: Option<PathBuf>,
    /// The chaptered Markdown transcript, when it was written separately.
    pub chaptered_md: Option<PathBuf>,
}

/// Turn `"auto"` / `"Russian"` / `"ru"` into a routing code.
pub fn normalize_language(value: &str) -> String {
    let value = value.trim();
    if value.is_empty()
        || value.eq_ignore_ascii_case("auto")
        || value.eq_ignore_ascii_case("detect")
    {
        return "auto".to_string();
    }
    languages::code_for_name(value)
}

/// Decide which ASR model runs, and which language code is reported.
///
/// With `language = "auto"` the language identifier votes; an unsure result
/// falls back to the multilingual model.
pub fn resolve_model(
    language: &str,
    wav: &Path,
    force_model: Option<&str>,
) -> Result<(String, String)> {
    if let Some(forced) = force_model {
        let spec = models::get_model(forced)?;
        let mut code = normalize_language(language);
        if code == "auto" {
            code = match spec.role {
                models::ROLE_RU => "ru",
                models::ROLE_EN => "en",
                _ => "und",
            }
            .to_string();
        }
        return Ok((forced.to_string(), code));
    }

    let code = normalize_language(language);
    if code == "auto" {
        let result = ml::langid::detect_language(wav)?;
        if result.confident() {
            tracing::info!(
                "language: {} ({}, p={:.2})",
                result.name,
                result.code,
                result.score
            );
            return Ok((
                ml::asr::model_for_language(&result.code).to_string(),
                result.code,
            ));
        }
        tracing::warn!(
            "language unclear (best: {} p={:.2}) → multilingual model",
            result.name,
            result.score
        );
        return Ok((models::MODEL_NEMOTRON.to_string(), result.code.clone()));
    }

    Ok((ml::asr::model_for_language(&code).to_string(), code))
}

/// Whether `source` points at a local file rather than a URL.
fn is_local_file(source: &str) -> bool {
    !source.contains("://") && Path::new(source).is_file()
}

/// Run the whole pipeline on `source`.
pub async fn transcribe_url(source: &str, options: &PipelineOptions) -> Result<PipelineResult> {
    if let Some(forced) = &options.force_model {
        // Validate the key early so a typo does not fail after the download.
        models::get_model(forced)?;
    }

    let out_dir = &options.out_dir;
    paths::ensure_dir(out_dir)?;
    let work_dir = out_dir.join(WORK_DIR_NAME);
    paths::ensure_dir(&work_dir)?;

    let mut audio_path: Option<PathBuf> = None;
    let mut wav_path: Option<PathBuf> = None;
    let mut duration_s: Option<f64> = None;
    let mut chapters_json: Option<PathBuf> = None;
    let mut chaptered_md: Option<PathBuf> = None;
    let mut index_entry: Option<(search::IndexEntry, Vec<Segment>)> = None;

    let outcome = run_pipeline(
        source,
        options,
        &work_dir,
        &mut audio_path,
        &mut wav_path,
        &mut duration_s,
        &mut chapters_json,
        &mut chaptered_md,
        &mut index_entry,
    )
    .await;

    if !options.keep_intermediate {
        let _ = std::fs::remove_dir_all(&work_dir);
        audio_path = None;
        wav_path = None;
    }

    let transcripts = match outcome {
        Ok(transcripts) => transcripts,
        Err(err) => return Err(err),
    };

    if let Some((entry, segments)) = index_entry {
        let index_path = crate::config::index_path();
        if let Err(err) = search::upsert(&index_path, &entry, &segments) {
            // The index is best-effort: a finished run must not fail because
            // of a locked or corrupt database.
            tracing::warn!("could not update search index: {err}");
        }
    }

    let mut summary_path = None;
    if options.summary {
        match &transcripts.txt {
            Some(txt) => {
                summary_path = summary::write_summary(
                    txt,
                    &options.summary_model,
                    options.summary_prompt.as_deref(),
                )
                .await;
            }
            None => {
                tracing::warn!("summary skipped: no .txt artefact for this output format");
            }
        }
    }

    tracing::info!(
        "done ({}, {})",
        languages::name_for_code(&transcripts.language),
        transcripts.model
    );

    Ok(PipelineResult {
        source_url: source.to_string(),
        audio_path,
        wav_path,
        transcripts,
        duration_s,
        summary: summary_path,
        chapters_json,
        chaptered_md,
    })
}

/// The body of the pipeline, split out so the cleanup can run in all cases.
#[allow(clippy::too_many_arguments)]
async fn run_pipeline(
    source: &str,
    options: &PipelineOptions,
    work_dir: &Path,
    audio_path: &mut Option<PathBuf>,
    wav_path: &mut Option<PathBuf>,
    duration_s: &mut Option<f64>,
    chapters_json: &mut Option<PathBuf>,
    chaptered_md: &mut Option<PathBuf>,
    index_entry: &mut Option<(search::IndexEntry, Vec<Segment>)>,
) -> Result<TranscriptionResult> {
    let info: Value;
    let media: PathBuf;
    if is_local_file(source) {
        let path = std::fs::canonicalize(source).map_err(|e| Error::io(source, e))?;
        media = path.clone();
        info = serde_json::json!({
            "id": path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            "title": path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            "webpage_url": path.to_string_lossy(),
        });
    } else {
        let downloaded = downloader::download_audio(
            source,
            work_dir,
            options.cookies.as_deref(),
            options.proxy.as_deref(),
            options.force_rotate,
        )
        .await?;
        media = downloaded.path;
        info = downloaded.info;
    }
    *audio_path = Some(media.clone());

    let stem = media.file_stem().map_or_else(
        || "transcript".to_string(),
        |s| s.to_string_lossy().into_owned(),
    );
    let wav = work_dir.join(format!("{stem}.16k.wav"));
    audio::to_wav(&media, &wav).await?;
    *duration_s = audio::wav_duration_s(&wav);
    *wav_path = Some(wav.clone());

    let (model_key, code) = resolve_model(&options.language, &wav, options.force_model.as_deref())?;
    let out_base = options.out_dir.join(options.stem.clone().unwrap_or(stem));

    let asr = ml::asr::transcribe(&model_key, &wav, &code)?;
    let transcripts = write_output_artefacts(
        &out_base,
        &asr.segments,
        &asr.text,
        &model_key,
        &asr.language,
        &options.output_format,
    )?;

    let chapter_list = chapters::chapters_from_info(Some(&info));
    if !chapter_list.is_empty()
        && let Err(err) = write_chapter_artefacts(
            &out_base,
            &chapter_list,
            &transcripts,
            &options.output_format,
            chapters_json,
            chaptered_md,
        )
    {
        tracing::warn!("could not write chapter files: {err}");
    }

    *index_entry = Some(build_index_entry(
        source,
        &media,
        &info,
        &chapter_list,
        &transcripts,
        &asr.segments,
    ));

    Ok(transcripts)
}

/// Write the requested transcript artefacts.
///
/// `output_format` accepts `md`, `txt`, `srt`, `vtt`, `json` and
/// `all`/`full`/`complete` (every format).
fn write_output_artefacts(
    out_base: &Path,
    segments: &[Segment],
    text: &str,
    model: &str,
    detected: &str,
    output_format: &str,
) -> Result<TranscriptionResult> {
    if let Some(parent) = out_base.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_path(parent)?;
    }

    let formats = output_formats(output_format);
    let mut result = TranscriptionResult {
        language: detected.to_string(),
        model: model.to_string(),
        ..TranscriptionResult::default()
    };

    if formats.contains(&"md") {
        let path = formatter::output_file(out_base, ".md");
        result.md = Some(formatter::write_md(&path, text, segments)?);
    }
    if formats.contains(&"txt") {
        let path = formatter::output_file(out_base, ".txt");
        result.txt = Some(formatter::write_txt(&path, text)?);
    }
    if formats.contains(&"srt") {
        let path = formatter::output_file(out_base, ".srt");
        result.srt = Some(formatter::write_srt(&path, segments)?);
    }
    if formats.contains(&"vtt") {
        let path = formatter::output_file(out_base, ".vtt");
        result.vtt = Some(formatter::write_vtt(&path, segments)?);
    }
    if formats.contains(&"json") {
        let path = formatter::output_file(out_base, ".json");
        let payload = TranscriptJson {
            model,
            role: "",
            language: detected,
            language_name: &languages::name_for_code(detected),
            text,
            segments,
        };
        result.json = Some(formatter::write_json(&path, &payload)?);
    }

    // The Python implementation points `txt` at the Markdown artefact when
    // that is the only file that was written, so `--summary` and the
    // stdout echo work in the default single-file mode too.
    if result.md.is_some() {
        result.txt = result.md.clone();
    }

    Ok(result)
}

/// The set of formats an `--output-format` value selects.
fn output_formats(output_format: &str) -> Vec<&'static str> {
    let value = output_format.trim().to_lowercase();
    match value.as_str() {
        "all" | "full" | "complete" => SUPPORTED_OUTPUT_FORMATS.to_vec(),
        "txt" | "srt" | "vtt" | "json" => vec![
            SUPPORTED_OUTPUT_FORMATS
                .iter()
                .find(|f| **f == value)
                .copied()
                .unwrap_or("md"),
        ],
        _ => vec!["md"],
    }
}

/// The JSON document written next to the other transcript artefacts.
#[derive(Debug, Serialize)]
struct TranscriptJson<'a> {
    model: &'a str,
    role: &'a str,
    language: &'a str,
    language_name: &'a str,
    text: &'a str,
    segments: &'a [Segment],
}

/// Write the chapter artefacts, or fold them into the Markdown transcript.
fn write_chapter_artefacts(
    out_base: &Path,
    chapters: &[Chapter],
    transcripts: &TranscriptionResult,
    output_format: &str,
    chapters_json: &mut Option<PathBuf>,
    chaptered_md: &mut Option<PathBuf>,
) -> Result<()> {
    if output_format.trim().eq_ignore_ascii_case("md") {
        // Single-file mode: fold the chaptered transcript into the one `.md`
        // produced by the ASR stage so exactly one artefact remains.
        let Some(md_path) = &transcripts.md else {
            return Ok(());
        };
        let segments = chapters::load_segments(out_base);
        let fallback = if segments.is_empty() {
            std::fs::read_to_string(md_path).unwrap_or_default()
        } else {
            String::new()
        };
        let chaptered = chapters::build_chaptered_markdown(chapters, &segments, &fallback);
        if chaptered.is_empty() {
            return Ok(());
        }
        std::fs::write(md_path, chaptered).with_path(md_path)?;
        return Ok(());
    }

    let chapters_path = formatter::output_file(out_base, ".chapters.json");
    *chapters_json = Some(chapters::write_chapters_json(&chapters_path, chapters)?);

    let fallback = match &transcripts.txt {
        Some(path) if path.is_file() => std::fs::read_to_string(path).unwrap_or_default(),
        _ => String::new(),
    };
    let segments = chapters::load_segments(out_base);
    let chaptered_path = formatter::output_file(out_base, ".chaptered.md");
    *chaptered_md = Some(chapters::write_chaptered_md(
        &chaptered_path,
        chapters,
        &segments,
        &fallback,
    )?);
    Ok(())
}

/// Build the row that is stored in the search index.
fn build_index_entry(
    source: &str,
    media: &Path,
    info: &Value,
    chapters: &[Chapter],
    transcripts: &TranscriptionResult,
    segments: &[Segment],
) -> (search::IndexEntry, Vec<Segment>) {
    let media_stem = media
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let video_id = info
        .get("id")
        .and_then(Value::as_str)
        .map_or_else(|| media_stem.clone(), str::to_string);
    let title = info
        .get("title")
        .and_then(Value::as_str)
        .map_or_else(|| video_id.clone(), str::to_string);
    let webpage = info
        .get("webpage_url")
        .and_then(Value::as_str)
        .or_else(|| info.get("original_url").and_then(Value::as_str))
        .map_or_else(|| source.to_string(), str::to_string);
    let duration = info.get("duration").and_then(Value::as_f64);
    let transcript_path = transcripts
        .txt
        .as_ref()
        .or(transcripts.md.as_ref())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();

    let chapter_entries = chapters
        .iter()
        .map(|chapter| search::ChapterEntry {
            title: chapter.title.clone(),
            start_time: chapter.start_time,
            end_time: chapter.end_time,
        })
        .collect();

    let segments = segments.to_vec();

    (
        search::IndexEntry {
            id: video_id,
            url: webpage,
            title,
            duration,
            language: transcripts.language.clone(),
            model: transcripts.model.clone(),
            chapters: chapter_entries,
            transcript: transcript_path,
        },
        segments,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_language_hints() {
        assert_eq!(normalize_language("auto"), "auto");
        assert_eq!(normalize_language(""), "auto");
        assert_eq!(normalize_language("detect"), "auto");
        assert_eq!(normalize_language("Russian"), "ru");
        assert_eq!(normalize_language("ru"), "ru");
        assert_eq!(normalize_language("English"), "en");
    }

    #[test]
    fn resolves_output_formats() {
        assert_eq!(output_formats("md"), vec!["md"]);
        assert_eq!(output_formats("srt"), vec!["srt"]);
        assert_eq!(output_formats("all").len(), 5);
        assert_eq!(output_formats("full").len(), 5);
        assert_eq!(output_formats("complete").len(), 5);
        assert_eq!(output_formats("nonsense"), vec!["md"]);
    }

    #[test]
    fn writes_the_requested_artefacts() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("talk");
        let segments = vec![Segment {
            start: 0.0,
            end: 1.5,
            text: "Hello".to_string(),
        }];
        let result =
            write_output_artefacts(&base, &segments, "Hello", "parakeet", "en", "md").unwrap();
        assert!(result.md.as_ref().unwrap().is_file());
        assert!(result.json.is_none());
        // `txt` points at the Markdown artefact in single-file mode.
        assert_eq!(result.txt, result.md);
    }

    #[test]
    fn writes_every_artefact_for_all() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("talk");
        let result = write_output_artefacts(&base, &[], "Hello", "parakeet", "en", "all").unwrap();
        assert!(result.md.as_ref().unwrap().is_file());
        assert!(result.txt.as_ref().unwrap().is_file());
        assert!(result.srt.as_ref().unwrap().is_file());
        assert!(result.vtt.as_ref().unwrap().is_file());
        assert!(result.json.as_ref().unwrap().is_file());
    }
}
