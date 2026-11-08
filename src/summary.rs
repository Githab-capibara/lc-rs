//! Optional post-transcript summary via a local Ollama instance.
//!
//! After the ASR stage writes `<id>.txt`, this module POSTs the transcript to
//! `http://localhost:11434/api/generate` and writes `<id>.summary.md` next to
//! it. Failures (Ollama down, unknown model, timeout) are warnings, not
//! errors: the transcript pipeline still succeeds.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::config;
use crate::errors::Result;

/// Default Ollama model used by `--summary`.
pub const DEFAULT_MODEL: &str = "llama3.1:8b";

/// Default Ollama endpoint used by `--summary`.
pub const DEFAULT_ENDPOINT: &str = "http://localhost:11434/api/generate";

/// Default summary timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// The built-in prompt template.
pub fn built_in_prompt() -> &'static str {
    include_str!("../assets/summary_prompt.txt")
}

/// Read the prompt template from `path`, or fall back to the built-in one.
pub fn load_prompt(path: Option<&Path>) -> Result<String> {
    match path {
        Some(path) => {
            Ok(std::fs::read_to_string(path).map_err(|e| crate::errors::Error::io(path, e))?)
        }
        None => Ok(built_in_prompt().to_string()),
    }
}

/// Insert the transcript into the prompt template.
pub fn build_prompt(template: &str, transcript: &str) -> String {
    if template.contains("{transcript}") {
        template.replace("{transcript}", transcript)
    } else {
        format!("{}\n\n---\n\n{transcript}", template.trim_end())
    }
}

/// The response body returned by Ollama.
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    #[serde(default)]
    response: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

/// POST `transcript` to Ollama and return the summary, or `None` on failure.
pub async fn generate(
    transcript: &str,
    model: &str,
    prompt: &str,
    endpoint: &str,
    timeout: Duration,
) -> Option<String> {
    // Ollama is local: a proxy would either fail the request or send the
    // transcript off-machine.
    let client = crate::downloader::direct_client().ok()?;
    let body = serde_json::json!({
        "model": model,
        "prompt": build_prompt(prompt, transcript),
        "stream": false,
    });

    let response = client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .timeout(timeout)
        .json(&body)
        .send()
        .await;

    let response = match response {
        Ok(response) => response,
        Err(err) if err.is_timeout() => {
            tracing::warn!(
                "Ollama request timed out after {}s. Summary skipped.",
                timeout.as_secs()
            );
            return None;
        }
        Err(err) => {
            tracing::warn!(
                "Ollama is not reachable at {endpoint} ({err}). Is Ollama running? Summary skipped."
            );
            return None;
        }
    };

    let status = response.status();
    if !status.is_success() {
        let detail = response.text().await.unwrap_or_default();
        let detail = detail.trim();
        tracing::warn!(
            "Ollama request failed (HTTP {}{}). Summary skipped.",
            status.as_u16(),
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {}", &detail[..detail.len().min(200)])
            }
        );
        return None;
    }

    let payload: OllamaResponse = match response.json().await {
        Ok(payload) => payload,
        Err(err) => {
            tracing::warn!("Ollama returned an unreadable body ({err}). Summary skipped.");
            return None;
        }
    };

    let text = payload.response.unwrap_or_default().trim().to_string();
    if payload.error.is_some() && text.is_empty() {
        tracing::warn!(
            "Ollama error: {}. Summary skipped.",
            payload.error.unwrap_or_default()
        );
        return None;
    }
    if text.is_empty() {
        tracing::warn!("Ollama returned an empty summary. Summary skipped.");
        return None;
    }
    Some(text)
}

/// Read `<id>.txt`, write `<id>.summary.md` next to it.
///
/// Never fails hard: callers can treat summarisation as best-effort.
pub async fn write_summary(
    txt_path: &Path,
    model: &str,
    prompt_path: Option<&Path>,
) -> Option<PathBuf> {
    if !txt_path.is_file() {
        tracing::warn!(
            "transcript not found at {}; skipping summary",
            txt_path.display()
        );
        return None;
    }
    let transcript = match std::fs::read_to_string(txt_path) {
        Ok(text) => text,
        Err(err) => {
            tracing::warn!(
                "could not read transcript at {}: {err}. Summary skipped.",
                txt_path.display()
            );
            return None;
        }
    };
    if transcript.trim().is_empty() {
        tracing::warn!("transcript is empty; skipping summary");
        return None;
    }
    let prompt = match load_prompt(prompt_path) {
        Ok(prompt) => prompt,
        Err(err) => {
            tracing::warn!("could not read summary prompt: {err}. Summary skipped.");
            return None;
        }
    };

    tracing::info!("ollama: model={model}");
    let text = generate(
        &transcript,
        model,
        &prompt,
        &config::summary_endpoint(),
        Duration::from_secs(config::DEFAULT_SUMMARY_TIMEOUT_SECS),
    )
    .await?;

    let stem = txt_path.file_stem().map_or_else(
        || "transcript".to_string(),
        |s| s.to_string_lossy().into_owned(),
    );
    let out = txt_path.with_file_name(format!("{stem}.summary.md"));
    if let Err(err) = std::fs::write(&out, format!("{text}\n")) {
        tracing::warn!(
            "could not write summary to {}: {err}. Summary skipped.",
            out.display()
        );
        return None;
    }
    tracing::info!("summary: {}", out.display());
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_substitution() {
        let template = "before {transcript} after";
        assert_eq!(build_prompt(template, "TEXT"), "before TEXT after");
    }

    #[test]
    fn prompt_without_placeholder_appends() {
        let out = build_prompt("summary please", "TEXT");
        assert!(out.ends_with("TEXT"));
        assert!(out.contains("---"));
    }

    #[test]
    fn built_in_prompt_has_placeholder() {
        assert!(built_in_prompt().contains("{transcript}"));
    }
}
