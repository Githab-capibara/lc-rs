//! Registry of the local checkpoints `lc-rs` uses, plus their downloader.
//!
//! Unlike the Python implementation — which downloaded `PyTorch` checkpoints —
//! this port works directly on ONNX graphs published on Hugging Face. Every
//! artefact is pinned by size *and* SHA-256 so a truncated CDN response is
//! detected and retried instead of silently producing a broken model.

use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::config;
use crate::errors::{Error, IoContext, Result};
use crate::paths;

/// Registry key of the language-identification model.
pub const MODEL_LANGID: &str = "langid-ecapa";
/// Registry key of the Russian ASR model.
pub const MODEL_QWEN: &str = "qwen3-asr-0.6b";
/// Registry key of the English ASR model.
pub const MODEL_PARAKEET: &str = "parakeet-tdt-0.6b-v3";
/// Registry key of the multilingual ASR model.
pub const MODEL_NEMOTRON: &str = "nemotron-3.5-asr-streaming-0.6b";

/// Role played by the language-identification model.
pub const ROLE_LANGID: &str = "language-id";
/// Role played by the Russian ASR model.
pub const ROLE_RU: &str = "russian";
/// Role played by the English ASR model.
pub const ROLE_EN: &str = "english";
/// Role played by the multilingual ASR model.
pub const ROLE_MULTI: &str = "multilingual";

/// File name of the embedded language-identification graph.
pub const LANGID_FILE: &str = "langid_inline.onnx";

/// One artefact inside a model directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ModelFile {
    /// File name relative to the model directory.
    pub name: &'static str,
    /// Expected size in bytes.
    pub size: u64,
    /// Expected SHA-256 of the file contents, lowercase hex.
    pub sha256: &'static str,
}

/// Static metadata about one local checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ModelSpec {
    /// Registry key.
    pub key: &'static str,
    /// Hugging Face repository the artefacts are fetched from.
    pub hf_repo: &'static str,
    /// Role the model plays in the routing table.
    pub role: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// Approximate total download size, in megabytes.
    pub approx_size_mb: u64,
    /// Artefacts that make up the checkpoint.
    pub files: &'static [ModelFile],
}

impl ModelSpec {
    /// Local directory holding this checkpoint.
    pub fn local_dir(&self) -> PathBuf {
        paths::model_dir(self.key)
    }

    /// Base URL the artefacts are fetched from.
    pub fn url_base(&self) -> String {
        format!("https://huggingface.co/{}/resolve/main", self.hf_repo)
    }

    /// URL of one artefact.
    pub fn file_url(&self, file: &ModelFile) -> String {
        format!("{}/{}", self.url_base(), file.name)
    }

    /// Whether the model is embedded in the binary (no download required).
    pub fn is_embedded(&self) -> bool {
        self.key == MODEL_LANGID
    }
}

const LANGID_FILES: &[ModelFile] = &[ModelFile {
    name: LANGID_FILE,
    size: 84_110_845,
    sha256: "b4a7a328af0e9b75eb23bffc24d7b03d3dae0c28e758230cd9b6c68a39a3797c",
}];

const QWEN_FILES: &[ModelFile] = &[
    ModelFile {
        name: "encoder.onnx",
        size: 745_762_694,
        sha256: "3c027f880f677615de85e1f6934906e1d5d77624b724096d33accef99c753eed",
    },
    ModelFile {
        name: "decoder_init.int4.onnx",
        size: 355_390,
        sha256: "6d54633bbb9e6b1ecf372f218d04adb3bf6f2bfc442ce55c988526d532896faa",
    },
    ModelFile {
        name: "decoder_step.int4.onnx",
        size: 354_888,
        sha256: "7eaa4f31d5eb6ae2937ff45222c32f692f2d5ac29b46cbae9046da332a9f317d",
    },
    ModelFile {
        name: "decoder_weights.int4.data",
        size: 962_460_672,
        sha256: "d68cd1c0695a7ba42651d06b1bc1158e2e58af5f9adbb6dba874fbbb8a4f22cf",
    },
    ModelFile {
        name: "embed_tokens.bin",
        size: 311_164_928,
        sha256: "e80150119fa5f7e56e85aed64c3a02d5c78eb7a37cfdcb973d0987316f15bee2",
    },
    ModelFile {
        name: "tokenizer.json",
        size: 11_429_377,
        sha256: "bd2a97b55c8f7f9c328c73ee9b9178771037e9f566dfca8e238a063d41cbac92",
    },
    ModelFile {
        name: "config.json",
        size: 1_250,
        sha256: "df31c4689abe9d782366fffd2454b546291d0205d082b3fd01b99fb76a45b11f",
    },
    ModelFile {
        name: "preprocessor_config.json",
        size: 298,
        sha256: "9c7c558a05f326fe4365e5e59e486383ff127dfd93ce1cd5e6e23f883cd02281",
    },
];

const PARAKEET_FILES: &[ModelFile] = &[
    ModelFile {
        name: "encoder-model.int8.onnx",
        size: 652_183_999,
        sha256: "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
    },
    ModelFile {
        name: "decoder_joint-model.int8.onnx",
        size: 18_202_004,
        sha256: "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
    },
    ModelFile {
        name: "nemo128.onnx",
        size: 139_764,
        sha256: "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f",
    },
    ModelFile {
        name: "vocab.txt",
        size: 93_939,
        sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    },
    ModelFile {
        name: "config.json",
        size: 97,
        sha256: "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
    },
];

/// Files of the Nemotron export this port was validated against: the three
/// graphs plus the vocabulary and the `GenAI` manifest that documents the
/// graph
/// semantics. The upstream folder also ships a Silero VAD graph, a tokenizer
/// and a processor config, none of which this port reads.
const NEMOTRON_FILES: &[ModelFile] = &[
    ModelFile {
        name: "encoder.onnx",
        size: 2_815_853,
        sha256: "008f9f689453a097dc98680eb4c8ce47a0ed3cb7c20498ff321dcb223b28dac7",
    },
    ModelFile {
        name: "encoder.onnx.data",
        size: 967_176_192,
        sha256: "d758dd3d6d8241a362c39ebad3ad3138a5f18c24248acee6c963837ffcb3c36c",
    },
    ModelFile {
        name: "decoder.onnx",
        size: 4_696,
        sha256: "9277dacdb7d0e2e32c8969f733ef8808fe438260adf1ea91875873b832fe5a41",
    },
    ModelFile {
        name: "decoder.onnx.data",
        size: 59_785_216,
        sha256: "e5fd55cbeeb268f9d383e2ee72735b9fbbb13aea4bc7cd38cb73b8e16f1366c7",
    },
    ModelFile {
        name: "joint.onnx",
        size: 2_136,
        sha256: "f585442de8cce8531f5ab2832ac75920b062ef6e354df79bec1b51d45bbc3737",
    },
    ModelFile {
        name: "joint.onnx.data",
        size: 37_830_656,
        sha256: "2e0fb1c060f3777a1a76e78d5589dd54f01505a06dffbd2588e315508b402c12",
    },
    ModelFile {
        name: "vocab.txt",
        size: 77_112,
        sha256: "646185775356ffa0b06f87f5e2570c3574695e1132328c82c78e07d1940a7ac2",
    },
    ModelFile {
        name: "genai_config.json",
        size: 1_962,
        sha256: "0fdbafa35aca9db89c69f82cadb7b66a2e5260114fcc19d05e87d87dde15fb4f",
    },
];

/// Every registered model.
const REGISTRY: &[ModelSpec] = &[
    ModelSpec {
        key: MODEL_LANGID,
        hf_repo: "local/lc-rs-assets",
        role: ROLE_LANGID,
        description: "ECAPA-TDNN language identifier, 45 languages (routing)",
        approx_size_mb: 84,
        files: LANGID_FILES,
    },
    ModelSpec {
        key: MODEL_QWEN,
        hf_repo: "andrewleech/qwen3-asr-0.6b-onnx",
        role: ROLE_RU,
        description: "Qwen3-ASR 0.6B — Russian (WER 5.13)",
        approx_size_mb: 2_030,
        files: QWEN_FILES,
    },
    ModelSpec {
        key: MODEL_PARAKEET,
        hf_repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
        role: ROLE_EN,
        description: "NVIDIA Parakeet TDT 0.6B v3 — English (Open ASR #1)",
        approx_size_mb: 670,
        files: PARAKEET_FILES,
    },
    ModelSpec {
        key: MODEL_NEMOTRON,
        hf_repo: "DimQ1/nemotron-3.5-asr-streaming-0.6b-onnx-int8-cpu",
        role: ROLE_MULTI,
        description: "NVIDIA Nemotron 3.5 ASR Streaming 0.6B — 40 languages, int8 graphs",
        approx_size_mb: 1_021,
        files: NEMOTRON_FILES,
    },
];

/// Every registered model.
pub fn known_models() -> &'static [ModelSpec] {
    REGISTRY
}

/// Look a model up by key.
pub fn get_model(key: &str) -> Result<&'static ModelSpec> {
    REGISTRY.iter().find(|spec| spec.key == key).ok_or_else(|| {
        let valid = REGISTRY
            .iter()
            .map(|m| m.key)
            .collect::<Vec<_>>()
            .join(", ");
        Error::Model(format!(
            "Unknown model: {key:?}\nRun `lc-rs model list` to see the supported models.\nValid keys: {valid}"
        ))
    })
}

/// Look a model up by role.
pub fn by_role(role: &str) -> Result<&'static ModelSpec> {
    REGISTRY
        .iter()
        .find(|spec| spec.role == role)
        .ok_or_else(|| Error::Model(format!("No model registered for role {role:?}")))
}

/// Whether one artefact is present and plausible.
fn file_ok(path: &Path, file: &ModelFile) -> bool {
    match std::fs::metadata(path) {
        Ok(meta) => meta.is_file() && meta.len() == file.size,
        Err(_) => false,
    }
}

/// Artefacts of `key` that are missing or truncated.
pub fn missing_files(key: &str) -> Result<Vec<&'static ModelFile>> {
    let spec = get_model(key)?;
    if spec.is_embedded() {
        // The language-identifier graph ships inside the binary.
        return Ok(Vec::new());
    }
    let dir = spec.local_dir();
    Ok(spec
        .files
        .iter()
        .filter(|file| !file_ok(&dir.join(file.name), file))
        .collect())
}

/// Whether the model is fully installed.
pub fn is_installed(key: &str) -> bool {
    missing_files(key).is_ok_and(|m| m.is_empty())
}

/// Keys of the models that are not installed.
pub fn missing_models(keys: Option<&[String]>) -> Vec<String> {
    let check: Vec<String> = match keys {
        Some(keys) => keys.to_vec(),
        None => REGISTRY.iter().map(|m| m.key.to_string()).collect(),
    };
    check.into_iter().filter(|k| !is_installed(k)).collect()
}

/// Raise a copy-pasteable error if any required model is incomplete.
pub fn require(keys: Option<&[String]>) -> Result<()> {
    let missing = missing_models(keys);
    if missing.is_empty() {
        return Ok(());
    }
    let mut lines = vec!["Missing or incomplete model checkpoints:".to_string()];
    for key in &missing {
        let files = missing_files(key)?;
        let names: Vec<&str> = files.iter().map(|f| f.name).collect();
        lines.push(format!("  - {key}: missing {}", names.join(", ")));
    }
    lines.push(String::new());
    lines.push(format!(
        "Download them with:  lc-rs model download {}",
        missing[0]
    ));
    lines.push("Or download everything:  lc-rs model download --all".to_string());
    Err(Error::Dependency(lines.join("\n")))
}

/// Installation status of one model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    /// The model metadata.
    pub spec: &'static ModelSpec,
    /// Whether every artefact is present.
    pub installed: bool,
    /// Artefacts that are missing.
    pub missing: Vec<&'static str>,
}

/// Installation status of every model.
pub fn list_status() -> Vec<ModelStatus> {
    REGISTRY
        .iter()
        .map(|spec| ModelStatus {
            spec,
            installed: is_installed(spec.key),
            missing: missing_files(spec.key)
                .unwrap_or_default()
                .into_iter()
                .map(|f| f.name)
                .collect(),
        })
        .collect()
}

/// Remove an installed model from disk.
pub fn remove_model(key: &str) -> Result<PathBuf> {
    let spec = get_model(key)?;
    let root = spec.local_dir();
    if !root.exists() {
        return Err(Error::Model(format!(
            "Model {key:?} is not installed at {}",
            root.display()
        )));
    }
    std::fs::remove_dir_all(&root).with_path(&root)?;
    Ok(root)
}

/// Compute the SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).with_path(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).with_path(path)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Outcome of downloading one model.
#[derive(Debug, Clone)]
pub struct DownloadOutcome {
    /// Directory the model was installed into.
    pub dir: PathBuf,
    /// Artefacts that were actually fetched.
    pub downloaded: Vec<String>,
    /// Artefacts that were already present.
    pub skipped: Vec<String>,
}

/// Download every artefact of `key`.
///
/// Downloads are resumable (HTTP `Range`), verified against the pinned
/// size and SHA-256, and moved into place atomically so an interrupted run
/// never leaves a half-written checkpoint behind.
pub async fn download_model(key: &str, force: bool) -> Result<DownloadOutcome> {
    let spec = get_model(key)?;
    let dir = spec.local_dir();
    paths::ensure_dir(&dir)?;

    if spec.is_embedded() {
        return install_embedded(spec, &dir, force).await;
    }

    let client = crate::downloader::http_client()?;
    let mut downloaded = Vec::new();
    let mut skipped = Vec::new();

    for file in spec.files {
        let dest = dir.join(file.name);
        if !force && file_ok(&dest, file) {
            skipped.push(file.name.to_string());
            continue;
        }
        tracing::info!("downloading {} ({} bytes)", file.name, file.size);
        download_file(&client, spec, file, &dest).await?;
        downloaded.push(file.name.to_string());
    }

    Ok(DownloadOutcome {
        dir,
        downloaded,
        skipped,
    })
}

/// Install the embedded language-identifier graph into the model directory.
async fn install_embedded(
    spec: &'static ModelSpec,
    dir: &Path,
    force: bool,
) -> Result<DownloadOutcome> {
    let file = &spec.files[0];
    let dest = dir.join(file.name);
    if !force && file_ok(&dest, file) {
        return Ok(DownloadOutcome {
            dir: dir.to_path_buf(),
            downloaded: Vec::new(),
            skipped: vec![file.name.to_string()],
        });
    }
    let bytes = crate::ml::langid::embedded_graph();
    let tmp = dir.join(format!("{}.part", file.name));
    std::fs::write(&tmp, bytes).with_path(&tmp)?;
    tokio::fs::rename(&tmp, &dest).await.with_path(&dest)?;
    Ok(DownloadOutcome {
        dir: dir.to_path_buf(),
        downloaded: vec![file.name.to_string()],
        skipped: Vec::new(),
    })
}

/// Download every registered model.
pub async fn download_all(force: bool) -> Result<Vec<DownloadOutcome>> {
    let mut out = Vec::new();
    for spec in REGISTRY {
        out.push(download_model(spec.key, force).await?);
    }
    Ok(out)
}

/// Fetch one artefact with resume, checksum verification and retries.
async fn download_file(
    client: &reqwest::Client,
    spec: &ModelSpec,
    file: &ModelFile,
    dest: &Path,
) -> Result<()> {
    let url = spec.file_url(file);
    let tmp = dest.with_file_name(format!("{}.part", file.name));
    let mut last_error: Option<String> = None;

    for attempt in 1..=4u32 {
        let existing = std::fs::metadata(&tmp).map_or(0, |m| m.len());
        if existing >= file.size {
            // A previous run left a complete (but unverified) file: re-check it.
            let _ = std::fs::remove_file(&tmp);
        }
        let start = std::fs::metadata(&tmp).map_or(0, |m| m.len());
        if start > file.size {
            let _ = std::fs::remove_file(&tmp);
        }

        let mut request = client.get(&url);
        if start > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={start}-"));
        }

        match request.send().await {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() {
                    last_error = Some(format!("HTTP {status}"));
                    tracing::warn!(
                        "download of {} failed (HTTP {status}), attempt {attempt}/4",
                        file.name
                    );
                    continue;
                }
                let append = start > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
                if start > 0 && !append {
                    // The server ignored our Range request: start over.
                    let _ = std::fs::remove_file(&tmp);
                } else if append && !content_range_starts_at(response.headers(), start) {
                    // The server answered with a *different* range than the one
                    // we asked for. Appending it would produce a file of the
                    // right size but the wrong content, so the partial file is
                    // discarded and the download starts over.
                    last_error = Some(format!(
                        "the server answered with an unexpected range (wanted bytes={start}-)"
                    ));
                    tracing::warn!(
                        "download of {} got an inconsistent range, restarting",
                        file.name
                    );
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }
                let mut file_out = tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(append)
                    .write(true)
                    .truncate(!append)
                    .open(&tmp)
                    .await
                    .with_path(&tmp)?;
                use tokio::io::AsyncWriteExt;
                let mut stream = response.bytes_stream();
                use futures_util::StreamExt;
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.map_err(|e| Error::Download(format!("{url}: {e}")))?;
                    file_out.write_all(&chunk).await.with_path(&tmp)?;
                }
                file_out.flush().await.with_path(&tmp)?;
                drop(file_out);

                let got = std::fs::metadata(&tmp).map_or(0, |m| m.len());
                if got != file.size {
                    last_error = Some(format!(
                        "size mismatch: got {got} bytes, expected {}",
                        file.size
                    ));
                    tracing::warn!(
                        "download of {} truncated ({got}/{}), retrying",
                        file.name,
                        file.size
                    );
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }
                let digest = sha256_file(&tmp)?;
                if digest != file.sha256 {
                    last_error = Some(format!(
                        "checksum mismatch: got {digest}, expected {}",
                        file.sha256
                    ));
                    tracing::warn!("download of {} failed integrity check, retrying", file.name);
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }
                tokio::fs::rename(&tmp, dest).await.with_path(dest)?;
                return Ok(());
            }
            Err(err) => {
                last_error = Some(err.to_string());
                tracing::warn!(
                    "download of {} failed ({err}), attempt {attempt}/4",
                    file.name
                );
            }
        }
    }

    Err(Error::Install(format!(
        "Failed to download {} from {url}: {}",
        file.name,
        last_error.unwrap_or_else(|| "unknown error".to_string())
    )))
}

/// Root directory every model is installed under.
pub fn models_root() -> PathBuf {
    config::models_dir()
}

/// Whether the `Content-Range` header of a response starts at `expected`.
///
/// A server that answers a `Range: bytes=N-` request with a *different*
/// range must not be trusted: appending the response to the partial file
/// would produce a file of exactly the expected size but with the wrong
/// bytes in the middle. The SHA-256 check would catch it, but only after
/// gigabytes of traffic; rejecting the response here is cheaper.
fn content_range_starts_at(headers: &reqwest::header::HeaderMap, expected: u64) -> bool {
    let Some(value) = headers.get(reqwest::header::CONTENT_RANGE) else {
        // No `Content-Range` at all: the server did not describe what it
        // sent, so the range cannot be trusted.
        return false;
    };
    let Ok(text) = value.to_str() else {
        return false;
    };
    // The header looks like `bytes 1234-9999/3006500864`.
    let Some(range) = text.trim().strip_prefix("bytes ") else {
        return false;
    };
    let Some((first, _)) = range.split_once('-') else {
        return false;
    };
    first
        .trim()
        .parse::<u64>()
        .is_ok_and(|start| start == expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_four_models() {
        assert_eq!(known_models().len(), 4);
    }

    #[test]
    fn get_model_rejects_unknown_keys() {
        assert!(get_model("nope").is_err());
        assert_eq!(get_model(MODEL_LANGID).unwrap().role, ROLE_LANGID);
    }

    #[test]
    fn by_role_works() {
        assert_eq!(by_role(ROLE_RU).unwrap().key, MODEL_QWEN);
        assert_eq!(by_role(ROLE_EN).unwrap().key, MODEL_PARAKEET);
        assert_eq!(by_role(ROLE_MULTI).unwrap().key, MODEL_NEMOTRON);
    }

    #[test]
    fn embedded_model_is_always_installed() {
        assert!(missing_files(MODEL_LANGID).unwrap().is_empty());
        assert!(is_installed(MODEL_LANGID));
    }

    #[test]
    fn file_urls_are_https() {
        let spec = get_model(MODEL_PARAKEET).unwrap();
        let url = spec.file_url(&spec.files[0]);
        assert!(url.starts_with("https://huggingface.co/"));
    }

    /// A resume must only be accepted when the server answered with exactly
    /// the range that was requested.
    #[test]
    fn content_range_is_validated() {
        let headers = |value: Option<&str>| {
            let mut map = reqwest::header::HeaderMap::new();
            if let Some(value) = value {
                map.insert(
                    reqwest::header::CONTENT_RANGE,
                    reqwest::header::HeaderValue::from_str(value).unwrap(),
                );
            }
            map
        };

        assert!(content_range_starts_at(
            &headers(Some("bytes 100-999/1000")),
            100
        ));
        assert!(content_range_starts_at(
            &headers(Some("bytes 100-/1000")),
            100
        ));

        // A different range than requested.
        assert!(!content_range_starts_at(
            &headers(Some("bytes 0-999/1000")),
            100
        ));
        assert!(!content_range_starts_at(
            &headers(Some("bytes 200-999/1000")),
            100
        ));

        // No header at all, or a malformed one.
        assert!(!content_range_starts_at(&headers(None), 100));
        assert!(!content_range_starts_at(
            &headers(Some("bytes abc-def/1000")),
            100
        ));
        assert!(!content_range_starts_at(
            &headers(Some("items 100-999/1000")),
            100
        ));
        assert!(!content_range_starts_at(&headers(Some("bytes 100")), 100));
    }
}
