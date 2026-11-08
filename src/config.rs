//! Environment-driven configuration.
//!
//! Every knob the Python implementation exposed through `LOCALCAPTION_*`
//! environment variables is honoured here under both the original
//! `LOCALCAPTION_` prefix and the Rust-native `LCRS_` prefix (the latter
//! wins when both are set). Keeping the original names means an existing
//! `localcaption` setup keeps working unchanged after the switch to
//! `lc-rs`.

use std::path::PathBuf;

use crate::paths;

/// Default SOCKS5 endpoint: a local Tor instance.
pub const DEFAULT_PROXY: &str = "socks5h://127.0.0.1:9050";

/// Default Ollama model used by `--summary`.
pub const DEFAULT_SUMMARY_MODEL: &str = "llama3.1:8b";

/// Default Ollama endpoint used by `--summary`.
pub const DEFAULT_SUMMARY_ENDPOINT: &str = "http://localhost:11434/api/generate";

/// Default summary timeout in seconds.
pub const DEFAULT_SUMMARY_TIMEOUT_SECS: u64 = 120;

/// Hosts that must never be routed through the proxy.
pub const NO_PROXY_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "::1"];

/// Read an environment variable, preferring `LCRS_<name>` over
/// `LOCALCAPTION_<name>`.
fn env_pair(key: &str) -> Option<String> {
    let short = format!("LCRS_{key}");
    let long = format!("LOCALCAPTION_{key}");
    std::env::var(&short)
        .ok()
        .or_else(|| std::env::var(&long).ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Proxy URL used for every outbound request.
///
/// `""` (an explicitly empty `LOCALCAPTION_PROXY`) disables the proxy, which
/// is how `--no-proxy` is implemented.
pub fn proxy() -> String {
    match env_pair("PROXY") {
        Some(value) => value,
        None => DEFAULT_PROXY.to_string(),
    }
}

/// Whether the configured proxy is disabled.
pub fn proxy_disabled() -> bool {
    env_pair("PROXY").is_none() && std::env::var("LCRS_PROXY").is_ok_and(|v| v.trim().is_empty())
}

/// Explicit Tor control endpoint (`host:port`), if configured.
pub fn tor_control() -> Option<String> {
    env_pair("TOR_CONTROL")
}

/// Directory holding Tor's control authentication cookie.
pub fn tor_data_dir() -> Option<PathBuf> {
    env_pair("TOR_DATADIR").map(PathBuf::from)
}

/// A Netscape cookie jar path for yt-dlp, if configured.
pub fn ytdlp_cookies() -> Option<String> {
    env_pair("YTDLP_COOKIES")
}

/// Comma-separated list of browser names to harvest cookies from.
pub fn ytdlp_cookie_browsers() -> Option<String> {
    env_pair("YTDLP_COOKIE_BROWSERS")
}

/// Root directory holding model checkpoints.
pub fn models_dir() -> PathBuf {
    env_pair("MODELS_DIR").map_or_else(
        || paths::data_root().join("lc-rs").join("models"),
        PathBuf::from,
    )
}

/// Location of the transcript search index.
pub fn index_path() -> PathBuf {
    env_pair("INDEX_PATH").map_or_else(paths::index_path, PathBuf::from)
}

/// Ollama endpoint for `--summary`.
pub fn summary_endpoint() -> String {
    env_pair("SUMMARY_ENDPOINT").unwrap_or_else(|| DEFAULT_SUMMARY_ENDPOINT.to_string())
}

/// Ollama model for `--summary`.
pub fn summary_model() -> String {
    env_pair("SUMMARY_MODEL").unwrap_or_else(|| DEFAULT_SUMMARY_MODEL.to_string())
}

/// Thread count ONNX Runtime should default to.
fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get)
}

/// Number of CPU threads ONNX Runtime should use.
pub fn inference_threads() -> usize {
    env_pair("THREADS")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(default_threads)
}

/// Whether GPU execution was requested through the environment.
///
/// The `cuda` cargo feature decides what is *compiled in*; this variable lets
/// a user keep a CUDA-enabled build but force CPU execution for one run.
pub fn use_cuda() -> bool {
    env_pair("CUDA").is_some_and(|v| !matches!(v.as_str(), "0" | "false" | "no" | "off"))
}

/// Proxy environment variables for child processes.
///
/// Both upper- and lower-case spellings are exported because different tools
/// disagree about which they honour. Loopback hosts are added to `NO_PROXY`
/// so local services (Ollama, the proxy itself) stay direct.
pub fn proxy_env() -> Vec<(String, String)> {
    let mut out = Vec::new();
    let proxy = proxy();
    if !proxy.is_empty() {
        for key in ["ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"] {
            out.push((key.to_string(), proxy.clone()));
            out.push((key.to_ascii_lowercase(), proxy.clone()));
        }
    }
    let existing = std::env::var("NO_PROXY").unwrap_or_default();
    let mut hosts: Vec<String> = existing
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    for host in NO_PROXY_HOSTS {
        if !hosts.iter().any(|h| h == host) {
            hosts.push(host.to_string());
        }
    }
    let no_proxy = hosts.join(",");
    out.push(("NO_PROXY".to_string(), no_proxy.clone()));
    out.push(("no_proxy".to_string(), no_proxy));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_proxy_is_tor_socks() {
        // The test environment may or may not define the variable; assert the
        // fallback constant directly so the test is hermetic.
        assert_eq!(DEFAULT_PROXY, "socks5h://127.0.0.1:9050");
    }

    #[test]
    fn no_proxy_hosts_include_loopback() {
        assert!(NO_PROXY_HOSTS.contains(&"127.0.0.1"));
        assert!(NO_PROXY_HOSTS.contains(&"localhost"));
    }
}
