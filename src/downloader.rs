//! Stage 1: download the best available audio stream with `yt-dlp`.
//!
//! The Python implementation drove `yt-dlp` through its Python API. This
//! port drives the *command-line* interface instead: the binary is spawned
//! with an absolute path, no shell is involved, and the resulting metadata is
//! read back from the JSON the tool prints after the file has been moved into
//! place.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;

use crate::config;
use crate::errors::{Error, Result};
use crate::net;
use crate::paths;
use crate::proc::{self, ProcessOutput};

/// `YouTube` periodically blocks specific player clients. The mobile clients
/// tend to be the most stable for audio-only retrieval; the order matters
/// because `yt-dlp` tries them left to right.
pub const DEFAULT_PLAYER_CLIENTS: [&str; 5] = ["tv", "mweb", "android", "ios", "web"];

/// Hard timeout for a single `yt-dlp` invocation.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_hours(3);

/// Hard timeout for a metadata-only playlist expansion.
const METADATA_TIMEOUT: Duration = Duration::from_mins(10);

/// Downloaded audio plus the `yt-dlp` metadata object.
#[derive(Debug, Clone)]
pub struct DownloadResult {
    /// Path of the downloaded media file.
    pub path: PathBuf,
    /// Metadata (`id`, `title`, `chapters`, `webpage_url`, …).
    pub info: Value,
}

/// Build an HTTP client that routes through the configured proxy.
pub fn http_client() -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .user_agent(concat!("lc-rs/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_hours(1));
    let proxy = config::proxy();
    if !proxy.is_empty() {
        let proxy = reqwest::Proxy::all(&proxy)
            .map_err(|e| Error::Usage(format!("invalid proxy URL {proxy:?}: {e}")))?;
        builder = builder.proxy(proxy).no_proxy();
    }
    builder.build().map_err(|e| Error::Download(e.to_string()))
}

/// Build an HTTP client that never uses a proxy (for loopback services).
pub fn direct_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(concat!("lc-rs/", env!("CARGO_PKG_VERSION")))
        .no_proxy()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|e| Error::Download(e.to_string()))
}

/// Absolute path of the `yt-dlp` executable.
pub fn yt_dlp_path() -> Result<PathBuf> {
    proc::require_executable(
        "yt-dlp",
        "Install it with `pipx install yt-dlp` or `pip install yt-dlp`.",
    )
}

/// Cookie source for `yt-dlp`: a Netscape jar path or a browser name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CookieSource {
    /// A Netscape-format cookie file.
    File(PathBuf),
    /// Cookies extracted from a locally installed browser.
    Browser(String),
}

impl CookieSource {
    /// Classify a raw `--cookies` value.
    ///
    /// A bare browser name becomes [`CookieSource::Browser`]; anything with a
    /// path separator or a file extension is treated as a file path. This
    /// mirrors the Python implementation and avoids `yt-dlp` silently
    /// writing a cookie jar named after a browser into the working directory.
    pub fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();
        let lower = trimmed.to_lowercase();
        const BROWSERS: [&str; 12] = [
            "chrome",
            "chromium",
            "edge",
            "firefox",
            "brave",
            "bravebrowser",
            "vivaldi",
            "opera",
            "safari",
            "konqueror",
            "ie",
            "silk",
        ];
        let has_separator = trimmed.contains('/') || trimmed.contains(std::path::MAIN_SEPARATOR);
        let has_extension = Path::new(trimmed)
            .extension()
            .is_some_and(|e| !e.is_empty());
        if BROWSERS.contains(&lower.as_str()) && !has_separator && !has_extension {
            return CookieSource::Browser(lower);
        }
        CookieSource::File(expand_tilde(trimmed))
    }
}

/// Expand a leading `~` in a path.
fn expand_tilde(raw: &str) -> PathBuf {
    if let Some(rest) = raw.strip_prefix('~')
        && (rest.is_empty() || rest.starts_with('/'))
        && let Some(home) = std::env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest.trim_start_matches('/'));
    }
    PathBuf::from(raw)
}

/// Auto-detected cookie source, if any.
pub fn detect_cookies() -> Option<CookieSource> {
    if let Some(explicit) = config::ytdlp_cookies() {
        return Some(CookieSource::parse(&explicit));
    }
    let default = expand_tilde("~/.localcaption/yt-dlp-cookies.txt");
    if default.is_file() {
        return Some(CookieSource::File(default));
    }
    let browsers = config::ytdlp_cookie_browsers().unwrap_or_else(|| "chrome,firefox".to_string());
    for name in browsers.split(',') {
        let name = name.trim().to_lowercase();
        if name.is_empty() {
            continue;
        }
        if browser_profile_exists(&name) {
            return Some(CookieSource::Browser(name));
        }
    }
    None
}

/// Whether a browser with the given name has a local cookie database.
///
/// The *profile directory* is not enough: a browser can leave an empty
/// profile behind, and `yt-dlp --cookies-from-browser` then fails with
/// "could not find chrome cookies database", which aborts the download. The
/// cookie database itself is what matters.
///
/// Every name is mapped to exactly the directories `yt-dlp` searches for
/// *that* browser. Mixing them up would report cookies that the tool
/// cannot find: a Chromium profile lives in `~/.config/chromium`, while
/// `yt-dlp --cookies-from-browser chrome` only looks in
/// `~/.config/google-chrome`.
fn browser_profile_exists(name: &str) -> bool {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return false;
    };
    browser_profile_exists_in(&home, name)
}

/// The lookup, with the home directory passed in, so that it can be tested.
fn browser_profile_exists_in(home: &Path, name: &str) -> bool {
    let candidates: Vec<PathBuf> = match name {
        "chrome" => vec![
            home.join(".config/google-chrome"),
            home.join("Library/Application Support/Google/Chrome"),
        ],
        "chromium" => vec![home.join(".config/chromium")],
        "brave" | "bravebrowser" => vec![
            home.join(".config/BraveSoftware/Brave-Browser"),
            home.join("Library/Application Support/BraveSoftware/Brave-Browser"),
        ],
        "edge" => vec![
            home.join(".config/microsoft-edge"),
            home.join("Library/Application Support/Microsoft Edge"),
        ],
        "firefox" => vec![
            home.join(".mozilla/firefox"),
            home.join("Library/Application Support/Firefox"),
        ],
        "vivaldi" => vec![
            home.join(".config/vivaldi"),
            home.join("Library/Application Support/Vivaldi"),
        ],
        "opera" => vec![
            home.join(".config/opera"),
            home.join("Library/Application Support/com.operasoftware.Opera"),
        ],
        "safari" => vec![home.join("Library/Cookies")],
        _ => Vec::new(),
    };
    candidates.iter().any(|root| profile_has_cookie_db(root))
}

/// Whether a browser profile tree holds a cookie database.
///
/// The Chromium family stores it as `<profile>/Cookies`, Firefox as
/// `<profile>/cookies.sqlite`, and Safari as a single
/// `Cookies.binarycookies` file outside the profile tree.
fn profile_has_cookie_db(root: &Path) -> bool {
    if root.join("Cookies.binarycookies").is_file() {
        return true;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            let name = path
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            if name == "Cookies" || name == "cookies.sqlite" {
                return true;
            }
            continue;
        }
        if !path.is_dir() {
            continue;
        }
        // A browser profile lives one level below the root: `Default/Cookies`.
        for file in ["Cookies", "cookies.sqlite"] {
            if path.join(file).is_file() {
                return true;
            }
        }
    }
    false
}

/// Validate a media URL or local path.
///
/// Local paths are accepted as-is; remote URLs must parse and use a scheme
/// `yt-dlp` can handle.
pub fn validate_source(source: &str) -> Result<()> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return Err(Error::Usage("empty source".to_string()));
    }
    if trimmed.chars().any(char::is_control) {
        return Err(Error::Usage(
            "source contains control characters".to_string(),
        ));
    }
    if let Some((scheme, _)) = trimmed.split_once("://") {
        let parsed = url::Url::parse(trimmed)
            .map_err(|e| Error::Usage(format!("invalid URL {trimmed:?}: {e}")))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(Error::Usage(format!(
                "unsupported URL scheme {:?} (only http and https are allowed)",
                parsed.scheme()
            )));
        }
        if parsed.host_str().is_none() {
            return Err(Error::Usage(format!("URL {trimmed:?} has no host")));
        }
        if scheme.is_empty() {
            return Err(Error::Usage("URL has an empty scheme".to_string()));
        }
    }
    Ok(())
}

/// Sequence of player-client lists used when retrying a blocked download.
///
/// Each later attempt drops the client tried first on the previous attempt —
/// the one most likely to be the one `YouTube` flagged.
pub fn client_attempts<'a>(clients: &[&'a str]) -> Vec<Vec<&'a str>> {
    let mut attempts = Vec::new();
    let mut current: Vec<&str> = clients.to_vec();
    while !current.is_empty() {
        attempts.push(current.clone());
        current.remove(0);
    }
    attempts
}

/// Download the best audio stream for `url` into `work_dir`.
///
/// `proxy` overrides the configured proxy; an empty string disables it.
/// `force_rotate` requests a fresh Tor exit circuit up front.
pub async fn download_audio(
    url: &str,
    work_dir: &Path,
    cookies: Option<&str>,
    proxy: Option<&str>,
    force_rotate: bool,
) -> Result<DownloadResult> {
    validate_source(url)?;
    paths::ensure_dir(work_dir)?;

    let program = yt_dlp_path()?;
    let resolved_proxy = match proxy {
        Some(value) => value.to_string(),
        None => config::proxy(),
    };
    let cookie_source = match cookies {
        Some(raw) => Some(CookieSource::parse(raw)),
        None => detect_cookies(),
    };

    if resolved_proxy.is_empty() {
        tracing::info!("yt-dlp: downloading without a proxy (direct connection)");
    }

    let attempts = client_attempts(&DEFAULT_PLAYER_CLIENTS);
    // A flagged Tor exit IP survives every player-client variation, so keep
    // retrying — rotating the exit each time — well past the client list.
    let max_attempts = attempts.len().max(10);

    for attempt in 1..=max_attempts {
        let clients = attempts[(attempt - 1) % attempts.len()].clone();
        if (attempt == 1 || force_rotate) && net::rotate_tor_circuit().await {
            tracing::info!("tor: rotated to a fresh exit circuit");
        }
        tracing::info!(
            "yt-dlp: downloading bestaudio via {} (clients: {})",
            if resolved_proxy.is_empty() {
                "direct"
            } else {
                &resolved_proxy
            },
            clients.join(", ")
        );

        let args = build_download_args(
            url,
            work_dir,
            &resolved_proxy,
            cookie_source.as_ref(),
            &clients,
        );
        let output = run_ytdlp(&program, &args, DOWNLOAD_TIMEOUT).await;

        match output {
            Ok(result) => {
                if let Some(path) = resolve_downloaded_path(&result.stdout, work_dir) {
                    let info = parse_info(&result.stdout).unwrap_or(Value::Null);
                    tracing::info!("downloaded audio: {}", path.display());
                    return Ok(DownloadResult { path, info });
                }
                tracing::warn!(
                    "yt-dlp finished but no audio file was found in {}",
                    work_dir.display()
                );
            }
            Err(err) => {
                let message = err.to_string();
                let blocked = net::looks_like_bot_block(&message);
                if attempt < max_attempts {
                    if blocked {
                        tracing::info!(
                            "download blocked (bot check) — rotating Tor exit and retrying"
                        );
                        if net::rotate_tor_circuit().await {
                            tracing::info!("tor: rotated to a fresh exit circuit for retry");
                        }
                    } else {
                        tracing::warn!("download attempt {attempt} failed ({message}); retrying");
                    }
                    continue;
                }
                if blocked {
                    return Err(Error::Download(
                        "yt-dlp was blocked by YouTube ('Sign in to confirm you're not a bot'). \
                         The Tor exit IP was flagged. Options: (1) drop a Netscape-format cookie \
                         jar at ~/.localcaption/yt-dlp-cookies.txt (or set LOCALCAPTION_YTDLP_COOKIES), \
                         (2) retry to grab a fresh Tor exit, (3) point LOCALCAPTION_PROXY at a \
                         non-Tor exit that YouTube does not flag."
                            .to_string(),
                    ));
                }
                return Err(Error::Download(format!("yt-dlp failed: {message}")));
            }
        }
    }

    Err(Error::Download(
        "yt-dlp finished but no audio file was found".to_string(),
    ))
}

/// Build the argument vector for one `yt-dlp` invocation.
fn build_download_args(
    url: &str,
    work_dir: &Path,
    proxy: &str,
    cookies: Option<&CookieSource>,
    clients: &[&str],
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "--format".to_string(),
        "bestaudio[acodec=opus]/bestaudio/best".to_string(),
        "--output".to_string(),
        work_dir
            .join("%(id)s.%(ext)s")
            .to_string_lossy()
            .into_owned(),
        "--no-playlist".to_string(),
        "--no-warnings".to_string(),
        "--restrict-filenames".to_string(),
        // The Python API option `overwrites` is spelled `--force-overwrites`
        // on the command line; there is no `--overwrites`.
        "--force-overwrites".to_string(),
        "--retries".to_string(),
        "5".to_string(),
        "--fragment-retries".to_string(),
        "5".to_string(),
        "--concurrent-fragments".to_string(),
        "4".to_string(),
        "--extractor-args".to_string(),
        format!("youtube:player_client={}", clients.join(",")),
        // `--print` implies `--quiet`; re-enable the progress meter on stderr
        // so long downloads still show activity.
        "--progress".to_string(),
        "--newline".to_string(),
        "--print".to_string(),
        "after_move:%(.)j".to_string(),
    ];
    if !proxy.is_empty() {
        args.push("--proxy".to_string());
        args.push(proxy.to_string());
    }
    match cookies {
        Some(CookieSource::File(path)) => {
            args.push("--cookies".to_string());
            args.push(path.to_string_lossy().into_owned());
        }
        Some(CookieSource::Browser(name)) => {
            args.push("--cookies-from-browser".to_string());
            args.push(name.clone());
        }
        None => {}
    }
    args.push("--".to_string());
    args.push(url.to_string());
    args
}

/// Spawn `yt-dlp`, streaming its stderr into the log while capturing stdout.
async fn run_ytdlp(program: &Path, args: &[String], timeout: Duration) -> Result<ProcessOutput> {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);

    let mut child = command.spawn().map_err(|e| {
        Error::Dependency(format!(
            "could not start yt-dlp ({}): {e}",
            program.display()
        ))
    })?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let (stderr_tx, mut stderr_rx) = mpsc::channel::<String>(64);
    let stderr_handle = stderr.map(|stream| {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stream).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim_end().to_string();
                if !line.is_empty() {
                    let _ = stderr_tx.send(line).await;
                }
            }
        })
    });

    let stdout_handle = stdout.map(|mut stream| {
        tokio::spawn(async move {
            let mut buffer = Vec::new();
            let _ = stream.read_to_end(&mut buffer).await;
            buffer
        })
    });

    let status = tokio::time::timeout(timeout, child.wait()).await;
    let stderr_task = stderr_handle;
    let stdout_bytes = match stdout_handle {
        Some(handle) => handle.await.unwrap_or_default(),
        None => Vec::new(),
    };
    if let Some(handle) = stderr_task {
        let _ = handle.await;
    }

    match status {
        Ok(Ok(status)) => {
            // Drain any remaining stderr lines.
            while let Ok(line) = stderr_rx.try_recv() {
                tracing::info!("yt-dlp: {line}");
            }
            Ok(ProcessOutput {
                code: status.code(),
                stdout: String::from_utf8_lossy(&stdout_bytes).into_owned(),
                stderr: String::new(),
            })
        }
        Ok(Err(err)) => Err(Error::Download(format!("yt-dlp failed: {err}"))),
        Err(_) => Err(Error::Download(format!(
            "yt-dlp timed out after {}s",
            timeout.as_secs()
        ))),
    }
}

/// Parse the JSON metadata `yt-dlp` printed after moving the file.
fn parse_info(stdout: &str) -> Option<Value> {
    for line in stdout.lines().rev() {
        let trimmed = line.trim();
        if trimmed.starts_with('{')
            && trimmed.ends_with('}')
            && let Ok(value) = serde_json::from_str::<Value>(trimmed)
        {
            return Some(value);
        }
    }
    None
}

/// Determine which file was actually downloaded.
fn resolve_downloaded_path(stdout: &str, work_dir: &Path) -> Option<PathBuf> {
    if let Some(info) = parse_info(stdout) {
        for key in ["filepath", "_filename"] {
            if let Some(value) = info.get(key).and_then(Value::as_str) {
                let path = PathBuf::from(value);
                if path.is_file() {
                    return Some(path);
                }
            }
        }
        let id = info.get("id").and_then(Value::as_str);
        if let Some(id) = id
            && let Some(found) = find_downloaded(work_dir, id)
        {
            return Some(found);
        }
    }
    // Last resort: the newest file in the work directory.
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(work_dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((modified, path))
        })
        .collect();
    candidates.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    candidates.into_iter().map(|(_, path)| path).next()
}

/// Find a downloaded file whose stem starts with `id`.
fn find_downloaded(work_dir: &Path, id: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(work_dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|stem| stem == id || stem.starts_with(id))
        })
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// Run `yt-dlp` in metadata-only mode and return the parsed info object.
pub async fn fetch_metadata(
    url: &str,
    proxy: Option<&str>,
    cookies: Option<&str>,
) -> Result<Value> {
    validate_source(url)?;
    let program = yt_dlp_path()?;
    let resolved_proxy = match proxy {
        Some(value) => value.to_string(),
        None => config::proxy(),
    };
    let cookie_source = match cookies {
        Some(raw) => Some(CookieSource::parse(raw)),
        None => detect_cookies(),
    };

    let mut args: Vec<String> = vec![
        "--skip-download".to_string(),
        "--no-playlist".to_string(),
        "--no-warnings".to_string(),
        "--print".to_string(),
        "%(.)j".to_string(),
    ];
    if !resolved_proxy.is_empty() {
        args.push("--proxy".to_string());
        args.push(resolved_proxy);
    }
    match cookie_source {
        Some(CookieSource::File(path)) => {
            args.push("--cookies".to_string());
            args.push(path.to_string_lossy().into_owned());
        }
        Some(CookieSource::Browser(name)) => {
            args.push("--cookies-from-browser".to_string());
            args.push(name);
        }
        None => {}
    }
    args.push("--".to_string());
    args.push(url.to_string());

    let output = run_ytdlp(&program, &args, METADATA_TIMEOUT).await?;
    if !output.success() {
        return Err(Error::Download(
            "yt-dlp could not read the media metadata".to_string(),
        ));
    }
    parse_info(&output.stdout)
        .ok_or_else(|| Error::Download("yt-dlp returned no metadata".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cookie_source_classifies_browsers() {
        assert_eq!(
            CookieSource::parse("firefox"),
            CookieSource::Browser("firefox".to_string())
        );
        assert_eq!(
            CookieSource::parse("./firefox.txt"),
            CookieSource::File(PathBuf::from("./firefox.txt"))
        );
    }

    #[test]
    fn client_attempts_drop_the_first_client() {
        let attempts = client_attempts(&["tv", "mweb", "android"]);
        assert_eq!(attempts.len(), 3);
        assert_eq!(attempts[0], vec!["tv", "mweb", "android"]);
        assert_eq!(attempts[2], vec!["android"]);
    }

    #[test]
    fn validates_sources() {
        assert!(validate_source("https://example.com/x").is_ok());
        assert!(validate_source("/tmp/audio.mp3").is_ok());
        assert!(validate_source("ftp://example.com/x").is_err());
        assert!(validate_source("").is_err());
        assert!(validate_source("https://").is_err());
    }

    #[test]
    fn download_args_contain_expected_flags() {
        let args = build_download_args(
            "https://example.com/v",
            Path::new("/tmp/work"),
            "socks5h://127.0.0.1:9050",
            None,
            &["tv", "web"],
        );
        assert!(args.contains(&"--proxy".to_string()));
        assert!(args.contains(&"--no-playlist".to_string()));
        assert!(args.iter().any(|a| a == "after_move:%(.)j"));
        assert_eq!(args.last().unwrap(), "https://example.com/v");
    }

    /// Every flag must exist in the installed `yt-dlp`.
    ///
    /// The Python API option `overwrites` has no command-line counterpart
    /// under that name — it is `--force-overwrites` — and the wrong spelling
    /// made *every* download fail with "no such option". The check is
    /// offline: it compares the flags against the tool's own help text.
    #[test]
    fn every_flag_exists_in_the_installed_ytdlp() {
        let Some(program) = crate::proc::find_executable("yt-dlp") else {
            // yt-dlp is not installed; there is nothing to check against.
            return;
        };
        let output = std::process::Command::new(&program)
            .arg("--help")
            .output()
            .expect("yt-dlp --help must run");
        assert!(output.status.success(), "yt-dlp --help failed");
        let help = String::from_utf8_lossy(&output.stdout);

        let args = build_download_args(
            "https://example.com/v",
            Path::new("/tmp/work"),
            "socks5h://127.0.0.1:9050",
            Some(&CookieSource::Browser("chrome".to_string())),
            &["tv", "web"],
        );

        for arg in &args {
            // `--` ends the option list; everything after it is a value.
            if arg == "--" {
                break;
            }
            if !arg.starts_with("--") {
                continue;
            }
            // `--print` takes a template, `--extractor-args` takes a value;
            // only the flag itself has to exist.
            let flag = arg.split_once('=').map_or(arg.as_str(), |(name, _)| name);
            assert!(
                help.contains(&format!(" {flag} ")) || help.contains(&format!("{flag} ")),
                "yt-dlp does not know the flag {flag}"
            );
        }
    }

    /// A browser is only usable when its *own* cookie database exists.
    ///
    /// A Chromium profile in `~/.config/chromium` must not make the tool
    /// claim that Chrome cookies are available: `yt-dlp --cookies-from-browser
    /// chrome` looks in `~/.config/google-chrome` and fails there, which
    /// aborts the whole download.
    #[test]
    fn browser_profiles_are_not_mixed_up() {
        let home = tempfile::tempdir().expect("a temporary home directory");
        let chromium = home.path().join(".config/chromium/Default");
        std::fs::create_dir_all(&chromium).unwrap();
        std::fs::write(chromium.join("Cookies"), b"sqlite").unwrap();

        assert!(browser_profile_exists_in(home.path(), "chromium"));
        assert!(!browser_profile_exists_in(home.path(), "chrome"));
        assert!(!browser_profile_exists_in(home.path(), "firefox"));

        // The same database, now under the Chrome directory.
        let chrome = home.path().join(".config/google-chrome/Default");
        std::fs::create_dir_all(&chrome).unwrap();
        std::fs::write(chrome.join("Cookies"), b"sqlite").unwrap();
        assert!(browser_profile_exists_in(home.path(), "chrome"));
    }

    /// An empty profile directory is not a source of cookies.
    #[test]
    fn an_empty_profile_is_not_a_cookie_source() {
        let home = tempfile::tempdir().expect("a temporary home directory");
        std::fs::create_dir_all(home.path().join(".config/google-chrome")).unwrap();
        assert!(!browser_profile_exists_in(home.path(), "chrome"));

        // Firefox keeps its jar as `cookies.sqlite` inside the profile.
        let firefox = home.path().join(".mozilla/firefox/abc.default");
        std::fs::create_dir_all(&firefox).unwrap();
        std::fs::write(firefox.join("cookies.sqlite"), b"sqlite").unwrap();
        assert!(browser_profile_exists_in(home.path(), "firefox"));
    }
}
