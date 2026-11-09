//! `lc-rs doctor` — diagnose a local installation.
//!
//! The Python implementation checked the interpreter, the external tools, the
//! proxy, the per-model virtualenvs and the checkpoints. This port checks the
//! same things, with two differences:
//!
//! * there is no separate model runtime to build — ONNX Runtime is linked
//!   into the binary, so the check reports the runtime version instead,
//! * `--fix` installs the missing system tools through the host's package
//!   manager and downloads the missing checkpoints.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::config;
use crate::downloader;
use crate::errors::{Error, Result};
use crate::models;
use crate::net;
use crate::proc;

/// Timeout for the proxy reachability probe.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Timeout for a package-manager invocation.
const INSTALL_TIMEOUT: Duration = Duration::from_mins(10);

/// One diagnostic line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    /// What was checked.
    pub label: String,
    /// Whether the check passed.
    pub ok: bool,
    /// Extra detail (a path, a version, a hint).
    pub detail: String,
}

/// Gaps that `--fix` can repair.
///
/// One flag per repairable dependency, mirroring `doctor`'s output rows; the
/// lint is opted out because splitting them into sub-structs would obscure the
/// one-to-one mapping with the printed table.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gaps {
    /// `ffmpeg` is missing.
    pub ffmpeg: bool,
    /// `git` is missing.
    pub git: bool,
    /// `curl` is missing.
    pub curl: bool,
    /// `yt-dlp` is missing.
    pub ytdlp: bool,
    /// Some checkpoints are missing.
    pub models: bool,
}

/// The outcome of a diagnostic sweep.
#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    /// Every check, in the order they were run.
    pub checks: Vec<Check>,
    /// Human-readable repair hints.
    pub fix_hints: Vec<String>,
    /// The gaps `--fix` can repair.
    pub gaps: Gaps,
}

impl Diagnostics {
    /// Whether every check passed.
    pub fn all_ok(&self) -> bool {
        self.checks.iter().all(|check| check.ok)
    }
}

/// Render a check line the way the Python implementation did.
fn line(check: &Check) -> String {
    let mark = if check.ok { "✅" } else { "❌" };
    if check.detail.is_empty() {
        format!("  {mark} {}", check.label)
    } else {
        format!("  {mark} {}  ({})", check.label, check.detail)
    }
}

/// Print a diagnostic sweep.
pub fn print_diagnostics(diagnostics: &Diagnostics) {
    for check in &diagnostics.checks {
        if let Some(section) = check.label.strip_prefix('\u{1}') {
            println!("\n{section}:");
            continue;
        }
        println!("{}", line(check));
    }
}

/// Run every diagnostic check.
pub async fn run() -> Diagnostics {
    let mut diagnostics = Diagnostics::default();

    // System tools.
    diagnostics.checks.push(Check {
        label: "\u{1}System tools".to_string(),
        ok: true,
        detail: String::new(),
    });
    diagnostics.checks.push(Check {
        label: "lc-rs".to_string(),
        ok: true,
        detail: crate::VERSION.to_string(),
    });
    for (tool, hint, gap) in [
        (
            "ffmpeg",
            "needed to decode audio — `brew install ffmpeg` or `pacman -S ffmpeg`",
            &mut diagnostics.gaps.ffmpeg,
        ),
        (
            "git",
            "needed by yt-dlp for some extractors",
            &mut diagnostics.gaps.git,
        ),
        (
            "curl",
            "needed to download models",
            &mut diagnostics.gaps.curl,
        ),
    ] {
        let path = proc::find_executable(tool);
        *gap = path.is_none();
        diagnostics.checks.push(Check {
            label: tool.to_string(),
            ok: path.is_some(),
            detail: match &path {
                Some(path) => path.display().to_string(),
                None => format!("missing — {hint}"),
            },
        });
    }

    // Downloader.
    diagnostics.checks.push(Check {
        label: "\u{1}Downloader".to_string(),
        ok: true,
        detail: String::new(),
    });
    match downloader::yt_dlp_path() {
        Ok(path) => diagnostics.checks.push(Check {
            label: "yt-dlp".to_string(),
            ok: true,
            detail: path.display().to_string(),
        }),
        Err(err) => {
            diagnostics.gaps.ytdlp = true;
            diagnostics.checks.push(Check {
                label: "yt-dlp".to_string(),
                ok: false,
                detail: format!("missing — {err}"),
            });
        }
    }

    // Network.
    diagnostics.checks.push(Check {
        label: "\u{1}Network".to_string(),
        ok: true,
        detail: String::new(),
    });
    let proxy = config::proxy();
    let reachable = proxy_reachable(&proxy).await;
    diagnostics.checks.push(Check {
        label: "proxy".to_string(),
        ok: reachable,
        detail: format!(
            "{proxy} ({})",
            if reachable {
                "reachable"
            } else {
                "not reachable"
            }
        ),
    });
    if !reachable {
        diagnostics.fix_hints.push(format!(
            "The SOCKS5 proxy at {proxy} is not reachable.\n\
             Start your proxy (e.g. Tor) or point lc-rs elsewhere:\n    \
             export LCRS_PROXY=socks5h://host:port"
        ));
    }

    // YouTube bot-check mitigations.
    diagnostics.checks.push(Check {
        label: "\u{1}YouTube / Tor bot-check mitigations".to_string(),
        ok: true,
        detail: String::new(),
    });
    let cookies = downloader::detect_cookies();
    diagnostics.checks.push(Check {
        label: "cookies".to_string(),
        ok: cookies.is_some(),
        detail: match &cookies {
            Some(source) => describe_cookie_source(source),
            None => "none — see LCRS_YTDLP_COOKIES / --cookies".to_string(),
        },
    });
    let can_rotate = net::rotate_tor_circuit().await;
    diagnostics.checks.push(Check {
        label: "tor-rotate".to_string(),
        ok: can_rotate,
        detail: if can_rotate {
            "ok (rotated the current circuit)".to_string()
        } else {
            "unavailable — SIGNAL NEWNYM not reachable (Tor control port 9051?)".to_string()
        },
    });
    if cookies.is_none() {
        diagnostics.fix_hints.push(
            "No usable browser cookies were found. yt-dlp needs a logged-in \
             session to get past YouTube's bot check on a Tor exit IP.\n  \
             1. Export a Netscape cookie jar:\n       \
             export LCRS_YTDLP_COOKIES=/path/to/cookies.txt\n  \
             2. Or point at a browser that holds one:\n       \
             export LCRS_YTDLP_COOKIE_BROWSERS=firefox,chrome\n     \
             (the browser is only used when its cookie database exists)\n  \
             3. Or bypass Tor for one run:\n       \
             lc-rs <url> --no-proxy"
                .to_string(),
        );
    }
    if !can_rotate {
        diagnostics.fix_hints.push(
            "Tor circuit rotation is unavailable, so every run leaves the \
             same exit IP.\n  \
             Enable the Tor control port (9051) and set LCRS_TOR_CONTROL, \
             or use --no-proxy."
                .to_string(),
        );
    }

    // Inference runtime.
    diagnostics.checks.push(Check {
        label: "\u{1}Model runtimes".to_string(),
        ok: true,
        detail: String::new(),
    });
    match crate::ml::environment() {
        Ok(_) => diagnostics.checks.push(Check {
            label: "onnxruntime".to_string(),
            ok: true,
            detail: format!(
                "in-process ({})",
                if cfg!(feature = "cuda") {
                    "CUDA support compiled in"
                } else {
                    "CPU"
                }
            ),
        }),
        Err(err) => diagnostics.checks.push(Check {
            label: "onnxruntime".to_string(),
            ok: false,
            detail: err.to_string(),
        }),
    }

    // Checkpoints.
    diagnostics.checks.push(Check {
        label: "\u{1}Models".to_string(),
        ok: true,
        detail: String::new(),
    });
    let statuses = models::list_status();
    let mut any_missing = false;
    for row in &statuses {
        if !row.installed {
            any_missing = true;
        }
        diagnostics.checks.push(Check {
            label: format!(
                "{} (~{})",
                row.spec.key,
                format_size_mb(row.spec.approx_size_mb)
            ),
            ok: row.installed,
            detail: if row.installed {
                row.spec.description.to_string()
            } else {
                format!("missing: {}", row.missing.join(", "))
            },
        });
    }
    if any_missing {
        diagnostics.gaps.models = true;
        diagnostics.fix_hints.push(
            "Some models are missing. To download everything:\n    \
             lc-rs model download --all\n    \
             (or just run: lc-rs doctor --fix)"
                .to_string(),
        );
    }

    diagnostics
}

/// Human-readable description of a cookie source.
fn describe_cookie_source(source: &downloader::CookieSource) -> String {
    match source {
        downloader::CookieSource::File(path) => path.display().to_string(),
        downloader::CookieSource::Browser(name) => format!("browser: {name}"),
    }
}

/// Format a size in megabytes the way the Python implementation did.
pub fn format_size_mb(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f64 / 1024.0)
    } else {
        format!("{mb} MB")
    }
}

/// Best-effort TCP check that the proxy is listening.
async fn proxy_reachable(proxy: &str) -> bool {
    if proxy.is_empty() {
        return false;
    }
    let Ok(parsed) = url::Url::parse(proxy) else {
        return false;
    };
    let Some(host) = parsed.host_str() else {
        return false;
    };
    let port = parsed.port_or_known_default().unwrap_or(0);
    if port == 0 {
        return false;
    }
    matches!(
        tokio::time::timeout(PROBE_TIMEOUT, tokio::net::TcpStream::connect((host, port))).await,
        Ok(Ok(_))
    )
}

/// Try to repair the gaps found by the diagnostic sweep.
pub async fn apply_fixes(gaps: &Gaps) -> Result<()> {
    println!("\nAttempting fixes:\n");

    for (name, gap) in [
        ("ffmpeg", gaps.ffmpeg),
        ("git", gaps.git),
        ("curl", gaps.curl),
    ] {
        if gap {
            println!("▸ Installing system dependency: {name}");
            install_system_dependency(name).await?;
            println!("  ✅ {name} installed");
        }
    }

    if gaps.ytdlp {
        println!("▸ Installing yt-dlp");
        install_ytdlp().await?;
        println!("  ✅ yt-dlp installed");
    }

    if gaps.models {
        println!("▸ Downloading missing models (via the proxy)");
        models::download_all(false).await?;
        println!("  ✅ models ready");
    }

    Ok(())
}

/// Package managers this port knows how to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageManager {
    /// Homebrew (macOS).
    Brew,
    /// APT (Debian, Ubuntu).
    Apt,
    /// DNF (Fedora).
    Dnf,
    /// pacman (Arch).
    Pacman,
}

/// Detect the host's package manager.
fn detect_package_manager() -> Option<(PackageManager, PathBuf)> {
    let candidates: [(PackageManager, &[&str]); 4] = [
        (PackageManager::Brew, &["brew"]),
        (PackageManager::Apt, &["apt-get"]),
        (PackageManager::Dnf, &["dnf"]),
        (PackageManager::Pacman, &["pacman"]),
    ];
    for (manager, names) in candidates {
        if let Some(path) = names.iter().find_map(|name| proc::find_executable(name)) {
            return Some((manager, path));
        }
    }
    None
}

/// Install a system tool through the host's package manager.
async fn install_system_dependency(name: &str) -> Result<()> {
    if proc::find_executable(name).is_some() {
        tracing::info!("{name}: already installed");
        return Ok(());
    }
    let Some((manager, program)) = detect_package_manager() else {
        return Err(Error::Install(format!(
            "No supported package manager found to install {name}.\n\
             Install it manually, then re-run `lc-rs doctor`."
        )));
    };
    let package = match name {
        "ffmpeg" => "ffmpeg",
        "git" => "git",
        "curl" => "curl",
        other => {
            return Err(Error::Install(format!(
                "Do not know how to install {other:?} automatically."
            )));
        }
    };
    // Build the argument vector per package manager. Every entry is an
    // absolute path or a plain flag: no shell is involved.
    let flags: &[&str] = match manager {
        PackageManager::Brew => &["install"],
        PackageManager::Apt | PackageManager::Dnf => &["install", "-y"],
        PackageManager::Pacman => &["-S", "--noconfirm"],
    };
    let mut args: Vec<PathBuf> = Vec::with_capacity(flags.len() + 2);
    args.push(program);
    args.extend(flags.iter().map(PathBuf::from));
    args.push(PathBuf::from(package));

    // apt-get and pacman need root privileges; use sudo when it is available.
    if matches!(manager, PackageManager::Apt | PackageManager::Pacman)
        && let Some(sudo) = proc::find_executable("sudo")
    {
        let mut with_sudo = vec![sudo];
        with_sudo.extend(args);
        args = with_sudo;
    }

    run_installer(&args, name).await
}

/// Run an installer command with a hard timeout.
async fn run_installer(args: &[PathBuf], name: &str) -> Result<()> {
    let program = args[0].clone();
    let rest: Vec<String> = args[1..]
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let output =
        proc::run_with_timeout(&program, &rest, INSTALL_TIMEOUT, &format!("install {name}"))
            .await?;
    if output.success() {
        return Ok(());
    }
    Err(Error::Install(format!(
        "Could not install {name}: {}",
        output.stderr.trim()
    )))
}

/// Install `yt-dlp` with `pipx` or `pip`.
async fn install_ytdlp() -> Result<()> {
    if let Some(pipx) = proc::find_executable("pipx") {
        let output = proc::run_with_timeout(
            &pipx,
            &["install", "yt-dlp"],
            INSTALL_TIMEOUT,
            "install yt-dlp",
        )
        .await?;
        if output.success() {
            return Ok(());
        }
    }
    if let Some(python) = proc::find_executable("python3") {
        let output = proc::run_with_timeout(
            &python,
            &["-m", "pip", "install", "--user", "-U", "yt-dlp"],
            INSTALL_TIMEOUT,
            "install yt-dlp",
        )
        .await?;
        if output.success() {
            return Ok(());
        }
        return Err(Error::Install(format!(
            "Could not install yt-dlp: {}",
            output.stderr.trim()
        )));
    }
    Err(Error::Install(
        "Neither pipx nor python3 was found; install yt-dlp manually.".to_string(),
    ))
}

/// Absolute path of the `ffmpeg` binary, if it is installed.
pub fn ffmpeg_path() -> Option<PathBuf> {
    proc::find_executable("ffmpeg")
}

/// Absolute path of the `yt-dlp` binary, if it is installed.
pub fn ytdlp_path() -> Option<PathBuf> {
    proc::find_executable("yt-dlp")
}

/// The directory the checkpoints are installed in.
pub fn models_root() -> PathBuf {
    models::models_root()
}

/// The location of the search index.
pub fn index_path() -> PathBuf {
    config::index_path()
}

/// A path that exists, for diagnostics.
pub fn exists(path: &Path) -> bool {
    path.exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_sizes() {
        assert_eq!(format_size_mb(84), "84 MB");
        assert_eq!(format_size_mb(2048), "2.0 GB");
    }

    #[test]
    fn line_marks_failures() {
        let check = Check {
            label: "ffmpeg".to_string(),
            ok: false,
            detail: "missing".to_string(),
        };
        assert_eq!(line(&check), "  ❌ ffmpeg  (missing)");
    }
}
