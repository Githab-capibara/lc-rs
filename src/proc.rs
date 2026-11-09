//! Safe external-process helpers.
//!
//! Every child process is spawned through [`std::process::Command`] (never a
//! shell), always with an absolute executable path, and always under a hard
//! timeout so a wedged helper cannot hang the CLI forever.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

use crate::errors::{Error, Result};

/// Locate an executable on `PATH` and return its absolute path.
///
/// The returned path is always absolute so the child process cannot be
/// influenced by the working directory or by `PATH` manipulation between the
/// lookup and the spawn.
pub fn find_executable(name: &str) -> Option<PathBuf> {
    if name.contains('/') {
        let path = PathBuf::from(name);
        return is_executable(&path).then_some(path);
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        let candidate = dir.join(name);
        if is_executable(&candidate) {
            return std::fs::canonicalize(&candidate).ok().or(Some(candidate));
        }
    }
    None
}

/// Whether `path` is a regular file with the executable bit set.
fn is_executable(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Same as [`find_executable`] but fails with an actionable error.
pub fn require_executable(name: &str, hint: &str) -> Result<PathBuf> {
    find_executable(name).ok_or_else(|| {
        Error::Dependency(format!(
            "Required tool '{name}' was not found on PATH. {hint}"
        ))
    })
}

/// Result of running a child process to completion.
#[derive(Debug)]
pub struct ProcessOutput {
    /// Exit code, or `None` when the process was killed by a signal.
    pub code: Option<i32>,
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
}

impl ProcessOutput {
    /// Whether the process exited successfully.
    pub fn success(&self) -> bool {
        self.code == Some(0)
    }
}

/// Apply the proxy policy to a child process.
///
/// `proxy` is the *effective* proxy of the run — the configured one, or the
/// empty string when the run disabled it. The proxy variables are exported
/// so that a tool which honours the environment reaches the same proxy as
/// the tool itself. When the proxy is disabled the variables are *removed*
/// instead, so the child does not inherit an ambient configuration that
/// would contradict the policy — otherwise `--no-proxy` would still reach
/// the network through the proxy, because the environment of the parent
/// process would still carry `HTTP_PROXY`.
///
/// The loopback hosts are excluded through `NO_PROXY`.
pub fn apply_proxy_env(command: &mut Command, proxy: &str) {
    if proxy.is_empty() {
        for key in [
            "ALL_PROXY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "http_proxy",
            "https_proxy",
        ] {
            command.env_remove(key);
        }
    }
    for (key, value) in crate::config::proxy_env_for(proxy) {
        command.env(key, value);
    }
}

/// Run a command with a hard timeout.
///
/// The child is killed when the timeout elapses; the error message then
/// reports the timeout instead of a bare signal.
pub async fn run_with_timeout(
    program: &Path,
    args: &[impl AsRef<OsStr>],
    timeout: Duration,
    label: &str,
) -> Result<ProcessOutput> {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    apply_proxy_env(&mut command, &crate::config::proxy());

    let child = command.spawn().map_err(|e| {
        Error::Dependency(format!(
            "could not start {label} ({}): {e}",
            program.display()
        ))
    })?;

    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => Ok(ProcessOutput {
            code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }),
        Ok(Err(e)) => Err(Error::Dependency(format!("{label} failed: {e}"))),
        Err(_) => Err(Error::Dependency(format!(
            "{label} timed out after {}s",
            timeout.as_secs()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_absolute_path() {
        let path = find_executable("sh").expect("sh is on PATH");
        assert!(path.is_absolute());
    }

    #[test]
    fn missing_executable_is_none() {
        assert!(find_executable("definitely-not-a-real-binary-xyz").is_none());
    }

    #[tokio::test]
    async fn timeout_kills_child() {
        let program = find_executable("sleep").unwrap();
        let result = run_with_timeout(&program, &["30"], Duration::from_millis(200), "sleep").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn captures_output() {
        let program = find_executable("echo").unwrap();
        let result = run_with_timeout(&program, &["hello"], Duration::from_secs(5), "echo")
            .await
            .unwrap();
        assert!(result.success());
        assert_eq!(result.stdout.trim(), "hello");
    }

    /// The child sees the proxy of the tool, and the loopback hosts are
    /// excluded.
    ///
    /// Without the policy a child inherits whatever the ambient
    /// environment holds, which may contradict `--no-proxy` — and then the
    /// download quietly leaves the machine directly.
    #[tokio::test]
    async fn the_configured_proxy_reaches_the_child_environment() {
        let Some(shell) = find_executable("sh") else {
            return;
        };
        let mut command = Command::new(shell);
        command
            .arg("-c")
            .arg("printf '%s' \"${HTTP_PROXY-unset}\"")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        apply_proxy_env(&mut command, "socks5h://127.0.0.1:9050");

        let output = command.output().await.expect("the child must run");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            "socks5h://127.0.0.1:9050"
        );
    }

    /// A disabled proxy is *removed* from the child environment.
    ///
    /// Leaving the ambient `HTTP_PROXY` in place would make `--no-proxy` a
    /// lie for every tool that honours the environment.
    #[tokio::test]
    async fn a_disabled_proxy_is_removed_from_the_child_environment() {
        let Some(shell) = find_executable("sh") else {
            return;
        };
        let mut command = Command::new(shell);
        command
            .arg("-c")
            .arg("printf '%s' \"${HTTP_PROXY-unset}\"")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        apply_proxy_env(&mut command, "");

        let output = command.output().await.expect("the child must run");
        assert_eq!(String::from_utf8_lossy(&output.stdout), "unset");
    }

    /// The loopback hosts stay direct for the child.
    #[tokio::test]
    async fn the_loopback_hosts_are_excluded_for_the_child() {
        let Some(shell) = find_executable("sh") else {
            return;
        };
        let mut command = Command::new(shell);
        command
            .arg("-c")
            .arg("printf '%s' \"${NO_PROXY-unset}\"")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        apply_proxy_env(&mut command, "socks5h://127.0.0.1:9050");

        let output = command.output().await.expect("the child must run");
        let no_proxy = String::from_utf8_lossy(&output.stdout).into_owned();
        for host in ["localhost", "127.0.0.1", "::1"] {
            assert!(
                no_proxy.split(',').any(|h| h.trim() == host),
                "{host} should be excluded through NO_PROXY, got {no_proxy:?}"
            );
        }
    }
}
