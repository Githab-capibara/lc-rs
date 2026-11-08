//! Network policy: every outbound byte rides a SOCKS5 proxy.
//!
//! The default is a local Tor SOCKS5 endpoint at `127.0.0.1:9050`. It is
//! applied to yt-dlp media downloads, model downloads and any other HTTP
//! traffic the tool performs. Loopback hosts are always excluded so local
//! services (Ollama, the proxy itself) stay direct.

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::config;

/// Timeout for the Tor control-port handshake.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(2);

/// Parse `scheme://host:port` into its host and port.
fn parse_host_port(endpoint: &str) -> Option<(String, u16)> {
    let rest = endpoint.split_once("://")?.1;
    let rest = rest.split('/').next()?;
    let (host, port) = rest.rsplit_once(':')?;
    let port = port.parse::<u16>().ok()?;
    if host.is_empty() {
        return None;
    }
    Some((host.trim_matches(['[', ']']).to_string(), port))
}

/// Host and port of the Tor control port.
///
/// `LOCALCAPTION_TOR_CONTROL` wins; otherwise the control port is derived
/// from the SOCKS port (Tor's convention is SOCKS + 1).
fn tor_control_host_port() -> Option<(String, u16)> {
    if let Some(explicit) = config::tor_control() {
        if let Some(parsed) = parse_host_port(&explicit) {
            return Some(parsed);
        }
        if let Some((host, port)) = explicit.rsplit_once(':')
            && let Ok(port) = port.parse::<u16>()
        {
            return Some((host.to_string(), port));
        }
    }
    let proxy = config::proxy();
    let (host, port) = parse_host_port(&proxy)?;
    Some((host, port.saturating_add(1)))
}

/// Read Tor's control-port authentication cookie, if one is available.
fn read_tor_control_cookie() -> Option<Vec<u8>> {
    let mut candidates: Vec<std::path::PathBuf> = vec![
        "/run/tor/control.authcookie".into(),
        "/var/run/tor/control.authcookie".into(),
    ];
    if let Some(uid) = uid() {
        candidates.push(format!("/run/user/{uid}/tor/control.authcookie").into());
    }
    if let Some(dir) = config::tor_data_dir() {
        candidates.push(dir.join("control.authcookie"));
    }
    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path)
            && !bytes.is_empty()
        {
            return Some(bytes);
        }
    }
    None
}

/// Current user id, when the platform exposes it.
fn uid() -> Option<u32> {
    // `/proc/self/status` is the portable-enough way to learn the uid without
    // pulling in the `libc` crate.
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Uid:") {
            let first = rest.split_whitespace().next()?;
            return first.parse::<u32>().ok();
        }
    }
    None
}

/// Ask Tor for a fresh exit circuit (`SIGNAL NEWNYM`).
///
/// Returns `false` when the control port is unreachable — the caller then
/// falls back to cookies or a non-Tor proxy. Only loopback proxies are
/// considered: a remote SOCKS5 proxy has no Tor instance to ask.
pub async fn rotate_tor_circuit() -> bool {
    let proxy = config::proxy();
    if let Some((host, _)) = parse_host_port(&proxy)
        && !matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1")
    {
        return false;
    }

    let Some((host, port)) = tor_control_host_port() else {
        return false;
    };
    let cookie = read_tor_control_cookie();

    let connect = TcpStream::connect((host.as_str(), port));
    let connected = tokio::time::timeout(CONTROL_TIMEOUT, connect).await;
    let Ok(Ok(mut stream)) = connected else {
        return false;
    };

    if let Some(cookie) = cookie {
        let hex = hex::encode(cookie);
        if stream
            .write_all(format!("AUTHENTICATE {hex}\r\n").as_bytes())
            .await
            .is_err()
        {
            return false;
        }
        let mut buf = [0u8; 1024];
        let read = tokio::time::timeout(CONTROL_TIMEOUT, stream.read(&mut buf)).await;
        let Ok(Ok(n)) = read else {
            return false;
        };
        let response = String::from_utf8_lossy(&buf[..n]);
        if !response.contains("250") {
            return false;
        }
    } else {
        // Plaintext / no-auth control port.
        if stream.write_all(b"AUTHENTICATE\r\n").await.is_err() {
            return false;
        }
        let mut buf = [0u8; 1024];
        let _ = tokio::time::timeout(CONTROL_TIMEOUT, stream.read(&mut buf)).await;
    }

    if stream.write_all(b"SIGNAL NEWNYM\r\n").await.is_err() {
        return false;
    }
    let mut buf = [0u8; 1024];
    match tokio::time::timeout(CONTROL_TIMEOUT, stream.read(&mut buf)).await {
        Ok(Ok(n)) => {
            let response = String::from_utf8_lossy(&buf[..n]);
            response.contains("OK")
        }
        _ => false,
    }
}

/// Heuristic: does this yt-dlp error mean `YouTube` flagged the IP or client?
///
/// Only `YouTube`'s specific bot-wall phrases match. A bare `403` is *not* a
/// reliable signal — it fires on unrelated sources and would trigger
/// needless Tor rotations.
pub fn looks_like_bot_block(message: &str) -> bool {
    let lowered = message.to_lowercase();
    [
        "sign in to confirm",
        "confirm you're not a bot",
        "sign in to confirm you’re not a bot",
        "bot check",
        "bot-check",
        "too many requests",
    ]
    .iter()
    .any(|token| lowered.contains(token))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_host_port() {
        assert_eq!(
            parse_host_port("socks5h://127.0.0.1:9050"),
            Some(("127.0.0.1".to_string(), 9050))
        );
        assert_eq!(parse_host_port("nonsense"), None);
    }

    #[test]
    fn detects_bot_block_phrases() {
        assert!(looks_like_bot_block(
            "ERROR: Sign in to confirm you're not a bot"
        ));
        assert!(looks_like_bot_block("HTTP Error 429: Too Many Requests"));
        assert!(!looks_like_bot_block("HTTP Error 403: Forbidden"));
        assert!(!looks_like_bot_block("video unavailable"));
    }
}
