# 07. Route every outbound request through the SOCKS5 proxy

- **Status:** Accepted
- **Date:** 2026-11-09
- **Deciders:** @Githab-capibara
- **Related:** [api/02-configuration.md](../api/02-configuration.md), [design/01-pipeline-stages.md](../design/01-pipeline-stages.md)

## Context

`lc-rs` talks to the network in three places: it downloads media with
`yt-dlp`, it downloads the model checkpoints from Hugging Face, and it can
post a transcript to a local Ollama instance for summarisation.

Two of those three are third-party endpoints. The tool's promise is that the
*audio* never leaves the machine — but the download itself still exposes the
user's IP address to the platform, and a plain HTTP client behind a
corporate network is frequently blocked.

The Python implementation already shipped a proxy-first policy: every
outbound request went through a SOCKS5 proxy, defaulting to a local Tor
instance at `127.0.0.1:9050`, with `--no-proxy` as the escape hatch. It
also supported rotating the Tor circuit before a download, which defeats the
"same exit IP for every request" pattern that bot-checks look for.

`reqwest` supports SOCKS5 through its `socks` feature, and the Tor control
port is a plain TCP protocol, so the policy can be implemented without an
additional dependency.

## Decision

Every outbound request goes through the SOCKS5 proxy configured by
`LOCALCAPTION_PROXY` / `LCRS_PROXY`, defaulting to
`socks5h://127.0.0.1:9050`:

- the media download is performed by `yt-dlp`, which is spawned with
  `ALL_PROXY`, `HTTP_PROXY`, `HTTPS_PROXY` (and their lower-case spellings)
  exported into its environment;
- the checkpoint download is performed by `reqwest`, whose client is built
  with a SOCKS5 proxy;
- `--no-proxy` sets the proxy to the empty string, which disables it for
  that run;
- `--rotate-tor` asks the Tor control port for a new circuit before the
  download, so the exit address changes;
- loopback hosts (`localhost`, `127.0.0.1`, `::1`) are always added to
  `NO_PROXY`, so the local Ollama instance and the proxy itself stay
  direct.

The proxy URL is validated before it is used, and only `http`, `https`,
`socks5` and `socks5h` schemes are accepted.

## Consequences

- **Easier:** the tool works from networks that block direct egress, and
  the default configuration needs no user action on a machine that already
  runs Tor.
- **Easier:** one policy covers all three network paths — media,
  checkpoints and summaries — instead of three independent configurations.
- **Easier:** the escape hatch is explicit and per-run: `--no-proxy`.
- **Harder:** every request pays the proxy's latency, and over Tor the
  download of a 3 GB checkpoint is slow. The downloader is resumable and
  retries, which mitigates this.
- **Harder:** a misconfigured proxy produces a connection error rather than
  a silent direct connection — which is the intended behaviour, but it does
  mean the tool fails where a proxy-less tool would have worked.
- **Harder:** the local Ollama instance needs the loopback exclusion to
  work at all when a proxy is configured.
- **Given up:** a transparent default of "direct connection". The tool
  always tries the proxy first.
- **Migration:** a machine that does not run Tor either starts one or passes
  `--no-proxy` / `LCRS_PROXY=""`.

## Alternatives considered

- **Direct connections by default, with a proxy opt-in.** Rejected: it is
  the opposite of the reference behaviour and it would expose the user's
  address by default.
- **Route only the media downloads through the proxy.** Rejected: the
  checkpoint downloads are the largest transfers and the most likely to be
  blocked.
- **Hard-code the Tor endpoint.** Rejected: `LOCALCAPTION_PROXY` already
  exists in the reference implementation and covers other SOCKS5 setups.
