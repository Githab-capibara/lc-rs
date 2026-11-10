//! Command-line interface.
//!
//! The surface mirrors `src/localcaption/cli.py` of the Python
//! implementation one-to-one:
//!
//! ```text
//! lc-rs <url-or-file> [options]    transcribe a video (default)
//! lc-rs --batch FILE [options]     transcribe a list of URLs/files
//! lc-rs doctor [--fix]             diagnose your install
//! lc-rs model <subcommand>         list / download / remove models
//! lc-rs search <term>              search past transcripts
//! ```
//!
//! The subcommand is dispatched by hand (instead of through clap's own
//! subcommand machinery) because the transcription command is the *default*:
//! `lc-rs <url>` and `lc-rs transcribe <url>` must behave identically.

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use crate::batch;
use crate::config;
use crate::doctor;
use crate::models;
use crate::pipeline::{self, PipelineOptions};
use crate::playlist;

/// The transcription command (also the default command).
///
/// The flag set is deliberately flat and one-to-one with the Python CLI, so
/// the `bool` count is not reduced further.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Parser)]
#[command(
    name = "lc-rs",
    version,
    about = "Fully-local video → transcript using yt-dlp + ffmpeg + local language-ID and ASR models.",
    disable_help_subcommand = true
)]
struct TranscribeArgs {
    /// `YouTube` URL, any URL yt-dlp supports, or path to a local video/audio file.
    #[arg(value_name = "URL")]
    url: Option<String>,

    /// Transcribe every non-empty, non-# line in FILE sequentially.
    #[arg(long, value_name = "FILE")]
    batch: Option<PathBuf>,

    /// Force a specific ASR model key (skip language routing); see `lc-rs model list`.
    #[arg(long, value_name = "KEY")]
    model: Option<String>,

    /// Output directory for transcript files (default: ./transcripts).
    #[arg(short = 'o', long, default_value = "transcripts", value_name = "DIR")]
    out: PathBuf,

    /// ISO code or language name, or 'auto' to detect (default: auto).
    #[arg(short = 'l', long, default_value = "auto", value_name = "LANG")]
    language: String,

    /// Output format(s). Default 'md' = a single Markdown file. 'all'/'full'/'complete' emits every format.
    #[arg(
        long,
        default_value = "md",
        value_name = "FORMAT",
        value_parser = ["md", "txt", "srt", "vtt", "json", "all", "full", "complete"]
    )]
    output_format: String,

    /// Keep the downloaded audio and intermediate WAV under <out>/.work/.
    #[arg(long)]
    keep_audio: bool,

    /// Do not echo the transcript to stdout when finished.
    #[arg(long)]
    no_print: bool,

    /// yt-dlp cookie source for `YouTube`: a browser name (firefox/chrome/edge/brave/vivaldi) or a Netscape cookie file.
    ///
    /// Passed to yt-dlp as `--cookies-from-browser` / `--cookies`.
    /// Auto-detected if omitted (`LCRS_YTDLP_COOKIES`, then
    /// `~/.localcaption/yt-dlp-cookies.txt`, then a logged-in browser).
    /// Useful when `YouTube` raises a `Sign in to confirm you're not a bot`
    /// block on a Tor exit IP.
    #[arg(long, value_name = "BROWSER_OR_FILE")]
    cookies: Option<String>,

    /// Bypass the configured proxy for this run (direct connection).
    ///
    /// Useful when the Tor exit is flagged but your direct line is clean.
    #[arg(long)]
    no_proxy: bool,

    /// Request a fresh Tor exit circuit before downloading.
    ///
    /// Automatically done when a bot-check block is detected; use this to
    /// force it. Requires the Tor control port (default 9051).
    #[arg(long)]
    rotate_tor: bool,

    /// Only transcribe the first video in a playlist URL; ignore the rest.
    ///
    /// By default a URL with `&list=` transcribes every video in the list
    /// (sequentially, with skip-if-already-done).
    #[arg(long)]
    no_playlist: bool,

    /// Only transcribe the first N videos of a playlist URL.
    #[arg(long, value_name = "N")]
    playlist_limit: Option<usize>,

    /// If a required model isn't installed, download it without asking.
    #[arg(long)]
    auto_download: bool,

    /// Write a Markdown summary via a local Ollama instance (<http://localhost:11434>).
    #[arg(long)]
    summary: bool,

    /// Ollama model for --summary.
    #[arg(long, default_value = config::DEFAULT_SUMMARY_MODEL, value_name = "MODEL")]
    summary_model: String,

    /// Path to a prompt template for --summary (default: built-in).
    #[arg(long, value_name = "FILE")]
    summary_prompt: Option<PathBuf>,
}

/// `lc-rs doctor` arguments.
#[derive(Debug, Parser)]
#[command(
    name = "lc-rs doctor",
    about = "Diagnose a local lc-rs install: external tools, proxy, model runtimes, and checkpoints. Use --fix to attempt repair."
)]
struct DoctorArgs {
    /// Attempt to install missing system tools and download missing models.
    #[arg(long)]
    fix: bool,
}

/// `lc-rs model` arguments.
#[derive(Debug, Parser)]
#[command(name = "lc-rs model", about = "Manage the local model checkpoints.")]
struct ModelArgs {
    /// The model subcommand.
    #[command(subcommand)]
    command: Option<ModelCommand>,
}

/// The `model` subcommands.
#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// List every supported model + install status.
    List,
    /// Download one model (or --all).
    Download {
        /// Model key.
        key: Option<String>,
        /// Download every registered model.
        #[arg(long)]
        all: bool,
        /// Re-download even if the model is already present.
        #[arg(long)]
        force: bool,
        /// Skip the confirmation prompt for --all.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Remove an installed model.
    #[command(alias = "remove", alias = "delete")]
    Rm {
        /// Model key.
        key: String,
        /// Skip the confirmation prompt.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Show details about one model.
    Info {
        /// Model key (e.g. parakeet-tdt-0.6b-v3).
        key: String,
    },
}

/// `lc-rs search` arguments.
#[derive(Debug, Parser)]
#[command(
    name = "lc-rs search",
    about = "Search past transcripts. Ranked by hit count; timestamps come from the transcript JSON/SRT when present."
)]
struct SearchArgs {
    /// Search term (case-insensitive substring).
    #[arg(value_name = "TERM", required = true)]
    term: Vec<String>,
}

/// Run the CLI and return the process exit code.
pub fn run(args: &[String]) -> i32 {
    init_tracing();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("lc-rs: could not start the async runtime: {err}");
            return 1;
        }
    };
    runtime.block_on(dispatch(args))
}

/// Install the `tracing` subscriber.
///
/// Without it every `tracing::warn!` and `tracing::info!` in the crate is
/// discarded, and the user never learns that — say — the summary was
/// skipped or that the search index could not be updated.
///
/// The filter comes from `RUST_LOG` and defaults to `info`. The output goes
/// to stderr, so that stdout carries only the artefacts: the transcript and
/// the reports the user asked for.
fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // `try_init` fails when a subscriber is already installed, which happens
    // in the tests; that is not an error.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_ansi(false)
        .without_time()
        .compact()
        .try_init();
}

/// Dispatch on the first argument.
async fn dispatch(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("doctor") => cmd_doctor(&args[1..]).await,
        Some("model") => cmd_model(&args[1..]).await,
        Some("search") => cmd_search(&args[1..]),
        Some("transcribe") => cmd_transcribe(&args[1..]).await,
        Some("help" | "--help-all") => {
            print_top_level_help();
            0
        }
        _ => cmd_transcribe(args).await,
    }
}

/// The top-level usage banner.
fn print_top_level_help() {
    print!(
        "usage: lc-rs <url-or-file> [options]    transcribe a video (default)\n\
         \x20      lc-rs --batch FILE [options]     transcribe a list of URLs/files\n\
         \x20      lc-rs doctor [--fix]             diagnose your install\n\
         \x20      lc-rs model <subcommand>         list / download / remove models\n\
         \x20      lc-rs search <term>              search past transcripts\n\
         \x20      lc-rs --help                     show transcribe help\n\
         \x20      lc-rs --version                  print version\n\
         \n\
         Run `lc-rs <subcommand> --help` for details on each.\n"
    );
}

/// Run the transcription command.
async fn cmd_transcribe(rest: &[String]) -> i32 {
    let args = match TranscribeArgs::try_parse_from(
        std::iter::once("lc-rs".to_string()).chain(rest.iter().cloned()),
    ) {
        Ok(parsed) => parsed,
        Err(err) => {
            let code = if err.use_stderr() { 2 } else { 0 };
            let _ = err.print();
            return code;
        }
    };

    let has_batch = args.batch.is_some();
    let has_url = args.url.is_some();
    if has_batch == has_url {
        let message = if has_batch {
            "provide a URL/file or --batch FILE, not both"
        } else {
            "provide a URL/file or --batch FILE"
        };
        eprintln!("lc-rs: error: {message}");
        return 2;
    }

    // An unknown model key is rejected before anything is downloaded: the
    // reference implementation names the valid keys and stops.
    if let Some(forced) = args.model.as_deref()
        && let Err(err) = models::get_model(forced)
    {
        eprintln!("lc-rs: {err}");
        return 1;
    }

    let required = required_models(&args.language, args.model.as_deref());
    if !ensure_models(&required, args.auto_download).await {
        return 1;
    }

    if let Some(batch_file) = &args.batch {
        return run_batch(&args, batch_file).await;
    }
    let url = args.url.clone().unwrap_or_default();
    run_one(&args, &url).await
}

/// The model keys a run is going to need.
fn required_models(language: &str, force_model: Option<&str>) -> Vec<String> {
    if let Some(forced) = force_model {
        return vec![forced.to_string()];
    }
    let code = pipeline::normalize_language(language);
    if code == "auto" {
        // The optional variants are not required by the default routing.
        return models::known_models()
            .iter()
            .filter(|spec| !spec.optional)
            .map(|spec| spec.key.to_string())
            .collect();
    }
    vec![crate::ml::asr::model_for_language(&code).to_string()]
}

/// Make sure the required checkpoints are installed.
async fn ensure_models(keys: &[String], auto: bool) -> bool {
    let missing = models::missing_models(Some(keys));
    if missing.is_empty() {
        return true;
    }

    println!("Missing model checkpoints:");
    for key in &missing {
        let Ok(spec) = models::get_model(key) else {
            continue;
        };
        println!(
            "  - {}: {} (~{})",
            spec.key,
            spec.description,
            doctor::format_size_mb(spec.approx_size_mb)
        );
    }

    let proceed = if auto {
        true
    } else if !std::io::stdin().is_terminal() {
        eprintln!(
            "Cannot prompt to download models (stdin is not a TTY).\n  \
             Run: lc-rs model download {}\n  \
             Or pass --auto-download.",
            missing[0]
        );
        return false;
    } else {
        print!("  Download now? [Y/n] ");
        let _ = std::io::stdout().flush();
        let mut reply = String::new();
        match std::io::stdin().lock().read_line(&mut reply) {
            Ok(0) | Err(_) => {
                println!("\nCancelled.");
                return false;
            }
            Ok(_) => {
                let reply = reply.trim().to_lowercase();
                matches!(reply.as_str(), "" | "y" | "yes")
            }
        }
    };

    if !proceed {
        println!("  Skipped. Run: lc-rs model download {}", missing[0]);
        return false;
    }

    for key in &missing {
        if let Err(err) = models::download_model(key, false).await {
            eprintln!("lc-rs: {err}");
            return false;
        }
    }
    true
}

/// Build the pipeline options from the command-line arguments.
fn pipeline_options(args: &TranscribeArgs) -> PipelineOptions {
    PipelineOptions {
        out_dir: args.out.clone(),
        language: args.language.clone(),
        force_model: args.model.clone(),
        keep_intermediate: args.keep_audio,
        stem: None,
        summary: args.summary,
        summary_model: args.summary_model.clone(),
        summary_prompt: args.summary_prompt.clone(),
        output_format: args.output_format.clone(),
        cookies: args.cookies.clone(),
        proxy: if args.no_proxy {
            Some(String::new())
        } else {
            None
        },
        force_rotate: args.rotate_tor,
    }
}

/// Transcribe a single source.
async fn run_one(args: &TranscribeArgs, url: &str) -> i32 {
    // A URL with &list= is a playlist. By default every video in the list is
    // transcribed (sequentially, skip-if-already-done). --no-playlist drops
    // back to the single-video behaviour.
    if !args.no_playlist && playlist::has_playlist_query(url) {
        return run_playlist(args, &[url.to_string()]).await;
    }

    let options = pipeline_options(args);
    let result = match pipeline::transcribe_url(url, &options).await {
        Ok(result) => result,
        Err(err) => {
            eprintln!("lc-rs: {err}");
            return 1;
        }
    };

    println!("transcript files:");
    for (kind, path) in result.transcripts.existing() {
        println!("  {kind:>4}: {}", path.display());
    }
    if let Some(path) = &result.summary {
        println!("  summary: {}", path.display());
    }
    if let Some(path) = &result.chapters_json {
        println!("  chapters: {}", path.display());
    }
    if let Some(path) = &result.chaptered_md {
        println!("  chaptered: {}", path.display());
    }

    if !args.no_print
        && let Some(txt) = &result.transcripts.txt
        && txt.is_file()
    {
        let body = std::fs::read_to_string(txt).unwrap_or_default();
        println!("\n{}\u{2500} transcript \u{2500}", "\u{2500}".repeat(30));
        print!("{body}");
    }
    0
}

/// Expand playlist URLs and transcribe every entry.
async fn run_playlist(args: &TranscribeArgs, urls: &[String]) -> i32 {
    if let Err(err) = crate::paths::ensure_dir(&args.out) {
        eprintln!("lc-rs: {err}");
        return 1;
    }

    let mut expanded: Vec<String> = Vec::new();
    for source in urls {
        match playlist::expand_playlist(
            source,
            if args.no_proxy { Some("") } else { None },
            args.cookies.as_deref(),
        )
        .await
        {
            Ok(info) => expanded.extend(info.urls),
            Err(err) => {
                eprintln!("lc-rs: {err}");
                return 1;
            }
        }
    }

    let total = expanded.len();
    if let Some(limit) = args.playlist_limit
        && limit > 0
        && limit < total
    {
        tracing::info!("playlist: transcribing first {limit} of {total} videos (--playlist-limit)");
        expanded.truncate(limit);
    }

    tracing::info!(
        "playlist: {} video(s) to transcribe -> {}/<id>/<id>.md  (skip-if-already-done)",
        expanded.len(),
        args.out.display()
    );

    let options = pipeline_options(args);
    match batch::transcribe_urls(&expanded, &options).await {
        Ok(result) => {
            println!("{}", result.summary());
            result.exit_code()
        }
        Err(err) => {
            eprintln!("lc-rs: {err}");
            1
        }
    }
}

/// Transcribe every source of a batch list.
async fn run_batch(args: &TranscribeArgs, batch_file: &Path) -> i32 {
    if !batch_file.is_file() {
        eprintln!("lc-rs: batch file not found: {}", batch_file.display());
        return 1;
    }
    let urls = match batch::read_url_list(batch_file) {
        Ok(urls) => urls,
        Err(err) => {
            eprintln!("lc-rs: cannot read batch file: {err}");
            return 1;
        }
    };
    if urls.is_empty() {
        eprintln!("lc-rs: no URLs in {}", batch_file.display());
        return 1;
    }

    let options = pipeline_options(args);
    match batch::transcribe_urls(&urls, &options).await {
        Ok(result) => {
            println!("{}", result.summary());
            result.exit_code()
        }
        Err(err) => {
            eprintln!("lc-rs: {err}");
            1
        }
    }
}

/// Run the `doctor` command.
///
/// The whole command is async, exactly like `model` and `transcribe`: it runs
/// on the runtime the entry point started, so it must never build a second
/// one (blocking inside a running runtime panics).
async fn cmd_doctor(rest: &[String]) -> i32 {
    let args = match DoctorArgs::try_parse_from(
        std::iter::once("lc-rs doctor".to_string()).chain(rest.iter().cloned()),
    ) {
        Ok(parsed) => parsed,
        Err(err) => {
            let code = if err.use_stderr() { 2 } else { 0 };
            let _ = err.print();
            return code;
        }
    };

    println!("lc-rs {}\n", crate::VERSION);

    let diagnostics = doctor::run().await;
    doctor::print_diagnostics(&diagnostics);

    if diagnostics.all_ok() {
        println!("\nAll checks passed. You're good to go: lc-rs <url-or-file>");
        return 0;
    }

    if !args.fix {
        if diagnostics.fix_hints.is_empty() {
            println!("Some checks failed.");
        } else {
            println!("\nHow to fix:\n");
            for hint in &diagnostics.fix_hints {
                for line in hint.lines() {
                    println!("  {line}");
                }
                println!();
            }
            println!("Some checks failed. See 'How to fix' above, or re-run with --fix.");
        }
        return 1;
    }

    if doctor::apply_fixes(&diagnostics.gaps).await.is_err() {
        println!("\nFix aborted. Address the error above and re-run.");
        return 1;
    }

    println!(
        "\n{}\nRe-running diagnostics to verify…\n",
        "\u{2500}".repeat(60)
    );
    let after = doctor::run().await;
    doctor::print_diagnostics(&after);
    if after.all_ok() {
        println!("\nAll checks passed. You're good to go: lc-rs <url-or-file>");
        return 0;
    }
    println!("\nSome checks still failing — see output above.");
    1
}

/// Run the `model` command family.
async fn cmd_model(rest: &[String]) -> i32 {
    let args = match ModelArgs::try_parse_from(
        std::iter::once("lc-rs model".to_string()).chain(rest.iter().cloned()),
    ) {
        Ok(parsed) => parsed,
        Err(err) => {
            let code = if err.use_stderr() { 2 } else { 0 };
            let _ = err.print();
            return code;
        }
    };

    match args.command {
        None => {
            print_model_usage();
            2
        }
        Some(ModelCommand::List) => model_list(),
        Some(ModelCommand::Download {
            key,
            all,
            force,
            yes: _,
        }) => model_download(key, all, force).await,
        Some(ModelCommand::Rm { key, yes }) => model_rm(&key, yes),
        Some(ModelCommand::Info { key }) => model_info(&key),
    }
}

/// The `model` usage banner.
fn print_model_usage() {
    print!(
        "usage: lc-rs model <subcommand> [options]\n\n\
         subcommands:\n\
         \x20 list                list every supported model + install status\n\
         \x20 download <key>      download one model (or --all)\n\
         \x20 rm <key>            remove an installed model\n\
         \x20 info <key>          show details about one model\n"
    );
}

/// `lc-rs model list`.
fn model_list() -> i32 {
    println!("Models used by lc-rs:\n");
    println!("  {:<32}{:<14}{:>10}   Status", "Key", "Role", "Size");
    println!(
        "  {:<32}{:<14}{:>10}   {}",
        "-".repeat(32),
        "-".repeat(14),
        "-".repeat(10),
        "-".repeat(11)
    );
    for row in models::list_status() {
        let size = doctor::format_size_mb(row.spec.approx_size_mb);
        let status = if row.installed {
            "✅ installed"
        } else if row.spec.optional {
            "optional — not installed"
        } else {
            "not installed"
        };
        println!(
            "  {:<32}{:<14}{:>10}   {status}",
            row.spec.key, row.spec.role, size
        );
    }
    println!("\nInstall location: {}", models::models_root().display());
    println!("\nTips:");
    println!("  • Russian audio → Qwen3-ASR, English → Parakeet, other → Nemotron.");
    println!("  • To download one:  lc-rs model download qwen3-asr-0.6b");
    println!("  • To download all:  lc-rs model download --all");
    0
}

/// `lc-rs model info`.
fn model_info(key: &str) -> i32 {
    let spec = match models::get_model(key) {
        Ok(spec) => spec,
        Err(err) => {
            eprintln!("lc-rs: {err}");
            return 2;
        }
    };
    let missing = models::missing_files(spec.key).unwrap_or_default();
    println!("Model:        {}", spec.key);
    println!("Role:         {}", spec.role);
    println!("Description:  {}", spec.description);
    println!(
        "Approx size:  {}",
        doctor::format_size_mb(spec.approx_size_mb)
    );
    println!(
        "Execution:    {}",
        if cfg!(feature = "cuda") {
            "ONNX Runtime (CUDA support compiled in)"
        } else {
            "ONNX Runtime (CPU)"
        }
    );
    if spec.is_embedded() {
        // The graph is compiled into the binary, so there is nothing to
        // download and the on-disk copy is optional.
        println!("Source:       embedded in the binary");
        println!(
            "Origin:       https://huggingface.co/{} (exported to ONNX)",
            spec.hf_repo
        );
    } else {
        println!("Source:       https://huggingface.co/{}", spec.hf_repo);
    }
    println!("Local path:   {}", spec.local_dir().display());
    if missing.is_empty() {
        let total: u64 = spec
            .files
            .iter()
            .map(|file| {
                std::fs::metadata(spec.local_dir().join(file.name)).map_or(0, |meta| meta.len())
            })
            .sum();
        if spec.is_embedded() {
            println!("Installed:    yes (compiled in, no download needed)");
        } else {
            println!(
                "Installed:    yes ({} on disk)",
                doctor::format_size_mb(total / (1024 * 1024))
            );
        }
    } else {
        println!("Installed:    no");
        let names: Vec<&str> = missing.iter().map(|file| file.name).collect();
        println!("Missing:      {}", names.join(", "));
        println!("\nDownload with:  lc-rs model download {}", spec.key);
    }
    0
}

/// `lc-rs model download`.
async fn model_download(key: Option<String>, all: bool, force: bool) -> i32 {
    let specs: Vec<&'static models::ModelSpec> = if all {
        // `--all` fetches the models the default routing needs. An optional
        // variant is a deliberate choice, never a default.
        models::known_models()
            .iter()
            .filter(|spec| !spec.optional)
            .collect()
    } else if let Some(key) = key {
        match models::get_model(&key) {
            Ok(spec) => vec![spec],
            Err(err) => {
                eprintln!("lc-rs: {err}");
                return 2;
            }
        }
    } else {
        eprintln!("lc-rs: error: provide a model key or --all");
        return 2;
    };

    let total: u64 = specs.iter().map(|spec| spec.approx_size_mb).sum();
    println!(
        "▸ Downloading {} model(s), ~{} total",
        specs.len(),
        doctor::format_size_mb(total)
    );
    println!("  via proxy {}", config::proxy());

    for spec in specs {
        match models::download_model(spec.key, force).await {
            Ok(outcome) => {
                println!("✅ {} → {}", spec.key, outcome.dir.display());
            }
            Err(err) => {
                eprintln!("\nlc-rs: {err}");
                return 1;
            }
        }
    }
    0
}

/// `lc-rs model rm`.
fn model_rm(key: &str, yes: bool) -> i32 {
    let spec = match models::get_model(key) {
        Ok(spec) => spec,
        Err(err) => {
            eprintln!("lc-rs: {err}");
            return 2;
        }
    };
    let dir = spec.local_dir();
    if !dir.exists() {
        eprintln!(
            "lc-rs: model {key:?} is not installed at {}\n  Run `lc-rs model list` to see installed models.",
            dir.display()
        );
        return 1;
    }

    println!("About to remove: {}", dir.display());
    if !yes {
        print!("Continue? [y/N] ");
        let _ = std::io::stdout().flush();
        let mut reply = String::new();
        match std::io::stdin().lock().read_line(&mut reply) {
            Ok(0) | Err(_) => {
                eprintln!("\nCancelled.");
                return 130;
            }
            Ok(_) => {
                let reply = reply.trim().to_lowercase();
                if !matches!(reply.as_str(), "y" | "yes") {
                    println!("Cancelled.");
                    return 0;
                }
            }
        }
    }

    match models::remove_model(key) {
        Ok(_) => {
            println!("✓ Removed.");
            0
        }
        Err(err) => {
            eprintln!("lc-rs: {err}");
            1
        }
    }
}

/// `lc-rs search`.
fn cmd_search(rest: &[String]) -> i32 {
    let args = match SearchArgs::try_parse_from(
        std::iter::once("lc-rs search".to_string()).chain(rest.iter().cloned()),
    ) {
        Ok(parsed) => parsed,
        Err(err) => {
            let code = if err.use_stderr() { 2 } else { 0 };
            let _ = err.print();
            return code;
        }
    };

    let term = args.term.join(" ");
    let index_path = config::index_path();
    let hits = match crate::search::search(&index_path, &term) {
        Ok(hits) => hits,
        Err(err) => {
            eprintln!("lc-rs: could not read search index: {err}");
            return 1;
        }
    };
    if hits.is_empty() {
        println!("No matches for {term:?} in {}", index_path.display());
        return 1;
    }

    for hit in hits {
        let timestamp = hit
            .start
            .map_or_else(|| "--:--".to_string(), crate::chapters::format_timestamp);
        println!("{}  {}  {}", hit.id, timestamp, hit.text);
        if !hit.title.is_empty() {
            println!("        {}", hit.title);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_transcribe_flags() {
        let args = TranscribeArgs::try_parse_from([
            "lc-rs",
            "https://example.com/v",
            "--output-format",
            "all",
            "-o",
            "/tmp/out",
            "--no-print",
            "--rotate-tor",
        ])
        .unwrap();
        assert_eq!(args.url.as_deref(), Some("https://example.com/v"));
        assert_eq!(args.output_format, "all");
        assert_eq!(args.out, PathBuf::from("/tmp/out"));
        assert!(args.no_print);
        assert!(args.rotate_tor);
        assert!(!args.no_proxy);
    }

    #[test]
    fn rejects_unknown_output_formats() {
        assert!(TranscribeArgs::try_parse_from(["lc-rs", "x", "--output-format", "pdf"]).is_err());
    }

    #[test]
    fn computes_the_required_models() {
        assert_eq!(required_models("ru", None), vec!["qwen3-asr-0.6b"]);
        assert_eq!(required_models("en", None), vec!["parakeet-tdt-0.6b-v3"]);
        assert_eq!(
            required_models("de", None),
            vec!["nemotron-3.5-asr-streaming-0.6b"]
        );
        assert_eq!(required_models("auto", None).len(), 4);
        assert_eq!(
            required_models("auto", Some("parakeet-tdt-0.6b-v3")),
            vec!["parakeet-tdt-0.6b-v3"]
        );
    }
}
