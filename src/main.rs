//! `lc-rs` binary entry point.

#![deny(warnings)]
#![forbid(unsafe_code)]

use std::process::ExitCode;

use lc_rs::cli;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    ExitCode::from(u8::try_from(cli::run(&argv)).unwrap_or(1))
}
