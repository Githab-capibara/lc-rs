//! GPU memory budget and the ONNX Runtime placement decision.
//!
//! The placement rules this module implements come straight from the way the
//! tool is meant to behave on a workstation that is not dedicated to it:
//!
//! 1. **Try the GPU first.** When the free VRAM covers what a checkpoint
//!    needs, every graph of that checkpoint runs on the card.
//! 2. **A fully busy card sends everything back to the CPU.** Another
//!    process (a local LLM server, a game) may hold the card: a placement
//!    that cannot get warmed up is worse than no placement at all, because
//!    the failed attempt is paid at every session creation.
//! 3. **A card with *some* room but not enough gets a bounded arena.** The
//!    ONNX Runtime arena is capped at the free space (`gpu_mem_limit`), so
//!    the runtime uses the space that exists and whatever cannot be
//!    allocated is executed on the CPU instead.
//!
//! The budget comes from `nvidia-smi`, which is a child process, not a
//! library: the crate keeps its promise of running inference in-process
//! without pulling in NVML bindings. When the budget cannot be measured
//! (no `nvidia-smi`, a driver that does not answer in time) the GPU is
//! still tried, and the retry in [`crate::ml`] is the safety net: a session
//! that cannot be created on the GPU is created on the CPU instead.
//!
//! 4. **A build without the `cuda` feature never asks at all.** See
//!    [`COMPILED`]: the placement it could report is one it could not
//!    execute, and the probe that would decide it is a child process spent
//!    for nothing.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

/// One MiB — the unit `nvidia-smi` reports VRAM in, and the unit the arena
/// limit is expressed in.
pub const MIB: usize = 1 << 20;

/// Free VRAM below this value counts as "the card is busy".
///
/// A few hundred megabytes is less than the cuDNN algorithm-search
/// workspace of one Conformer encoder window, so a placement with less
/// room than this would burn the whole load on a session that cannot
/// survive it. The model goes straight to the CPU instead.
pub const MIN_USEFUL_FREE_MIB: usize = 256;

/// Activations, cuDNN workspace and arena rounding, as a share of the
/// weights: `3 / 2`.
///
/// Weights are copied into VRAM once; the multiplier covers everything the
/// runtime allocates on top of them for a 45 s inference window. It
/// deliberately errs on the generous side: over-estimating only ever costs
/// a placement, never a run.
pub const ACTIVATION_OVERHEAD_NUM: usize = 3;
pub const ACTIVATION_OVERHEAD_DEN: usize = 2;

/// Flat headroom added to every checkpoint, in MiB, so that a small graph
/// is not planned onto a card that has just enough room for its weights.
pub const ACTIVATION_FLOOR_MIB: usize = 256;

/// How long the `nvidia-smi` probe may take before the run gives up on it.
///
/// The call is made on the model load path, which is seconds of work, so a
/// two-second ceiling costs nothing and bounds the worst case (a wedged
/// driver, an NFS-mounted `/usr/bin`).
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Interval between the probe's exit polls.
const PROBE_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// The `nvidia-smi` query, in the order [`GpuMemory`] parses it.
const PROBE_QUERY: &str = "memory.total,memory.used,memory.free";

/// Whether this binary can place a session on the GPU at all.
///
/// The policy below describes what *could* be done with a card; this says
/// whether the binary was built to do it. A build without the `cuda` cargo
/// feature never registers the CUDA execution provider, so reporting a GPU
/// placement for it would be a report of something that cannot happen —
/// `lc-rs doctor` on such a build states the CPU placement instead, and the
/// card is never even probed.
#[cfg(feature = "cuda")]
pub const COMPILED: bool = true;
#[cfg(not(feature = "cuda"))]
pub const COMPILED: bool = false;

/// Where a checkpoint runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// The whole model runs on the CPU.
    Cpu,
    /// The model runs on the GPU.
    ///
    /// `memory_limit_mib` caps the ONNX Runtime device arena; `None` leaves
    /// the arena unbounded, which is correct only when the whole checkpoint
    /// fits in the free VRAM.
    Gpu {
        /// The arena cap in MiB, or `None` for an unbounded arena.
        memory_limit_mib: Option<usize>,
    },
}

impl Placement {
    /// Whether this placement asks for the GPU.
    pub fn on_gpu(&self) -> bool {
        matches!(self, Self::Gpu { .. })
    }
}

/// A card's memory picture, all fields in MiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMemory {
    /// Total VRAM on the card.
    pub total_mib: usize,
    /// VRAM already handed out to every process on the card.
    pub used_mib: usize,
    /// VRAM nobody has claimed yet.
    pub free_mib: usize,
}

impl GpuMemory {
    /// Parse one `nvidia-smi --format=csv,noheader,nounits` line.
    ///
    /// The three columns are the total, the used and the free VRAM, in this
    /// order. Anything that does not carry exactly three non-negative
    /// integers is not a measurement this module may act on, so it is
    /// rejected rather than guessed at.
    pub fn parse(line: &str) -> Option<Self> {
        let columns: Vec<usize> = line
            .trim()
            .split(',')
            .filter_map(|column| column.trim().parse::<usize>().ok())
            .collect();
        match columns.as_slice() {
            [total_mib, used_mib, free_mib] => Some(Self {
                total_mib: *total_mib,
                used_mib: *used_mib,
                free_mib: *free_mib,
            }),
            _ => None,
        }
    }
}

/// The placement decision for one checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// Where the checkpoint is going to run.
    pub placement: Placement,
    /// The free VRAM the decision was made with, in MiB, or `None` when the
    /// budget could not be measured.
    pub free_mib: Option<usize>,
    /// The VRAM the checkpoint is expected to need, in MiB.
    pub required_mib: usize,
}

impl Decision {
    /// VRAM a checkpoint of `model_bytes` bytes is expected to need.
    ///
    /// Weights plus headroom: the weights are copied into VRAM as they are,
    /// and the multiplier covers activations, the cuDNN algorithm-search
    /// workspace and the arena's rounding for a 45 s window.
    pub fn required_mib(model_bytes: u64) -> usize {
        let bytes = usize::try_from(model_bytes).unwrap_or(usize::MAX);
        bytes.div_ceil(MIB) * ACTIVATION_OVERHEAD_NUM / ACTIVATION_OVERHEAD_DEN
            + ACTIVATION_FLOOR_MIB
    }

    /// Apply the placement rules to a measured budget.
    ///
    /// The function is pure — it takes the budget instead of measuring it —
    /// so the three rules can be exercised without a second GPU.
    pub fn for_budget(free_mib: Option<usize>, model_bytes: u64) -> Self {
        let required_mib = Self::required_mib(model_bytes);
        // An unmeasured card is tried: on a machine without `nvidia-smi`
        // (a headless server, a driver that answers late) the GPU is still
        // the best guess, and a failed session creation falls back to the
        // CPU anyway.
        let placement = match free_mib {
            None => Placement::Gpu {
                memory_limit_mib: None,
            },
            Some(free) if free >= required_mib => Placement::Gpu {
                memory_limit_mib: None,
            },
            Some(free) if free >= MIN_USEFUL_FREE_MIB => Placement::Gpu {
                memory_limit_mib: Some(free),
            },
            Some(_) => Placement::Cpu,
        };
        Self {
            placement,
            free_mib,
            required_mib,
        }
    }

    /// The device a run of this checkpoint would use, as `lc-rs model info`
    /// reports it.
    pub fn execution_summary(&self) -> String {
        match self.placement {
            Placement::Cpu => "ONNX Runtime (CPU)".to_string(),
            Placement::Gpu {
                memory_limit_mib: None,
            } => "ONNX Runtime (GPU, full arena)".to_string(),
            Placement::Gpu {
                memory_limit_mib: Some(limit),
            } => format!("ONNX Runtime (GPU, arena capped at {limit} MiB, the rest on the CPU)"),
        }
    }

    /// A one-line explanation for the logs and `lc-rs doctor`.
    pub fn summary(&self) -> String {
        let free = self
            .free_mib
            .map_or_else(|| "unknown".to_string(), |free| format!("{free} MiB"));
        match self.placement {
            Placement::Gpu {
                memory_limit_mib: None,
            } => format!(
                "GPU (unbounded arena: {free} free, {} MiB needed)",
                self.required_mib
            ),
            Placement::Gpu {
                memory_limit_mib: Some(limit),
            } => format!(
                "GPU with a {limit} MiB arena ({free} free, {} MiB needed); \
                 what does not fit runs on the CPU",
                self.required_mib
            ),
            Placement::Cpu => format!(
                "CPU ({free} free, {} MiB needed; {})",
                self.required_mib,
                cpu_reason(self.free_mib)
            ),
        }
    }
}

/// Why a decision landed on the CPU, as one phrase.
///
/// The three answers are different problems with the same symptom — a slow
/// run — and a report that lumped them together would be useless: one is
/// fixed by a build flag, one by rebuilding, one by closing a program.
fn cpu_reason(free_mib: Option<usize>) -> String {
    if !COMPILED {
        return "the build has no CUDA execution provider".to_string();
    }
    match free_mib {
        Some(_) => "the card is too busy to host the model".to_string(),
        None if !crate::config::use_cuda() => "the GPU was disabled through LCRS_CUDA".to_string(),
        None => "the card could not be measured".to_string(),
    }
}

/// Read the card's memory picture.
///
/// Returns `None` when there is no NVIDIA card to measure: `nvidia-smi` is
/// missing, times out, or answers something this module cannot parse.
pub fn probe_memory() -> Option<GpuMemory> {
    probe_memory_with(&find_nvidia_smi()?)
}

/// Locate `nvidia-smi` on this machine.
fn find_nvidia_smi() -> Option<PathBuf> {
    crate::proc::find_executable("nvidia-smi").or_else(|| {
        // `proc::find_executable` searches `PATH`; a driver installed
        // without one (a container that mounts the tools directory by
        // hand) is still worth a look.
        let fallback = Path::new("/usr/bin/nvidia-smi");
        fallback.is_file().then(|| fallback.to_path_buf())
    })
}

/// Run the probe against a specific `nvidia-smi` binary.
///
/// Split out from [`probe_memory`] so the timeout and the parsing can be
/// tested against a stand-in command instead of a real driver.
fn probe_memory_with(command: &Path) -> Option<GpuMemory> {
    let mut child = Command::new(command)
        .args([
            &format!("--query-gpu={PROBE_QUERY}"),
            "--format=csv,noheader,nounits",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    // Bound the wait: a wedged tool must not hold the model load path.
    let mut waited = Duration::ZERO;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if waited < PROBE_TIMEOUT => {
                std::thread::sleep(PROBE_POLL_INTERVAL);
                waited += PROBE_POLL_INTERVAL;
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }

    // `nvidia-smi` answers in a few lines for multi-GPU hosts; the first
    // card is the one the CUDA provider uses by default (`device_id` 0).
    //
    // The pipe is drained by hand instead of through `wait_with_output`:
    // the polling loop above has already reaped the child, and asking for
    // its status twice is an error on some hosts — which would turn a
    // working probe into "the card cannot be measured".
    let mut stdout = Vec::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = std::io::Read::read_to_end(&mut pipe, &mut stdout);
    }
    std::str::from_utf8(&stdout)
        .ok()?
        .lines()
        .find_map(GpuMemory::parse)
}

/// The placement decision for a checkpoint of `model_bytes` bytes on this
/// machine.
///
/// The budget is measured once per call, which is once per session
/// creation, so a batch re-reads the card between models and sees memory
/// the previous model actually left free.
pub fn plan(model_bytes: u64) -> Decision {
    // Nothing a build without the `cuda` feature can do is worth probing
    // the card for: the CPU is the only answer it can give.
    if !COMPILED || !crate::config::use_cuda() {
        return plan_for(None, model_bytes);
    }
    plan_for(free_budget(), model_bytes)
}

/// The free VRAM the policy measures: the forced one when
/// `LCRS_GPU_FREE_MIB` is set, the card's otherwise.
fn free_budget() -> Option<usize> {
    crate::config::gpu_free_mib().or_else(|| probe_memory().map(|memory| memory.free_mib))
}

/// The placement for a budget that has already been measured.
///
/// Split out so that a report can print a whole table of placements against
/// *one* reading of the card — `plan` would re-probe per model, and a card
/// that a run is loading onto would move under the report.
pub fn plan_for(free_mib: Option<usize>, model_bytes: u64) -> Decision {
    // The reading is kept even when the placement is off: a report that
    // shows the card's numbers next to the reason the build cannot use it
    // is more useful than one that says "unknown".
    let required_mib = Decision::required_mib(model_bytes);
    if !COMPILED || !crate::config::use_cuda() {
        return Decision {
            placement: Placement::Cpu,
            free_mib,
            required_mib,
        };
    }
    Decision::for_budget(free_mib, model_bytes)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// The VRAM numbers of the machine this was written on, in MiB.
    const CARD: GpuMemory = GpuMemory {
        total_mib: 4096,
        used_mib: 2916,
        free_mib: 1116,
    };

    #[test]
    fn the_probe_csv_line_is_parsed() {
        let memory = GpuMemory::parse("4096, 2916, 1116").expect("a nvidia-smi line");
        assert_eq!(memory, CARD);
    }

    #[test]
    fn the_probe_csv_line_is_parsed_across_whitespace() {
        let memory = GpuMemory::parse("  4096 , 2916 , 1116  ").expect("a padded line");
        assert_eq!(memory.free_mib, 1116);
    }

    #[test]
    fn a_line_without_three_integers_is_rejected() {
        assert!(
            GpuMemory::parse("").is_none(),
            "an empty answer is no measurement"
        );
        assert!(
            GpuMemory::parse("4096, 2916").is_none(),
            "two columns is no measurement"
        );
        assert!(
            GpuMemory::parse("N/A").is_none(),
            "a card that reports N/A is no measurement"
        );
        assert!(
            GpuMemory::parse("4096, -1, 1116").is_none(),
            "a negative column is no measurement"
        );
    }

    #[test]
    fn a_checkpoint_needs_its_weights_plus_headroom() {
        // 600 MiB of weights plus half of that plus the flat floor.
        assert_eq!(Decision::required_mib(600 * MIB as u64), 600 * 3 / 2 + 256);
        assert_eq!(Decision::required_mib(0), 256);
        // The budget only ever grows with the checkpoint.
        assert!(
            Decision::required_mib(600 * MIB as u64) < Decision::required_mib(601 * MIB as u64)
        );
    }

    #[test]
    fn a_free_card_hosts_the_whole_model() {
        let decision = Decision::for_budget(Some(6000), 600 * MIB as u64);
        assert_eq!(
            decision.placement,
            Placement::Gpu {
                memory_limit_mib: None
            },
            "a card with room for the whole checkpoint gets no arena cap"
        );
    }

    #[test]
    fn the_required_vram_is_enough_and_nothing_less() {
        let required = Decision::required_mib(600 * MIB as u64);
        assert!(
            Decision::for_budget(Some(required), 600 * MIB as u64).placement
                == Placement::Gpu {
                    memory_limit_mib: None
                },
            "exactly the required VRAM still counts as enough"
        );
        assert!(
            Decision::for_budget(Some(required - 1), 600 * MIB as u64).placement
                == Placement::Gpu {
                    memory_limit_mib: Some(required - 1)
                },
            "one MiB short of it caps the arena instead"
        );
    }

    /// The floor is the boundary: just below it the card counts as busy,
    /// just at it the space that exists is still used.
    #[test]
    fn a_fully_busy_card_keeps_the_model_on_the_cpu() {
        let decision = Decision::for_budget(Some(MIN_USEFUL_FREE_MIB - 1), 600 * MIB as u64);
        assert_eq!(
            decision.placement,
            Placement::Cpu,
            "a card with less room than the floor is not worth a placement"
        );
        let at_the_floor = Decision::for_budget(Some(MIN_USEFUL_FREE_MIB), 600 * MIB as u64);
        assert_eq!(
            at_the_floor.placement,
            Placement::Gpu {
                memory_limit_mib: Some(MIN_USEFUL_FREE_MIB)
            },
            "the floor itself is still room the runtime may use"
        );
    }

    /// The room the card actually has on this machine — a 4 GB card with a
    /// local LLM server on it — lands in the capped-arena branch.
    #[test]
    fn a_card_with_partial_room_caps_the_arena() {
        let decision = Decision::for_budget(Some(CARD.free_mib), 600 * MIB as u64);
        assert_eq!(
            decision.placement,
            Placement::Gpu {
                memory_limit_mib: Some(CARD.free_mib)
            },
            "the space that exists is used, and the arena is capped at it"
        );
    }

    /// The floor is what decides "fully busy", not the size of the model:
    /// a small checkpoint on a full card must not be placed either.
    #[test]
    fn a_small_model_on_a_full_card_also_stays_on_the_cpu() {
        let decision = Decision::for_budget(Some(64), 40 * MIB as u64);
        assert_eq!(
            decision.placement,
            Placement::Cpu,
            "the floor does not shrink for a small checkpoint"
        );
    }

    /// An unmeasured card is tried.
    #[test]
    fn an_unmeasured_card_is_tried() {
        let decision = Decision::for_budget(None, 600 * MIB as u64);
        assert!(
            decision.placement.on_gpu(),
            "without a measurement the GPU is still the best guess"
        );
        assert_eq!(
            decision.placement,
            Placement::Gpu {
                memory_limit_mib: None
            }
        );
    }

    #[test]
    fn the_summary_names_the_placement_and_the_budget() {
        let full = Decision::for_budget(Some(6000), 600 * MIB as u64);
        assert!(
            full.summary().contains("unbounded arena"),
            "unexpected summary: {}",
            full.summary()
        );
        let capped = Decision::for_budget(Some(1116), 600 * MIB as u64);
        assert!(
            capped.summary().contains("1116 MiB arena"),
            "unexpected summary: {}",
            capped.summary()
        );
        let cpu = Decision::for_budget(Some(100), 600 * MIB as u64);
        if COMPILED {
            assert!(
                cpu.summary().contains("too busy"),
                "unexpected summary: {}",
                cpu.summary()
            );
        } else {
            assert!(
                cpu.summary().contains("no CUDA execution provider"),
                "a CPU build blames the build, not the card: {}",
                cpu.summary()
            );
        }
    }

    /// The reason a run stayed on the CPU is what tells the user what to
    /// change, so the three causes must not collapse into one phrase.
    #[test]
    fn the_summary_names_why_the_model_is_on_the_cpu() {
        if !COMPILED {
            let impossible = plan_for(None, 600 * MIB as u64);
            assert!(
                impossible.summary().contains("no CUDA execution provider"),
                "a CPU build must not blame the card: {}",
                impossible.summary()
            );
            return;
        }
        let busy = Decision::for_budget(Some(100), 600 * MIB as u64);
        assert!(
            busy.summary().contains("too busy"),
            "a busy card is its own cause: {}",
            busy.summary()
        );
    }

    /// The gate in front of the table: a placement this build cannot
    /// execute is not a placement it may report.
    #[test]
    fn a_placement_the_build_cannot_execute_is_not_reported() {
        let decision = plan_for(Some(6000), 600 * MIB as u64);
        if COMPILED && crate::config::use_cuda() {
            assert_eq!(
                decision.placement,
                Placement::Gpu {
                    memory_limit_mib: None
                },
                "a build with the provider places a checkpoint that fits"
            );
        } else {
            assert_eq!(
                decision.placement,
                Placement::Cpu,
                "a build without the provider reports the CPU"
            );
            assert!(
                decision.summary().contains("no CUDA execution provider"),
                "unexpected summary: {}",
                decision.summary()
            );
        }
    }

    /// Write a stand-in `nvidia-smi` that prints `stdout` and exits.
    fn fake_probe(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, body).expect("the stand-in probe");
        let mut permissions = fs::metadata(&path)
            .expect("the stand-in probe")
            .permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            permissions.set_mode(0o755);
        }
        fs::set_permissions(&path, permissions).expect("the stand-in probe");
        path
    }

    #[cfg(unix)]
    #[test]
    fn the_probe_reads_a_real_command() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let probe = fake_probe(
            dir.path(),
            "nvidia-smi",
            "#!/bin/sh\necho '4096, 2916, 1116'\n",
        );
        let memory = probe_memory_with(&probe).expect("the stand-in answered");
        assert_eq!(memory, CARD);
    }

    #[cfg(unix)]
    #[test]
    fn a_probe_that_writes_no_csv_reports_no_card() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let probe = fake_probe(dir.path(), "nvidia-smi", "#!/bin/sh\nexit 0\n");
        assert!(
            probe_memory_with(&probe).is_none(),
            "a command that answers nothing leaves the budget unmeasured"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_probe_that_never_answers_is_abandoned() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let probe = fake_probe(dir.path(), "nvidia-smi", "#!/bin/sh\nsleep 60\n");
        let started = std::time::Instant::now();
        assert!(
            probe_memory_with(&probe).is_none(),
            "a hanging probe must not be waited out"
        );
        assert!(
            started.elapsed() < PROBE_TIMEOUT * 2,
            "the probe gave up within the timeout: {:?}",
            started.elapsed()
        );
    }

    /// The probe sits on the model load path, so its timeout has to stay
    /// well under the seconds a checkpoint load takes anyway.
    #[test]
    fn the_probe_timeout_stays_short() {
        assert!(
            PROBE_TIMEOUT <= Duration::from_secs(5),
            "the probe gave up within the timeout: {PROBE_TIMEOUT:?}"
        );
    }
}
