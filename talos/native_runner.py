"""The talos-native runner: a small Cargo binary that runs one nonce of the staged algorithm
without TIG's fuel instrumentation. Rendered into the monorepo workspace the way
tig-binary/scripts/build_so fills in entry_point_template.rs, then built with plain
`cargo build`. Shipped into the C3 job directory, so standard library only.

Everything here follows tig-runtime/src/main.rs and tig-binary/src/entry_point_template.rs at
MONOREPO_REF: the seed, the track parse, the hyperparameter map, the exit codes, and the empty
solution written when the algorithm saved none. Two deliberate differences: no fuel counter
(a watchdog thread exits 87 at the per-track budget instead), and on GPU the default stream
rather than tig-runtime's fuel-check stream, which synchronises after every kernel launch."""
from __future__ import annotations

from pathlib import Path

PACKAGE = "talos-native"
# The toolchain build_so uses at MONOREPO_REF, so native and metered builds share rustc/LLVM.
TOOLCHAIN = "+nightly-2025-02-10"

_CUDARC = ('cudarc = { git = "https://github.com/tig-foundation/cudarc.git", '
           'branch = "runtime-fuel/cudnn-cublas", features = '
           '["cuda-version-from-build-system"] }')

CARGO_TOML = """[package]
name = "talos-native"
version = "0.1.0"
edition = "2021"
publish = false

[[bin]]
name = "talos-native"
path = "src/main.rs"

[dependencies]
anyhow = "1.0.81"
serde_json = { version = "1.0.113" }
tig-algorithms = { path = "../tig-algorithms", features = ["{CHALLENGE}"] }
tig-challenges = { path = "../tig-challenges", features = ["{CHALLENGE}"] }
tig-structs = { path = "../tig-structs" }
tig-utils = { path = "../tig-utils" }
{CUDARC}
"""

_COMMON = r"""use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tig_algorithms::{CHALLENGE}::{ALGORITHM} as algorithm;
use tig_challenges::{CHALLENGE}::*;
use tig_structs::core::BenchmarkSettings;
use tig_utils::dejsonify;

// tig-runtime's exit codes at MONOREPO_REF: 84 the algorithm returned Err (a panic inside
// solve_challenge becomes Err in entry_point), 87 the fuel ran out.
const RUNTIME_ERROR: i32 = 84;
const OUT_OF_FUEL: i32 = 87;

struct Args {
    settings: String,
    rand_hash: String,
    nonce: u64,
    output: PathBuf,
    budget_us: Option<u64>,
    hyperparameters: Option<String>,
    ptx: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let mut next = |what: &str| it.next().ok_or_else(|| anyhow!("missing {}", what));
    let settings = next("SETTINGS")?;
    let rand_hash = next("RAND_HASH")?;
    let nonce = next("NONCE")?.parse()?;
    let output = PathBuf::from(next("OUTPUT")?);
    let (mut budget_us, mut hyperparameters, mut ptx) = (None, None, None);
    while let Ok(flag) = next("flag") {
        let value = next(flag.as_str())?;
        match flag.as_str() {
            "--budget-us" => budget_us = Some(value.parse()?),
            "--hyperparameters" => hyperparameters = Some(value),
            "--ptx" => ptx = Some(PathBuf::from(value)),
            _ => return Err(anyhow!("unknown flag {}", flag)),
        }
    }
    Ok(Args { settings, rand_hash, nonce, output, budget_us, hyperparameters, ptx })
}

fn seed_and_track(args: &Args) -> Result<([u8; 32], Track)> {
    let settings: BenchmarkSettings = dejsonify(&args.settings)?;
    let seed = settings.calc_seed(&args.rand_hash, args.nonce);
    let track_id = if settings.track_id.starts_with('"') && settings.track_id.ends_with('"') {
        settings.track_id.clone()
    } else {
        format!(r#""{}""#, settings.track_id)
    };
    let track = serde_json::from_str(&track_id)
        .map_err(|_| anyhow!("Failed to parse track_id '{}'", settings.track_id))?;
    Ok((seed, track))
}

fn hyperparameters(args: &Args) -> Result<Option<Map<String, Value>>> {
    match &args.hyperparameters {
        Some(s) => Ok(Some(serde_json::from_str(s)?)),
        None => Ok(None),
    }
}

// The shape tig-verifier reads (a "solution" string field), plus the solve time at this save.
fn write_output(path: &Path, nonce: u64, solution: &Solution, solve_us: u64) -> Result<()> {
    let doc = serde_json::json!({
        "nonce": nonce,
        "solution": serde_json::to_string(solution)?,
        "solve_us": solve_us,
    });
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, doc.to_string())?;
    // The watchdog may exit mid-save: a rename never leaves half a document behind.
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// The fuel stand-in: exits 87 once the solve has run for the budget, leaving the last saved
// solution for tig-verifier, as a metered run that runs out of fuel does.
fn watchdog(budget_us: Option<u64>) {
    if let Some(us) = budget_us {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_micros(us));
            std::process::exit(OUT_OF_FUEL);
        });
    }
}

fn main() {
    let result = parse_args().and_then(run);
    if let Err(e) = result {
        eprintln!("Runtime Error: {}", e);
        std::process::exit(RUNTIME_ERROR);
    }
}
"""

_CPU_RUN = r"""
fn run(args: Args) -> Result<()> {
    let (seed, track) = seed_and_track(&args)?;
    let hyperparameters = hyperparameters(&args)?;
    let challenge = Challenge::generate_instance(&seed, &track)?;
    let (output, nonce) = (args.output.clone(), args.nonce);
    let start = Instant::now();
    let save = |solution: &Solution| -> Result<()> {
        write_output(&output, nonce, solution, start.elapsed().as_micros() as u64)
    };
    watchdog(args.budget_us);
    let result = catch_unwind(AssertUnwindSafe(|| {
        algorithm::solve_challenge(&challenge, &save, &hyperparameters)
    }))
    .unwrap_or_else(|_| Err(anyhow!("Panic occurred calling solve_challenge")));
    if !output.exists() {
        save(&Solution::new())?;
    }
    result
}
"""

_GPU_RUN = r"""
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
    runtime::result::device::get_device_prop,
};

fn run(args: Args) -> Result<()> {
    let (seed, track) = seed_and_track(&args)?;
    let hyperparameters = hyperparameters(&args)?;
    let ptx_path = args.ptx.clone().ok_or_else(|| anyhow!("--ptx is required on GPU"))?;
    let ptx = Ptx::from_src(std::fs::read_to_string(&ptx_path)?);
    let ctx = CudaContext::new(0)?;
    ctx.set_blocking_synchronize()?;
    let module = ctx.load_module(ptx)?;
    let stream = ctx.default_stream();
    let prop = get_device_prop(0)?;
    let challenge =
        Challenge::generate_instance(&seed, &track, module.clone(), stream.clone(), &prop)?;
    let initialize_kernel = module.load_function("initialize_kernel")?;
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        stream
            .launch_builder(&initialize_kernel)
            .arg(&(u64::from_be_bytes(seed[8..16].try_into().unwrap())))
            .launch(cfg)?;
    }
    let (output, nonce) = (args.output.clone(), args.nonce);
    let start = Instant::now();
    let save = |solution: &Solution| -> Result<()> {
        stream.synchronize()?; // as tig-runtime's save: the time covers finished kernels
        write_output(&output, nonce, solution, start.elapsed().as_micros() as u64)
    };
    watchdog(args.budget_us);
    let result = catch_unwind(AssertUnwindSafe(|| {
        algorithm::solve_challenge(&challenge, &save, &hyperparameters,
                                   module.clone(), stream.clone(), &prop)
    }))
    .unwrap_or_else(|_| Err(anyhow!("Panic occurred calling solve_challenge")));
    if !output.exists() {
        save(&Solution::new())?;
    }
    result
}
"""


def _fill(text: str, challenge: str, algorithm: str) -> str:
    return text.replace("{CHALLENGE}", challenge).replace("{ALGORITHM}", algorithm)


def render(challenge: str, algorithm: str, is_gpu: bool) -> dict[str, str]:
    """The runner's files, keyed by their path under the monorepo root."""
    main = _fill(_COMMON + (_GPU_RUN if is_gpu else _CPU_RUN), challenge, algorithm)
    cargo = _fill(CARGO_TOML, challenge, algorithm).replace("{CUDARC}",
                                                            _CUDARC if is_gpu else "")
    return {f"{PACKAGE}/Cargo.toml": cargo, f"{PACKAGE}/src/main.rs": main}


def add_workspace_member(cargo_toml: str) -> str:
    """The workspace manifest with the runner listed once in `members`. Idempotent, because the
    local backend's /app volume keeps the manifest from one job to the next."""
    if f'"{PACKAGE}"' in cargo_toml:
        return cargo_toml
    marker = "members = ["
    i = cargo_toml.find(marker)
    if i < 0:
        raise ValueError("workspace Cargo.toml has no `members = [` list")
    i += len(marker)
    return cargo_toml[:i] + f'\n    "{PACKAGE}",' + cargo_toml[i:]


def stage(monorepo: Path, challenge: str, algorithm: str, is_gpu: bool) -> None:
    for rel, text in render(challenge, algorithm, is_gpu).items():
        p = Path(monorepo) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    manifest = Path(monorepo) / "Cargo.toml"
    manifest.write_text(add_workspace_member(manifest.read_text(encoding="utf-8")),
                        encoding="utf-8", newline="\n")
