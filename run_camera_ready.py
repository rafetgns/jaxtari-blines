#!/usr/bin/env python3
"""
run_camera_ready.py — JAXtari-15 camera-ready runs (see ../evals_to_run.md).

Every (algorithm config, game, seed) is its own process (NUM_SEEDS=1), so seeds
spread over GPUs and a crashed seed does not take the others down. One worker
thread per GPU pulls the next run from a shared queue as soon as its previous
run finishes. GPU occupancy is checked with the guard from run_multiple_exp.py.

Each run writes its stdout/stderr to logs/camera_ready/<config>/<game>_seed<s>.log
and, on success, a .done marker next to it. Re-launching skips runs that have a
marker, so the script can simply be restarted after a crash or for failed runs.

Usage:
    uv run python run_camera_ready.py                 # all GPUs
    uv run python run_camera_ready.py --gpus 0 1 2 3
    uv run python run_camera_ready.py --dry-run       # list pending runs
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

from run_multiple_exp import _parse_gpus, acquire_free_gpu, gpu_occupancy

# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment grid (evals_to_run.md)
# ═══════════════════════════════════════════════════════════════════════════════

GAMES: list[str] = [
    "asteroids", "beamrider", "breakout", "enduro", "freeway",
    "frostbite", "gravitar", "kangaroo", "montezumarevenge", "mspacman",
    "phoenix", "pong", "seaquest", "skiing", "tennis",
]

SEEDS: list[int] = [1, 2, 3, 4, 5]

# (name, configs, alg overrides) in launch order: the upscaled (many-env) setups
# first since they finish fastest; PPO-cleanrl RGB (8 envs, ~2h per run) goes last.
# `name` becomes EXP_NAME, so wandb run names (<game>_<name>_<oc|pixel>_<seed>)
# and model dirs stay apart between setups.
SETUPS: list[tuple[str, list[str], dict]] = [
    ("pqn_tuned", ["pqn_oc_tuned", "pqn_rgb_tuned"], {}),
    ("ppo_tuned", ["ppo_oc_tuned", "ppo_rgb_tuned"], {}),
    ("ppo_cleanrl", ["ppo_oc_cleanrl"], {}),
    ("pqn_paper", ["pqn_oc_paper", "pqn_rgb_paper"], {"TOTAL_TIMESTEPS": 10_000_000}),
    ("ppo_cleanrl", ["ppo_rgb_cleanrl"], {}),
]

COMMON_OVERRIDES: dict = {
    "PROJECT": "blines_camera_ready",
    "WANDB_MODE": "online",
    "WANDB_SAVE_MODEL": True,  # last checkpoint -> wandb artifact
    "VIDEO_MAX_FRAMES": 3000,  # eval video encoding is CPU-bound; 27k-frame episodes took minutes
    "NUM_SEEDS": 1,
}

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs" / "camera_ready"
JAXTARI_DIR = ROOT.parent / "JAXtari"

# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Run:
    setup: str
    config: str
    game: str
    seed: int
    alg_overrides: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def label(self) -> str:
        return f"{self.config}/{self.game}/seed{self.seed}"

    @property
    def log_path(self) -> Path:
        return LOG_DIR / self.config / f"{self.game}_seed{self.seed}.log"

    @property
    def done_path(self) -> Path:
        return self.log_path.with_suffix(".done")

    def command(self) -> list[str]:
        overrides = {**COMMON_OVERRIDES, "ENV_ID": self.game, "SEED": self.seed}
        cmd = [sys.executable, "main.py", f"+alg={self.config}", f"+ALG_CONFIG={self.config}"]
        cmd += [f"{k}={v}" for k, v in overrides.items()]
        cmd += [f"alg.EXP_NAME={self.setup}"]
        cmd += [f"alg.{k}={v}" for k, v in self.alg_overrides.items()]
        cmd += [f"hydra.run.dir=outputs/camera_ready/{self.config}/{self.game}_seed{self.seed}"]
        return cmd


def build_runs() -> list[Run]:
    return [
        Run(setup, config, game, seed, overrides)
        for setup, configs, overrides in SETUPS
        for game in GAMES
        for config in configs
        for seed in SEEDS
    ]


def git_state(path: Path) -> tuple[str, bool]:
    git = ["git", "-C", str(path)]
    commit = subprocess.run([*git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip())
    return commit, dirty


_procs: set[subprocess.Popen] = set()
_procs_lock = threading.Lock()
_stopping = threading.Event()


def run_one(run: Run, gpu_id: int) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WANDB_RUN_GROUP"] = f"{run.config}_{run.game}"
    env["WANDB_TAGS"] = f"{run.setup},{run.config},{run.game}"

    run.log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GPU {gpu_id}] ▶  {run.label}", flush=True)
    t0 = time.perf_counter()
    with open(run.log_path, "w") as log:
        proc = subprocess.Popen(run.command(), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        with _procs_lock:
            _procs.add(proc)
        rc = proc.wait()
        with _procs_lock:
            _procs.discard(proc)

    minutes = (time.perf_counter() - t0) / 60
    if rc == 0:
        run.done_path.write_text(f"gpu={gpu_id} minutes={minutes:.1f}\n")
    icon = "✓" if rc == 0 else f"✗ (rc={rc}, see {run.log_path.relative_to(ROOT)})"
    print(f"[GPU {gpu_id}]    {icon}  {run.label}  [{minutes:.1f}m]", flush=True)
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JAXtari-15 camera-ready grid on all GPUs.")
    parser.add_argument("--gpus", nargs="+", help="GPU IDs (default: all GPUs nvidia-smi reports)")
    parser.add_argument("--dry-run", action="store_true", help="only list the pending runs")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="launch even if JAXtari or jaxtari-blines has uncommitted changes")
    args = parser.parse_args()

    if args.gpus:
        gpu_ids = sorted(set(_parse_gpus(args.gpus)))
    else:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        gpu_ids = [int(line) for line in out.split()]

    all_runs = build_runs()
    pending = [r for r in all_runs if not r.done_path.exists()]

    for name, path in [("JAXtari", JAXTARI_DIR), ("jaxtari-blines", ROOT)]:
        commit, dirty = git_state(path)
        print(f"{name:15s}: {commit}{' (DIRTY)' if dirty else ''}")
        if dirty and not args.allow_dirty and not args.dry_run:
            print(f"\n{name} has uncommitted changes, so the logged commit would not identify the code. "
                  "Commit first or pass --allow-dirty.")
            sys.exit(1)
    print(f"GPUs           : {gpu_ids}")
    print(f"Runs           : {len(pending)} pending / {len(all_runs)} total "
          f"({len(GAMES)} games × {sum(len(c) for _, c, _ in SETUPS)} configs × {len(SEEDS)} seeds)")
    if args.dry_run:
        for r in pending:
            print("   ", r.label, " ".join(r.command()[2:]))
        return
    if not pending:
        print("Nothing to do.")
        return

    status = gpu_occupancy(gpu_ids)
    missing = [g for g in gpu_ids if status[g] == "missing"]
    if missing:
        print(f"GPUs {missing} not found by nvidia-smi.")
        sys.exit(1)

    queue: Queue[Run] = Queue()
    for r in pending:
        queue.put(r)
    failed: list[str] = []
    failed_lock = threading.Lock()
    in_use: set[int] = set()
    guard = threading.Lock()

    def worker() -> None:
        while not _stopping.is_set():
            try:
                run = queue.get_nowait()
            except Empty:
                return
            try:
                gpu_id = acquire_free_gpu(gpu_ids, in_use, guard)
            except RuntimeError as e:
                with failed_lock:
                    failed.append(f"{run.label}: {e}")
                return
            try:
                rc = run_one(run, gpu_id)
            finally:
                with guard:
                    in_use.discard(gpu_id)
            if rc != 0 and not _stopping.is_set():
                with failed_lock:
                    failed.append(f"{run.label} (GPU {gpu_id}, rc={rc})")

    def stop(signum, _frame) -> None:
        print(f"\nsignal {signum}: terminating running jobs ...", flush=True)
        _stopping.set()
        with _procs_lock:
            for p in _procs:
                p.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    threads = [threading.Thread(target=worker, daemon=True) for _ in gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(timeout=1)  # short joins keep the main thread responsive to signals

    print("=" * 60)
    if failed:
        print(f"FAILURES ({len(failed)}/{len(pending)}) — re-launch to retry them:")
        for f in failed:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("Stopped." if _stopping.is_set() else f"All {len(pending)} runs completed successfully.")


if __name__ == "__main__":
    main()
