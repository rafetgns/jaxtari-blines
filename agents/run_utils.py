def plan_chunks(total_steps: int, steps_per_update: int, scan_steps: int):
    """Split an env-step budget into whole jitted chunks.

    Returns (scan_steps, num_chunks, steps_per_chunk). Only whole chunks are run,
    so every agent sees exactly num_chunks * steps_per_chunk <= total_steps env steps.
    """
    num_updates = int(total_steps) // steps_per_update  # YAML may give e.g. 1e7 as a float
    if num_updates == 0:
        raise ValueError(f"TOTAL_TIMESTEPS={total_steps} is smaller than one update ({steps_per_update} steps)")
    scan_steps = min(scan_steps, num_updates)
    num_chunks = num_updates // scan_steps
    return scan_steps, num_chunks, scan_steps * steps_per_update


def log_model_artifact(model_path: str, run_name: str):
    """Upload a saved checkpoint to the active wandb run as a model artifact."""
    import wandb

    artifact = wandb.Artifact(name=f"model-{run_name}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print(f"model uploaded to wandb as artifact model-{run_name}")


def eval_mod_configs(config: dict):
    """(mods, label) pairs to evaluate on: the default env plus each eval mod."""
    eval_mods = list(config["EVAL_MODS"] or config["TRAIN_MODS"] or [])
    mod_configs = [([], "default")]
    for mod in eval_mods:
        mods = list(mod) if isinstance(mod, (list, tuple)) else [mod]
        label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods)
        mod_configs.append((mods, label))
    return mod_configs


def game_states(states):
    """Walk wrapper states down to the game's own state (for rendering eval videos).

    Every jaxtari wrapper state (PixelState, ObjectCentricState, LogState, ...)
    keeps its inner state as `atari_state`, down to AtariWrapper's AtariState,
    whose `env_state` is the game state; the depth depends on the wrapper stack.
    """
    from jaxtari.wrappers import AtariState

    while not isinstance(states, AtariState):
        states = states.atari_state
    return states.env_state


def rm_metrics(infos) -> dict:
    """Charts for the RM side of the last finished episodes; {} for runs without a reward machine.

    `infos` are the stacked step infos of a chunk; like charts/avg_episodic_return,
    the values are taken at the chunk's last step and averaged over the envs.
    """
    import numpy as np

    if "returned_episode_rm_return" not in infos:
        return {}
    last = {k: np.asarray(v[-1]) for k, v in infos.items() if k.startswith("returned_episode_rm_")}
    num_states = last["returned_episode_rm_state_returns"].shape[-1]
    state_returns = last["returned_episode_rm_state_returns"].reshape(-1, num_states).mean(axis=0)
    state_steps = last["returned_episode_rm_state_steps"].reshape(-1, num_states).mean(axis=0)
    metrics = {
        "charts/avg_episodic_rm_return": float(last["returned_episode_rm_return"].mean()),
        "charts/avg_episodic_rm_shaping": float(last["returned_episode_rm_shaping"].mean()),
    }
    for u in range(num_states):
        metrics[f"rm/return_u{u}"] = float(state_returns[u])
        metrics[f"rm/steps_u{u}"] = float(state_steps[u])
    return metrics


def _git_commit(path: str) -> str | None:
    """HEAD of the git repo containing *path*, suffixed with '-dirty' on uncommitted changes."""
    import subprocess

    try:
        git = ["git", "-C", path]
        commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return commit + ("-dirty" if dirty else "")


def code_versions() -> dict:
    """Commits of jaxtari-blines and the installed jaxtari, for logging with each run.

    jaxtari is resolved from its import location (editable checkout); for a
    non-editable git install the commit recorded by the installer is used instead.
    """
    import json
    import os
    from importlib import metadata

    import jaxtari

    jaxtari_commit = _git_commit(os.path.dirname(jaxtari.__file__))
    if jaxtari_commit is None:
        try:
            direct_url = json.loads(metadata.distribution("jaxtari").read_text("direct_url.json") or "{}")
            jaxtari_commit = direct_url.get("vcs_info", {}).get("commit_id")
        except metadata.PackageNotFoundError:
            pass
    return {
        "BLINES_COMMIT": _git_commit(os.path.dirname(os.path.abspath(__file__))),
        "JAXTARI_COMMIT": jaxtari_commit,
    }
