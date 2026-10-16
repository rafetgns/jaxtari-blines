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
