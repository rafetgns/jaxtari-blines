from reward_machines.games.asteroids_rm import AsteroidsRm
from reward_machines.games.enduro_rm import EnduroRm
from reward_machines.games.pong_rm import PongRm
from reward_machines.games.frostbite_rm import FrostbiteRm
from reward_machines.games.freeway_rm import FreewayRm
from reward_machines.games.seaquest_rm import SeaquestRm
from reward_machines.games.phoenix_rm import PhoenixRm
from reward_machines.games.tennis_rm import TennisRm
from reward_machines.games.kangaroo_rm import KangarooRm
from reward_machines.games.beamrider_rm import BeamriderRm
from reward_machines.games.mspacman_rm import MsPacmanRm
from reward_machines.games.breakout_rm import BreakoutRm
from reward_machines.games.gravitar_rm import GravitarRm
from reward_machines.games.montezumarevenge_rm import MontezumaRm
from reward_machines.games.skiing_rm import SkiingRm

GAME_RM_REGISTRY = {
    "pong": PongRm,
    "seaquest": SeaquestRm,
    "frostbite": FrostbiteRm,
    "freeway": FreewayRm,
    "phoenix": PhoenixRm,
    "tennis": TennisRm,
    "kangaroo": KangarooRm,
    "beamrider": BeamriderRm,
    "enduro": EnduroRm,
    "asteroids": AsteroidsRm,
    "mspacman": MsPacmanRm,
    "breakout": BreakoutRm,
    "gravitar": GravitarRm,
    "montezumarevenge": MontezumaRm,
    "skiing": SkiingRm,
}


def rm_settings(config: dict):
    """Reward-machine kwargs for an agent's make_env from the run config, or None.

    REWARD_MACHINE (and RM_SHAPING) apply to object-centric runs only.
    """
    if not config.get("REWARD_MACHINE", False) or config.get("PIXEL_BASED", True):
        return None
    if config["ENV_ID"] not in GAME_RM_REGISTRY:
        raise ValueError(f"no reward machine for '{config['ENV_ID']}'; available: {sorted(GAME_RM_REGISTRY)}")
    return {"use_shaping": bool(config.get("RM_SHAPING", False)), "gamma": float(config.get("GAMMA", 0.99))}


def rm_run_suffix(config: dict) -> str:
    """'_rm' / '_rm-shaped' for run names, '' without a reward machine."""
    settings = rm_settings(config)
    if settings is None:
        return ""
    return "_rm-shaped" if settings["use_shaping"] else "_rm"


def wrap_reward_machine(env, env_id: str, eval: bool = False, **kwargs):
    """Wrap a flattened object-centric env with the game's reward machine."""
    from reward_machines.reward_machine_wrapper import RewardMachineWrapper

    return RewardMachineWrapper(env, GAME_RM_REGISTRY[env_id](), eval=eval, **kwargs)
