# ATTRIBUTION
#   PR #32     Mobin Liaghi, "Add DreamerV3 baseline implementation", based on
#              rafarodsa/dreamer-v3-purejax (Rafael Rodriguez-Sanchez): one jitted
#              collect-and-train loop, block-GRU RSSM, one loss and optimiser for the
#              world model, actor and critic, replay value loss, refreshed replay latents.
#              Changed: linen instead of nnx, own replay buffer instead of flashbax
#              internals, the fixes listed below, repo-standard env/eval/logging.
#   dreamerv3  danijar/dreamerv3 (MIT): loss terms, lambda returns, two-hot, return
#              normalisation, LaProp + AGC optimiser, network layout (size12m preset)
#              and default hyperparameters.
#   this repo  agents/ppo/ppo.py: make_env and the chunked training loop.
#              agents/agent57/agent57_eval.py: the recurrent evaluation structure.
#
#   Algorithm: DreamerV3 (Hafner et al., "Mastering diverse control tasks through world
#   models", Nature 2025). Fixed relative to PR #32: images scaled to [-0.5, 0.5] with a
#   linear decoder (was 0-255 targets for a tanh output), reconstruction loss summed over
#   pixels (was averaged), 4 frames per action (was 16), continue head trained on
#   1 - terminal (was gamma * (1 - done), truncations included), seed from SEED, and
#   episode returns from LogWrapper. Deviations from the official code: float32 instead of
#   bfloat16 and the repo's env settings (episodic life during training, no sticky actions).
#
# Replay layout (as in the official code): record t holds obs_t, the reward that led to
# obs_t, is_first/is_last/is_terminal of obs_t, the action taken at obs_t and the posterior
# latent after obs_t (the "replay context" a training sequence starts from). The jaxtari
# wrappers reset in the same step that ends a game, so a finished game is stored as
# [.., last frame repeated with the final reward and is_last, reset frame with is_first].
# The reset record costs one extra policy step but no env step.

import os
import random
import time
from functools import partial
from typing import NamedTuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import jaxtari
from jaxtari import spaces
from jaxtari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)
from rtpt import RTPT

from agents.dreamerv3.dreamerv3_eval import evaluate
from agents.run_utils import eval_mod_configs, plan_chunks


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False, image_size=64, clip_reward=False):
    """Same wrapper stack as the other agents, except: one frame per step (the RSSM is the
    memory, so no frame stack), 64x64 pixels, raw object features (the encoder applies
    symlog instead of NormalizeObservationWrapper) and, by default, unclipped rewards
    (DreamerV3 handles reward scale with symlog two-hot)."""
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")
    clip_reward = clip_reward and not eval

    def thunk():
        env = jaxtari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,  # only active during training
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(image_size, image_size),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=False,
                frame_stack_size=1,
                frame_skip=4,
                max_pooling=True,
                clip_reward=clip_reward,
            )
        else:
            env = FlattenObservationWrapper(
                ObjectCentricWrapper(
                    env,
                    frame_stack_size=1,
                    frame_skip=4,
                    clip_reward=clip_reward,
                )
            )
        env = LogWrapper(env)
        return env
    return thunk


def prepare_pixels(obs):
    """(C, H, W, 1) wrapper frames -> (H, W, C) model input; works with leading batch axes."""
    return jnp.moveaxis(obs[..., 0], -3, -1)


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def unimix_logits(logits, unimix):
    """Mix 1% uniform into a categorical, as in DreamerV3 (latents and policy)."""
    probs = jax.nn.softmax(logits, -1)
    probs = (1 - unimix) * probs + unimix / logits.shape[-1]
    return jnp.log(probs)


def categorical_kl(p_logits, q_logits):
    logp = jax.nn.log_softmax(p_logits, -1)
    logq = jax.nn.log_softmax(q_logits, -1)
    return (jnp.exp(logp) * (logp - logq)).sum(-1)


def categorical_entropy(logits):
    logp = jax.nn.log_softmax(logits, -1)
    return -(jnp.exp(logp) * logp).sum(-1)


class TwoHot(NamedTuple):
    """Symexp two-hot over 255 bins equally spaced in symlog space (official dreamerv3)."""

    bins: jax.Array

    def mean(self, logits):
        probs = jax.nn.softmax(logits, -1)
        m = (self.bins.shape[0] - 1) // 2
        # sum the two halves from the middle out, so large opposite bins cancel accurately
        wavg = probs[..., m] * self.bins[m] + (
            (probs[..., :m] * self.bins[:m])[..., ::-1] + probs[..., m + 1:] * self.bins[m + 1:]
        ).sum(-1)
        return symexp(wavg)

    def loss(self, logits, target):
        x = symlog(target)
        n = self.bins.shape[0]
        below = jnp.clip((self.bins <= x[..., None]).astype(jnp.int32).sum(-1) - 1, 0, n - 1)
        above = jnp.clip(n - (self.bins > x[..., None]).astype(jnp.int32).sum(-1), 0, n - 1)
        equal = below == above
        dist_below = jnp.where(equal, 1.0, jnp.abs(self.bins[below] - x))
        dist_above = jnp.where(equal, 1.0, jnp.abs(self.bins[above] - x))
        total = dist_below + dist_above
        target = (jax.nn.one_hot(below, n) * (dist_above / total)[..., None]
                  + jax.nn.one_hot(above, n) * (dist_below / total)[..., None])
        return -(target * jax.nn.log_softmax(logits, -1)).sum(-1)


def lambda_return(last, term, rew, boot, disc, lam):
    """Official dreamerv3 lambda return over (batch, time); returns (batch, time - 1).

    R_t = r_{t+1} + disc (1 - term_{t+1}) ((1 - l_{t+1}) boot_{t+1} + l_{t+1} R_{t+1}),
    l_{t+1} = lam (1 - last_{t+1}), R_T = boot_T: it bootstraps fully at episode ends.
    """
    live = (1 - term.astype(jnp.float32))[:, 1:] * disc
    cont = (1 - last.astype(jnp.float32))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]

    def step(ret, x):
        interm_t, live_t, cont_t = x
        ret = interm_t + live_t * cont_t * ret
        return ret, ret

    _, rets = jax.lax.scan(step, boot[:, -1], (interm.T, live.T, cont.T), reverse=True)
    return rets.T


# --- networks ------------------------------------------------------------------------

dense_init = nn.initializers.variance_scaling(1.0, "fan_avg", "truncated_normal")


def out_init(scale):
    return nn.initializers.zeros if scale == 0.0 else nn.initializers.variance_scaling(scale, "fan_avg", "truncated_normal")


def norm_act(x):
    return nn.silu(nn.RMSNorm(epsilon=1e-4)(x))


class MLP(nn.Module):
    layers: int
    units: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.layers):
            x = norm_act(nn.Dense(self.units, kernel_init=dense_init)(x))
        return x


class Head(nn.Module):
    layers: int
    units: int
    out_dim: int
    outscale: float

    @nn.compact
    def __call__(self, x):
        x = MLP(self.layers, self.units)(x)
        return nn.Dense(self.out_dim, kernel_init=out_init(self.outscale))(x)


class BlockLinear(nn.Module):
    """Block-diagonal linear layer: `blocks` independent dense layers on equal input slices."""

    units: int
    blocks: int

    @nn.compact
    def __call__(self, x):
        in_per_block = x.shape[-1] // self.blocks
        kernel = self.param(
            "kernel",
            nn.initializers.variance_scaling(1.0, "fan_avg", "truncated_normal", batch_axis=(0,)),
            (self.blocks, in_per_block, self.units // self.blocks),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.units,))
        x = x.reshape(*x.shape[:-1], self.blocks, in_per_block)
        x = jnp.einsum("...gi,gio->...go", x, kernel)
        return x.reshape(*x.shape[:-2], self.units) + bias


class Encoder(nn.Module):
    pixel_based: bool
    depth: int
    mults: tuple
    kernel: int
    layers: int
    units: int

    @nn.compact
    def __call__(self, obs):
        if not self.pixel_based:
            return MLP(self.layers, self.units)(symlog(obs.astype(jnp.float32)))
        batch_shape = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:]).astype(jnp.float32) / 255.0 - 0.5
        for mult in self.mults:
            x = nn.Conv(self.depth * mult, (self.kernel, self.kernel), padding="SAME", kernel_init=dense_init)(x)
            x = nn.max_pool(x, (2, 2), (2, 2))
            x = norm_act(x)
        return x.reshape(*batch_shape, -1)


class Decoder(nn.Module):
    """Predicts obs / 255 for pixels (MSE summed over pixels) and symlog(obs) for vectors."""

    pixel_based: bool
    obs_shape: tuple
    depth: int
    mults: tuple
    kernel: int
    layers: int
    units: int

    @nn.compact
    def __call__(self, feat):
        if not self.pixel_based:
            x = MLP(self.layers, self.units)(feat)
            return nn.Dense(self.obs_shape[-1], kernel_init=dense_init)(x)
        batch_shape = feat.shape[:-1]
        height, width, channels = self.obs_shape
        scale = 2 ** len(self.mults)
        channels_in = self.depth * self.mults[-1]
        x = nn.Dense(height // scale * width // scale * channels_in, kernel_init=dense_init)(feat)
        x = norm_act(x.reshape(-1, height // scale, width // scale, channels_in))
        for mult in reversed(self.mults[:-1]):
            x = x.repeat(2, axis=1).repeat(2, axis=2)
            x = norm_act(nn.Conv(self.depth * mult, (self.kernel, self.kernel), padding="SAME", kernel_init=dense_init)(x))
        x = x.repeat(2, axis=1).repeat(2, axis=2)
        x = nn.Conv(channels, (self.kernel, self.kernel), padding="SAME", kernel_init=dense_init)(x)
        return x.reshape(*batch_shape, height, width, channels) + 0.5


class DreamerV3(nn.Module):
    """RSSM world model with encoder/decoder and reward/continue heads, plus actor and critic.

    The latent state is (deter, stoch): deter (..., DETER) is the block-GRU memory,
    stoch (..., STOCH, CLASSES) a one-hot sample per categorical. Sampling happens outside
    the module, so the same methods serve training (sampled) and evaluation (mode).
    """

    num_actions: int
    obs_shape: tuple
    pixel_based: bool
    deter: int
    hidden: int
    stoch: int
    classes: int
    blocks: int
    depth: int
    units: int
    cnn_mults: tuple
    kernel: int
    enc_layers: int
    head_layers: int
    ac_layers: int
    num_bins: int

    def setup(self):
        self.dyn_in = [nn.Dense(self.hidden, kernel_init=dense_init) for _ in range(3)]
        self.dyn_in_norm = [nn.RMSNorm(epsilon=1e-4) for _ in range(3)]
        self.dyn_hidden = BlockLinear(self.deter, self.blocks)
        self.dyn_hidden_norm = nn.RMSNorm(epsilon=1e-4)
        self.dyn_gates = BlockLinear(3 * self.deter, self.blocks)
        self.prior_net = Head(2, self.hidden, self.stoch * self.classes, 1.0)
        self.post_net = Head(1, self.hidden, self.stoch * self.classes, 1.0)
        self.encoder = Encoder(self.pixel_based, self.depth, self.cnn_mults, self.kernel, self.enc_layers, self.units)
        self.decoder = Decoder(self.pixel_based, self.obs_shape, self.depth, self.cnn_mults, self.kernel,
                               self.enc_layers, self.units)
        self.reward_head = Head(self.head_layers, self.units, self.num_bins, 0.0)
        self.cont_head = Head(self.head_layers, self.units, 1, 1.0)
        self.actor = Head(self.ac_layers, self.units, self.num_actions, 0.01)
        self.critic = Head(self.ac_layers, self.units, self.num_bins, 0.0)

    def _core(self, deter, stoch, action):
        """Block-GRU step (official dreamerv3 RSSM._core)."""
        xs = (deter, stoch.reshape(*stoch.shape[:-2], -1), jax.nn.one_hot(action, self.num_actions))
        x = jnp.concatenate([nn.silu(n(d(x))) for x, d, n in zip(xs, self.dyn_in, self.dyn_in_norm)], -1)
        g = self.blocks
        x = jnp.repeat(x[..., None, :], g, axis=-2)
        x = jnp.concatenate([deter.reshape(*deter.shape[:-1], g, -1), x], -1)
        x = nn.silu(self.dyn_hidden_norm(self.dyn_hidden(x.reshape(*x.shape[:-2], -1))))
        x = self.dyn_gates(x).reshape(*deter.shape[:-1], g, -1)
        reset, cand, update = [y.reshape(*deter.shape[:-1], -1) for y in jnp.split(x, 3, -1)]
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        return update * cand + (1 - update) * deter

    def _latent(self, net, x):
        return net(x).reshape(*x.shape[:-1], self.stoch, self.classes)

    def obs_step(self, deter, stoch, prev_action, embed, is_first):
        """Posterior step: reset at episode starts, advance the memory, condition on obs."""
        keep = 1.0 - is_first.astype(jnp.float32)
        deter = deter * keep[..., None]
        stoch = stoch * keep[..., None, None]
        deter = self._core(deter, stoch, prev_action)
        return deter, self._latent(self.post_net, jnp.concatenate([deter, embed], -1))

    def img_step(self, deter, stoch, action):
        """Prior step used for imagination."""
        deter = self._core(deter, stoch, action)
        return deter, self.prior(deter)

    def prior(self, deter):
        return self._latent(self.prior_net, deter)

    def encode(self, obs):
        return self.encoder(obs)

    def decode(self, feat):
        return self.decoder(feat)

    def reward(self, feat):
        return self.reward_head(feat)

    def cont(self, feat):
        return self.cont_head(feat)[..., 0]

    def policy(self, feat):
        return self.actor(feat)

    def value(self, feat):
        return self.critic(feat)

    def init_all(self, obs):
        """Touch every submodule once, for model.init."""
        embed = self.encode(obs)
        deter = jnp.zeros((*obs.shape[:-len(self.obs_shape)], self.deter))
        stoch = jnp.zeros((*deter.shape[:-1], self.stoch, self.classes))
        action = jnp.zeros(deter.shape[:-1], dtype=jnp.int32)
        deter, _ = self.obs_step(deter, stoch, action, embed, jnp.ones(deter.shape[:-1], dtype=jnp.bool_))
        deter, _ = self.img_step(deter, stoch, action)
        feat = jnp.concatenate([stoch.reshape(*stoch.shape[:-2], -1), deter], -1)
        return self.decode(feat), self.reward(feat), self.cont(feat), self.policy(feat), self.value(feat)


def feature(deter, stoch):
    return jnp.concatenate([stoch.reshape(*stoch.shape[:-2], -1), deter], -1)


def sample_stoch(key, logits, unimix):
    """One-hot sample with straight-through gradients from the unimix'd categorical."""
    logits = unimix_logits(logits, unimix)
    probs = jax.nn.softmax(logits, -1)
    sample = jax.nn.one_hot(jax.random.categorical(key, logits), logits.shape[-1])
    return sample + probs - jax.lax.stop_gradient(probs)


# --- optimiser (official dreamerv3: AGC, then LaProp with bias correction) --------------

def clip_by_agc(clip, pmin=1e-3):
    """Adaptive gradient clipping per parameter tensor: ||g|| <= clip * max(pmin, ||p||)."""

    def update(updates, state, params):
        def f(g, p):
            upper = clip * jnp.maximum(pmin, jnp.linalg.norm(p.ravel()))
            return g / jnp.maximum(1.0, jnp.linalg.norm(g.ravel()) / upper)

        return jax.tree.map(f, updates, params), state

    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


class LaPropState(NamedTuple):
    count: jax.Array
    mu: optax.Updates
    nu: optax.Updates


def laprop(learning_rate, b1, b2, eps):
    """LaProp (Ziyin et al., 2020): normalise by the RMS first, then apply momentum."""

    def init(params):
        return LaPropState(jnp.zeros((), jnp.int32), jax.tree.map(jnp.zeros_like, params),
                           jax.tree.map(jnp.zeros_like, params))

    def update(grads, state, params=None):
        count = state.count + 1
        nu = jax.tree.map(lambda n, g: b2 * n + (1 - b2) * g * g, state.nu, grads)
        normed = jax.tree.map(lambda g, n: g / (jnp.sqrt(n / (1 - b2 ** count)) + eps), grads, nu)
        mu = jax.tree.map(lambda m, u: b1 * m + (1 - b1) * u, state.mu, normed)
        lr = learning_rate(state.count)
        updates = jax.tree.map(lambda m: -lr * m / (1 - b1 ** count), mu)
        return updates, LaPropState(count, mu, nu)

    return optax.GradientTransformation(init, update)


# --- replay ----------------------------------------------------------------------------

class Replay(NamedTuple):
    """Circular buffer of records, (capacity, num_envs, ...) per field."""

    data: dict
    pos: jax.Array
    full: jax.Array


class TrainState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    slow_critic: dict
    retnorm: jax.Array  # EMA of the 5th and 95th return percentiles
    updates: jax.Array


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    pixel_based = config["PIXEL_BASED"]
    env_id = config["ENV_ID"]

    run_name = f"{env_id}_{config['EXP_NAME']}_{'pixel' if pixel_based else 'oc'}_{config['SEED']}"
    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name,
        save_code=True,
        mode=config.get("WANDB_MODE", "online"),
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    # do not modify the seeding
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])

    num_envs = config["NUM_ENVS"]
    batch_size = config["BATCH_SIZE"]
    batch_length = config["BATCH_LENGTH"]
    gamma = config["GAMMA"]
    lam = config["GAE_LAMBDA"]
    unimix = config["UNIMIX"]
    twohot = TwoHot(jnp.linspace(-20.0, 20.0, config["NUM_BINS"]))
    assert config["NUM_BINS"] % 2 == 1, "NUM_BINS must be odd"

    env_kwargs = dict(
        pixel_based=pixel_based,
        native_downscaling=config.get("NATIVE_DOWNSCALING", True),
        image_size=config.get("IMAGE_SIZE", 64),
        clip_reward=config.get("CLIP_REWARD", False),
    )
    env = make_env(env_id, list(config.get("TRAIN_MODS") or []), eval=False, **env_kwargs)()
    assert isinstance(env.action_space(), spaces.Discrete), "only discrete action space is supported"
    action_dim = env.action_space().n
    prepare_obs = prepare_pixels if pixel_based else (lambda obs: obs)
    obs_example = prepare_obs(env.reset(jax.random.PRNGKey(0))[0])
    obs_shape, obs_dtype = obs_example.shape, obs_example.dtype

    # train ratio = replayed steps per env step; split it into env steps and gradient steps per update
    env_steps_per_grad = batch_size * batch_length / config["TRAIN_RATIO"]
    if env_steps_per_grad >= num_envs:
        rollout_length, grad_steps = max(1, round(env_steps_per_grad / num_envs)), 1
    else:
        rollout_length, grad_steps = 1, max(1, round(num_envs / env_steps_per_grad))
    steps_per_update = num_envs * rollout_length
    print(f"[dreamerv3] {rollout_length} env steps x {num_envs} envs per {grad_steps} gradient step(s): "
          f"train ratio {grad_steps * batch_size * batch_length / steps_per_update:.1f}")
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(config["TOTAL_TIMESTEPS"], steps_per_update, config.get("SCAN_STEPS", 1000))

    model = DreamerV3(
        num_actions=action_dim,
        obs_shape=tuple(obs_shape),
        pixel_based=pixel_based,
        deter=config["DETER"],
        hidden=config["HIDDEN"],
        stoch=config["STOCH"],
        classes=config["CLASSES"],
        blocks=config["BLOCKS"],
        depth=config.get("DEPTH", 16),  # pixel-only keys have defaults for the OC config
        units=config["UNITS"],
        cnn_mults=tuple(config.get("CNN_MULTS", (2, 3, 4, 4))),
        kernel=config.get("KERNEL", 5),
        enc_layers=config["ENC_LAYERS"],
        head_layers=config["HEAD_LAYERS"],
        ac_layers=config["AC_LAYERS"],
        num_bins=config["NUM_BINS"],
    )
    if pixel_based:
        assert obs_shape[0] % 2 ** len(model.cnn_mults) == 0, "IMAGE_SIZE must be divisible by 2**len(CNN_MULTS)"

    def apply(params, method, *args):
        return model.apply({"params": params}, *args, method=method)

    # same root split as the other agents, so env resets for a given seed match across agents
    key, init_key = jax.random.split(key, 2)
    params = model.init(init_key, obs_example[None], method=DreamerV3.init_all)["params"]
    print(f"[dreamerv3] parameters: {sum(x.size for x in jax.tree.leaves(params)) / 1e6:.2f}M")

    tx = optax.chain(
        clip_by_agc(config["AGC"]),
        laprop(optax.linear_schedule(0.0, config["LEARNING_RATE"], config["WARMUP"]),
               config["BETA1"], config["BETA2"], config["EPS"]),
    )
    train_state = TrainState(
        params=params,
        opt_state=tx.init(params),
        slow_critic=jax.tree.map(jnp.copy, params["critic"]),  # own buffers: the carry is donated
        retnorm=jnp.zeros(2),
        updates=jnp.array(0, dtype=jnp.int32),
    )

    capacity = int(config["BUFFER_SIZE"]) // num_envs
    assert capacity > batch_length + 1, "BUFFER_SIZE too small for BATCH_LENGTH"

    def empty(shape, dtype):
        return jnp.zeros((capacity, num_envs, *shape), dtype)

    replay = Replay(
        data={
            "obs": empty(obs_shape, obs_dtype),
            "action": empty((), jnp.int32),
            "reward": empty((), jnp.float32),
            "is_first": empty((), jnp.bool_),
            "is_last": empty((), jnp.bool_),
            "is_terminal": empty((), jnp.bool_),
            "deter": empty((config["DETER"],), jnp.bfloat16),
            "stoch": empty((config["STOCH"],), jnp.uint8),  # class index per categorical
        },
        pos=jnp.array(0, dtype=jnp.int32),
        full=jnp.array(False),
    )

    def replay_add(replay, record):
        data = jax.tree.map(lambda buf, x: buf.at[replay.pos].set(x.astype(buf.dtype)), replay.data, record)
        pos = (replay.pos + 1) % capacity
        return Replay(data, pos, replay.full | (pos == 0))

    def replay_sample(replay, key):
        """Uniform sequences of BATCH_LENGTH + 1 records (the first is the replay context)."""
        env_key, time_key = jax.random.split(key)
        size = jnp.where(replay.full, capacity, replay.pos)
        envs = jax.random.randint(env_key, (batch_size,), 0, num_envs)
        offsets = jax.random.randint(time_key, (batch_size,), 0, size - batch_length)
        start = jnp.where(replay.full, replay.pos, 0) + offsets  # never crosses the write position
        times = (start[:, None] + jnp.arange(batch_length + 1)) % capacity
        return jax.tree.map(lambda x: x[times, envs[:, None]], replay.data), (times, envs)

    def replay_refresh(replay, index, deter, stoch):
        """Write the latents computed during training back as replay context."""
        times, envs = index
        data = dict(replay.data)
        data["deter"] = data["deter"].at[times[:, 1:], envs[:, None]].set(deter.astype(jnp.bfloat16))
        data["stoch"] = data["stoch"].at[times[:, 1:], envs[:, None]].set(jnp.argmax(stoch, -1).astype(jnp.uint8))
        return replay._replace(data=data)

    def loss_fn(params, slow_critic, retnorm, batch, key):
        obs_key, img_key = jax.random.split(key)
        slow_params = {**params, "critic": slow_critic}
        data = jax.tree.map(lambda x: x[:, 1:], batch)
        reward, is_first = data["reward"], data["is_first"]
        is_last, is_terminal = data["is_last"].astype(jnp.float32), data["is_terminal"].astype(jnp.float32)
        swap = lambda x: jnp.swapaxes(x, 0, 1)

        # world model: posteriors along the replayed sequence, from the stored context latent
        embed = apply(params, DreamerV3.encode, data["obs"])

        def observe(carry, x):
            deter, stoch = carry
            emb, prev_action, first, step_key = x
            deter, post_logits = apply(params, DreamerV3.obs_step, deter, stoch, prev_action, emb, first)
            stoch = sample_stoch(step_key, post_logits, unimix)
            return (deter, stoch), (deter, stoch, post_logits)

        context = (batch["deter"][:, 0].astype(jnp.float32),
                   jax.nn.one_hot(batch["stoch"][:, 0], config["CLASSES"]))
        _, (deter, stoch, post_logits) = jax.lax.scan(
            observe, context,
            (swap(embed), swap(batch["action"][:, :-1]), swap(is_first), jax.random.split(obs_key, batch_length)),
        )
        deter, stoch, post_logits = swap(deter), swap(stoch), swap(post_logits)
        prior_logits = apply(params, DreamerV3.prior, deter)
        feat = feature(deter, stoch)

        recon = apply(params, DreamerV3.decode, feat)
        if pixel_based:
            rec_loss = ((recon - data["obs"].astype(jnp.float32) / 255.0) ** 2).sum((-3, -2, -1))
        else:
            rec_loss = ((recon - symlog(data["obs"].astype(jnp.float32))) ** 2).sum(-1)
        rew_loss = twohot.loss(apply(params, DreamerV3.reward, feat), reward)
        con_loss = optax.sigmoid_binary_cross_entropy(apply(params, DreamerV3.cont, feat), 1.0 - is_terminal)
        post_u, prior_u = unimix_logits(post_logits, unimix), unimix_logits(prior_logits, unimix)
        sg = jax.lax.stop_gradient
        dyn_loss = jnp.maximum(config["FREE_NATS"], categorical_kl(sg(post_u), prior_u).sum(-1))
        rep_loss = jnp.maximum(config["FREE_NATS"], categorical_kl(post_u, sg(prior_u)).sum(-1))

        # imagination from every posterior state, with the world model frozen
        n = batch_size * batch_length

        def imagine(carry, step_key):
            deter, stoch = carry
            action_key, latent_key = jax.random.split(step_key)
            f = feature(deter, stoch)
            action = jax.random.categorical(action_key, unimix_logits(apply(params, DreamerV3.policy, f), unimix))
            deter, prior = apply(params, DreamerV3.img_step, deter, stoch, action)
            return (deter, sample_stoch(latent_key, prior, unimix)), (f, action)

        start = (sg(deter).reshape(n, -1), sg(stoch).reshape(n, *stoch.shape[-2:]))
        last, (img_feat, img_action) = jax.lax.scan(
            imagine, start, jax.random.split(img_key, config["IMAG_HORIZON"]))
        img_feat = sg(jnp.concatenate([swap(img_feat), feature(*last)[:, None]], 1))  # (n, H + 1, F)
        img_action = swap(img_action)  # (n, H): the action at each state but the last

        img_reward = twohot.mean(apply(params, DreamerV3.reward, img_feat))
        img_cont = jax.nn.sigmoid(apply(params, DreamerV3.cont, img_feat))
        img_cont = img_cont.at[:, 0].set(1.0 - is_terminal.reshape(n))
        weight = sg(jnp.cumprod(gamma * img_cont, 1) / gamma)
        value_logits = apply(params, DreamerV3.value, img_feat)
        value = twohot.mean(value_logits)
        slow_value = twohot.mean(apply(slow_params, DreamerV3.value, img_feat))
        ret = lambda_return(jnp.zeros_like(img_cont), 1.0 - img_cont, img_reward, value, gamma, lam)

        # return normalisation: EMA of the 5th/95th percentiles, scale at least RETNORM_LIMIT
        percentiles = jnp.percentile(ret, jnp.array([config["RETNORM_LOW"], config["RETNORM_HIGH"]]))
        retnorm = (1 - config["RETNORM_RATE"]) * retnorm + config["RETNORM_RATE"] * sg(percentiles)
        ret_scale = jnp.maximum(config["RETNORM_LIMIT"], retnorm[1] - retnorm[0])
        adv = sg((ret - value[:, :-1]) / ret_scale)

        policy_logits = unimix_logits(apply(params, DreamerV3.policy, img_feat[:, :-1]), unimix)
        logpi = jnp.take_along_axis(jax.nn.log_softmax(policy_logits, -1), img_action[..., None], -1)[..., 0]
        entropy = categorical_entropy(policy_logits)
        policy_loss = weight[:, :-1] * -(logpi * adv + config["ACTOR_ENT"] * entropy)
        value_loss = weight[:, :-1] * (twohot.loss(value_logits[:, :-1], sg(ret))
                                      + config["SLOW_REG"] * twohot.loss(value_logits[:, :-1], sg(slow_value[:, :-1])))

        # replay value loss: lambda returns along the replayed sequence, bootstrapped from imagination
        rep_value_logits = apply(params, DreamerV3.value, sg(feat))
        rep_slow_value = twohot.mean(apply(slow_params, DreamerV3.value, sg(feat)))
        rep_ret = lambda_return(is_last, is_terminal, reward, ret[:, 0].reshape(batch_size, batch_length), gamma, lam)
        repval_loss = (1.0 - is_last[:, :-1]) * (
            twohot.loss(rep_value_logits[:, :-1], sg(rep_ret))
            + config["SLOW_REG"] * twohot.loss(rep_value_logits[:, :-1], sg(rep_slow_value[:, :-1])))

        losses = {
            "rec": rec_loss.mean(), "rew": rew_loss.mean(), "con": con_loss.mean(),
            "dyn": dyn_loss.mean(), "rep": rep_loss.mean(), "policy": policy_loss.mean(),
            "value": value_loss.mean(), "repval": repval_loss.mean(),
        }
        loss = sum(config[f"{k.upper()}_SCALE"] * v for k, v in losses.items())
        metrics = {
            **{f"losses/{k}": v for k, v in losses.items()},
            "losses/loss": loss,
            "charts/prior_entropy": categorical_entropy(prior_u).sum(-1).mean(),
            "charts/post_entropy": categorical_entropy(post_u).sum(-1).mean(),
            "charts/actor_entropy": entropy.mean(),
            "charts/imag_return": ret.mean(),
            "charts/imag_value": value.mean(),
            "charts/return_scale": ret_scale,
        }
        return loss, (sg(deter), sg(stoch), retnorm, metrics)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def train_step(train_state, replay, key):
        sample_key, loss_key = jax.random.split(key)
        batch, index = replay_sample(replay, sample_key)
        (_, (deter, stoch, retnorm, metrics)), grads = grad_fn(
            train_state.params, train_state.slow_critic, train_state.retnorm, batch, loss_key)
        updates, opt_state = tx.update(grads, train_state.opt_state, train_state.params)
        params = optax.apply_updates(train_state.params, updates)
        rate = config["SLOW_CRITIC_RATE"]
        slow_critic = jax.tree.map(lambda s, p: (1 - rate) * s + rate * p, train_state.slow_critic, params["critic"])
        train_state = TrainState(params, opt_state, slow_critic, retnorm, train_state.updates + 1)
        return train_state, replay_refresh(replay, index, deter, stoch), metrics

    # policy for collection: sampled latent and action
    def act(params, policy_state, obs, key):
        deter, stoch, prev_action, is_first = policy_state
        latent_key, action_key = jax.random.split(key)
        embed = apply(params, DreamerV3.encode, obs)
        deter, post_logits = apply(params, DreamerV3.obs_step, deter, stoch, prev_action, embed, is_first)
        stoch = jax.lax.stop_gradient(sample_stoch(latent_key, post_logits, unimix))
        logits = unimix_logits(apply(params, DreamerV3.policy, feature(deter, stoch)), unimix)
        return jax.random.categorical(action_key, logits), deter, stoch

    def step_envs(env_state, action):
        return jax.vmap(env.step)(env_state, action)

    obs, env_state = jax.vmap(env.reset)(jax.random.split(key, num_envs))
    obs = prepare_obs(obs)
    false = jnp.zeros(num_envs, dtype=jnp.bool_)
    current = {"obs": obs, "reward": jnp.zeros(num_envs), "is_first": ~false, "is_last": false, "is_terminal": false}
    policy_state = (
        jnp.zeros((num_envs, config["DETER"])),
        jnp.zeros((num_envs, config["STOCH"], config["CLASSES"])),
        jnp.zeros(num_envs, dtype=jnp.int32),
    )
    # every carry leaf needs its own buffer, since the carry is donated
    current, held_obs, policy_state = jax.tree.map(jnp.copy, (current, obs, policy_state))
    carry = (train_state, replay, env_state, current, held_obs, policy_state, key, jnp.array(0, dtype=jnp.int32))

    def collect_step(c, _):
        train_state, replay, env_state, current, held_obs, (deter, stoch, prev_action), key, global_step = c
        key, act_key = jax.random.split(key)
        action, deter, stoch = act(train_state.params, (deter, stoch, prev_action, current["is_first"]), current["obs"], act_key)
        replay = replay_add(replay, {**current, "action": action, "deter": deter, "stoch": jnp.argmax(stoch, -1)})

        next_obs, next_env_state, reward, terminated, truncated, info = step_envs(env_state, action)
        next_obs = prepare_obs(next_obs)
        # after a record with is_last the env is not stepped: the next record is the reset frame
        stepped = ~current["is_last"]
        reset = jnp.logical_or(info["env_done"], truncated)  # the wrapper already returned the reset frame
        where = lambda cond, a, b: jax.tree.map(lambda x, y: jnp.where(cond.reshape(-1, *[1] * (x.ndim - 1)), x, y), a, b)
        env_state = where(stepped, next_env_state, env_state)
        after_step = {
            "obs": where(reset, current["obs"], next_obs),  # stand-in for the lost terminal frame
            "reward": reward.astype(jnp.float32),
            "is_first": false,
            "is_last": jnp.logical_or(terminated, truncated),
            "is_terminal": terminated,
        }
        after_reset = {"obs": held_obs, "reward": jnp.zeros(num_envs), "is_first": ~false, "is_last": false,
                       "is_terminal": false}
        current = where(stepped, after_step, after_reset)
        held_obs = where(stepped, next_obs, held_obs)
        info = {k: v for k, v in info.items() if k.startswith("returned_episode")}
        return (train_state, replay, env_state, current, held_obs, (deter, stoch, action), key,
                global_step + num_envs), info

    zero_metrics = jax.eval_shape(lambda: train_step(train_state, replay, key)[2])
    zero_metrics = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), zero_metrics)

    def update(c):
        """ROLLOUT_LENGTH env steps for every env, then the gradient steps once LEARNING_STARTS is reached."""
        c, infos = jax.lax.scan(collect_step, c, None, length=rollout_length)
        train_state, replay, env_state, current, held_obs, policy_state, key, global_step = c

        def train(args):
            train_state, replay, key = args

            def grad_step(carry, step_key):
                train_state, replay = carry
                train_state, replay, metrics = train_step(train_state, replay, step_key)
                return (train_state, replay), metrics

            (train_state, replay), metrics = jax.lax.scan(grad_step, (train_state, replay), jax.random.split(key, grad_steps))
            return train_state, replay, jax.tree.map(lambda x: x[-1], metrics)

        key, train_key = jax.random.split(key)
        train_state, replay, metrics = jax.lax.cond(
            global_step >= config["LEARNING_STARTS"],
            train,
            lambda args: (args[0], args[1], zero_metrics),
            (train_state, replay, train_key),
        )
        infos = jax.tree.map(lambda x: x[-1], infos)
        return (train_state, replay, env_state, current, held_obs, policy_state, key, global_step), (infos, metrics)

    # greedy evaluation policy: posterior mode and argmax action
    def eval_policy(params):
        def init(n):
            return (jnp.zeros((n, config["DETER"])), jnp.zeros((n, config["STOCH"], config["CLASSES"])),
                    jnp.zeros(n, dtype=jnp.int32), jnp.ones(n, dtype=jnp.bool_))

        def act_greedy(state, obs):
            deter, stoch, prev_action, is_first = state
            embed = apply(params, DreamerV3.encode, obs)
            deter, post_logits = apply(params, DreamerV3.obs_step, deter, stoch, prev_action, embed, is_first)
            stoch = jax.nn.one_hot(jnp.argmax(post_logits, -1), config["CLASSES"])
            action = jnp.argmax(apply(params, DreamerV3.policy, feature(deter, stoch)), -1)
            return action, (deter, stoch, prev_action, is_first)

        def observe(state, actions, reward, obs, done):
            deter, stoch, _, _ = state
            return deter, stoch, actions, done

        return init, act_greedy, observe

    def save_and_eval(params, step_count):
        if config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model')
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, params]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        metrics = {}
        for mods_cfg, mod_label in eval_mod_configs(config):
            episodic_returns, env_states = evaluate(
                eval_policy(params),
                partial(make_env, mods=mods_cfg, eval=True, **env_kwargs),
                env_id,
                eval_episodes=10,
                prepare_obs=prepare_obs,
                seed=config["SEED"] + 42,  # use a different seed for evaluation
            )
            metrics[mod_label] = float(np.mean(jax.device_get(episodic_returns)))
            wandb.log({f"eval/episodic_return_{mod_label}": metrics[mod_label]}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                # Instantiate a clean renderer immune to the training env's downscaling
                clean_renderer = jaxtari.make(env_id, mods=mods_cfg).renderer
                frames = jnp.transpose(jax.vmap(clean_renderer.render)(env_states), (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
        return metrics

    def train_chunk(carry):
        return jax.lax.scan(lambda c, _: update(c), carry, None, length=scan_steps)

    print("[dreamerv3] start compile...")
    compile_start = time.perf_counter()
    # donate the carry so XLA reuses the replay buffer's memory
    compiled = jax.jit(train_chunk, donate_argnums=(0,)).lower(carry).compile()
    print(f"[dreamerv3] compilation time: {time.perf_counter() - compile_start:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[dreamerv3] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    global_step = 0
    for chunk in range(num_chunks):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(carry[0].params, int(carry[-1]))

        chunk_start = time.perf_counter()
        carry, (infos, train_metrics) = compiled(carry)
        global_step = int(carry[-1])
        metrics = {
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            **{k: float(v[-1]) for k, v in train_metrics.items()},
            "charts/gradient_steps": int(carry[0].updates),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=global_step)
        print(
            f"[dreamerv3] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
            f"loss {metrics['losses/loss']:.4f} | rec {metrics['losses/rec']:.2f} | "
            f"SPS {metrics['charts/SPS']}"
        )

    eval_metrics = save_and_eval(carry[0].params, global_step + 1)
    wandb.finish()
    return eval_metrics
