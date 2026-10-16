# ATTRIBUTION
#   acme      google-deepmind/acme, agents/jax/r2d2/learning.py @ 89080fe (Apache-2.0)
#             n-step transformed Q-loss, priority mixture, burn-in layout.
#             Changed: plain JAX instead of rlax; stop_gradient at the burn-in boundary.
#   rlax      google-deepmind/rlax, _src/multistep.py, transforms.py (Apache-2.0)
#             Value rescaling and n-step returns. Changed: rewritten to avoid the
#             dependency; h_inv rearranged for float32 precision.
#   flashbax  instadeepai/flashbax, prioritised trajectory buffer, used as a library.
#   this repo agents/dqn/dqn.py and dqn_eval.py: make_env, epsilon schedule, chunked
#             training loop and the evaluation structure. Changed: recurrent
#             evaluator, LSTM state threading.
#             agents/rainbow/rainbow.py: prioritised replay usage and dtype pattern.
#
#   Algorithms follow R2D2 (Kapturowski et al., ICLR 2019), NGU (Badia et al., ICLR
#   2020) and Agent57 (Badia et al., ICML 2020). Deviations are listed in the report.
#
# STAGE selects how much of Agent57 is switched on; all stages share one code path:
#   r2d2     one Q-network, extrinsic reward only
#   ngu      + intrinsic reward and a family of (beta, gamma) arms, one per env;
#            the Q-network learns r_e + beta * r_i
#   split_q  + separate extrinsic and intrinsic Q-networks ("heads"), combined as
#            Q = h(h^-1(Q_e) + beta * h^-1(Q_i))
#   agent57  + a sliding-window UCB bandit per env that picks the arm per game
# The Q-network parameters are stacked over the heads (1, or 2 for split_q), so
# acting, the loss and the optimiser are the same code for every stage.

import os
import random
import time
from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flashbax as fbx
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jaxtari
from jaxtari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper
)
from agents.agent57.agent57_eval import evaluate
from agents.run_utils import eval_mod_configs, plan_chunks
from rtpt import RTPT


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxtari.make(env_id, mods=mods)
        env = AtariWrapper(
            env,
            sticky_actions=0.0,
            episodic_life=not eval,
            first_fire=True,
            noop_max=30,
            full_action_space=False,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=False,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=not eval,
            )
        else:
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=not eval,
                    )
                )
            )
        env = LogWrapper(env)
        return env
    return thunk

#----torso--------------------------------------------------------------
class Torso(nn.Module):
    """Observation -> flat feature vector.
    Width 512, depth 2, from Group 27's sweep (Raban confirmed in channel).
   pixel  [R2D2] Table 2: "the same 3-layer convolutional network as DQN".
   Copied verbatim from dqn.py's QNetwork: 32/64/64, kernels 8/4/3,
   strides 4/2/1, padding VALID. 4 stacked 84x84 frames -> 3136 features (7x7x64).
    """
    pixel_based: bool 
    mlp_width: int = 512
    mlp_depth: int = 2
    @nn.compact
    def __call__(self, x):
        if self.pixel_based:
            #observations are stored in this format:(batch, channels, height, width)and flax's Conv expect (batch, height, width, channels)
            x = jnp.transpose(x, (0, 2, 3, 1))
            x= x.astype(jnp.float32) / 255.0 # the netork expects float32 inputs in [0,1]
            x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
        else:
            for i in range(self.mlp_depth):
                x = nn.Dense(self.mlp_width, kernel_init=orthogonal(np.sqrt(2.0)),bias_init=constant(0.0))(x)
                x = nn.relu(x)
        return x

#----RecurrentQnetwork--------------------------------------------------------------
class RecurrentQNetwork(nn.Module):
    """Torso -> LSTM -> dueling head. The full R2D2 Q-network.

    [R2D2] Table 2: torso, then LSTM(512), then dueling value and advantage
    heads each with a 512 hidden layer. The LSTM also receives the previous
    reward and a one-hot of the previous action.

    Recurrent, so the signature carries state:new_carry, q_values = net(carry, obs, prev_action, prev_reward)
    """
    
    action_dim: int
    pixel_based: bool
    hidden_size: int = 512
    dueling_units: int = 512
    mlp_width: int = 512
    mlp_depth: int = 2
    num_arms: int = 0   

    @nn.compact
    def __call__(self, carry, obs, prev_action, prev_reward, arm, prev_intrinsic):
        first = (prev_action < 0)[:, None]
        carry = jax.tree.map(lambda c: jnp.where(first, 0.0, c), carry)
        x = Torso(pixel_based=self.pixel_based,mlp_width=self.mlp_width,mlp_depth=self.mlp_depth)(obs)
        # [R2D2] the LSTM input carries the previous action and reward.
        inputs = [
            x,
            jax.nn.one_hot(prev_action, self.action_dim),
            prev_reward[:, None],
        ]
        # [NGU] the network is also told which arm it is playing as, and the
        # curiosity reward from the last step. That is how one set of weights
        # can play all arms. Decided in Python before jit: with num_arms = 0
        # these inputs do not exist and the network is exactly r2d2's.

        if self.num_arms > 0:
            inputs += [
                jax.nn.one_hot(arm, self.num_arms),
                prev_intrinsic[:, None],
            ]
        x = jnp.concatenate(inputs, axis=-1)

        carry, x = nn.OptimizedLSTMCell(self.hidden_size)(carry, x)

        # Dueling head: V says how good the state is, A says how much better
        # each action is than average. 
        v = nn.Dense(self.dueling_units)(x)
        v = nn.relu(v)
        v = nn.Dense(1)(v)

        a = nn.Dense(self.dueling_units)(x)
        a = nn.relu(a)
        a = nn.Dense(self.action_dim)(a)
        q = v + (a - a.mean(axis=-1, keepdims=True))
        return carry, q

    @staticmethod
    def initial_carry(batch_size, hidden_size=512):
        """Zeroed (c, h). An LSTM carries two vectors, not one."""
        return nn.OptimizedLSTMCell(hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )

# ---- NGU: controllable-state embedding ---------------------------------------
class EmbeddingNet(nn.Module):
    """Observation -> controllable-state embedding.
 
    [NGU] the embedding is 32-dimensional on Atari, on top of the same torso the
    Q-network uses. Kept linear at the output: these vectors are compared with
    euclidean distances in the episodic memory, and a ReLU would fold half the
    space onto zero.
    """
    pixel_based: bool
    embedding_dim: int = 32
    mlp_width: int = 512
    mlp_depth: int = 2
 
    @nn.compact
    def __call__(self, obs):
        x = Torso(
            pixel_based=self.pixel_based,
            mlp_width=self.mlp_width,
            mlp_depth=self.mlp_depth,
        )(obs)
        return nn.Dense(self.embedding_dim)(x)


class InverseDynamics(nn.Module):
    """(embedding_t, embedding_t+1) -> logits over actions.
 
    [NGU] one hidden layer of 128 units on the concatenated pair. Exists only to
    train EmbeddingNet; nothing downstream uses these logits.
    """
    action_dim: int
    hidden: int = 128
 
    @nn.compact
    def __call__(self, emb_t, emb_next):
        x = jnp.concatenate([emb_t, emb_next], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.action_dim)(x)

class EmbeddingTrainer(nn.Module):
    """EmbeddingNet + InverseDynamics under one set of parameters.
 
    Flax modules own their submodules, so training both halves together needs
    them in one module. `embed` is what the episodic memory calls later.
    """
    action_dim: int
    pixel_based: bool
    embedding_dim: int = 32
    hidden: int = 128
    mlp_width: int = 512
    mlp_depth: int = 2
 
    def setup(self):
        self.embedding = EmbeddingNet(
            pixel_based=self.pixel_based,
            embedding_dim=self.embedding_dim,
            mlp_width=self.mlp_width,
            mlp_depth=self.mlp_depth,
        )
        self.classifier = InverseDynamics(action_dim=self.action_dim, hidden=self.hidden)
 
    def __call__(self, obs, next_obs):
        """Training path: both observations in, action logits out."""
        return self.classifier(self.embedding(obs), self.embedding(next_obs))
 
    def embed(self, obs):
        """Inference path: the 32-d embedding the episodic memory compares."""
        return self.embedding(obs)

def embedding_loss(params, obs, next_obs, actions, model, mask=None):
    """[NGU] maximum likelihood of the action actually taken.
 
    obs, next_obs: (B, *obs_shape)   consecutive observations
    actions:       (B,) int32        the action taken between them
    model:         the EmbeddingTrainer; last, so it can be fixed  the same way r2d2_loss fixes `network`
 
    Returns the cross-entropy loss and the accuracy. Accuracy is the number to
    watch: at chance level (1/action_dim) the embedding has learned nothing, and
    novelty computed from it would be meaningless.
    """
    logits = model.apply(params, obs, next_obs)
    log_probs = jax.nn.log_softmax(logits)
    chosen = jnp.take_along_axis(log_probs, actions[:, None], axis=-1)[..., 0]
    # mask: 1 for a real (obs, next_obs) pair, 0 where next_obs already belongs to
    # the next episode; there the action says nothing about the change.
    if mask is None:
        mask = jnp.ones_like(chosen)
    denom = jnp.maximum(mask.sum(), 1.0)
    loss = -jnp.sum(chosen * mask) / denom
    accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == actions) * mask) / denom
    return loss, accuracy

# ---- NGU: intrinsic reward ---------------------------------------------------
#The intrinsic reward multiplies two novelty signals: r_i = r_episodic * min(max(alpha, 1), L)
#r_episodic      k-NN over the embeddings seen so far in this episode.
#   alpha        RND prediction error, normalised. Only ever scales r_episodic up (clipped to [1, L]).
#
# The agent then learns on r = r_e + beta * r_i, where beta depends on the arm.

@flax.struct.dataclass
class EpisodicMemory:
    """One fixed-size ring buffer of embeddings per env.
    The memory is a preallocated array plus a count of
    valid entries and a write pointer.
    """
    embeddings: jnp.ndarray    
    count: jnp.ndarray        
    write: jnp.ndarray        
    dist_mean: jnp.ndarray     
    dist_n: jnp.ndarray        

def init_episodic_memory(num_envs, size, dim):
    return EpisodicMemory(
        embeddings=jnp.zeros((num_envs, size, dim), jnp.float32),
        count=jnp.zeros((num_envs,), jnp.int32),
        write=jnp.zeros((num_envs,), jnp.int32),
        dist_mean=jnp.array(1.0, jnp.float32),
        dist_n=jnp.array(0.0, jnp.float32),
    )
def episodic_reward(memory, embedding, first, cfg):
    """[NGU] Algorithm 1: episodic novelty of `embedding`, then store it.
 
    memory:    EpisodicMemory
    embedding: (E, D) controllable-state embedding of the current observation
    first:     (E,) bool, True on the first step of an episode (prev_action < 0).
    That env's memory is wiped before it is queried.
    Returns (reward (E,), updated memory).
    """
    k = min(cfg["NUM_NEIGHBOURS"], memory.embeddings.shape[1])
    eps = cfg["KERNEL_EPSILON"]
    xi = cfg["CLUSTER_DISTANCE"]
    c = cfg["PSEUDO_COUNT_C"]
    s_max = cfg["MAX_SIMILARITY"]
 
    # a new episode starts with an empty memory
    count = jnp.where(first, 0, memory.count)
    write = jnp.where(first, 0, memory.write)
 
    # 1. squared distance to every slot
    size = memory.embeddings.shape[1]
    d2 = jnp.sum(jnp.square(memory.embeddings - embedding[:, None, :]), axis=-1)   
    valid_slot = jnp.arange(size)[None, :] < count[:, None]
    d2 = jnp.where(valid_slot, d2, jnp.inf)
 
    # 2. the k nearest
    neg_nn, _ = jax.lax.top_k(-d2, k)
    nn_d2 = -neg_nn                                   # (E, k), inf where fewer than k exist
    valid_nn = jnp.isfinite(nn_d2)
 
    # 3. update the running mean of squared k-NN distances, then normalise it
    n_new = jnp.sum(valid_nn)
    sum_new = jnp.sum(jnp.where(valid_nn, nn_d2, 0.0))
    total = memory.dist_n + n_new
    dist_mean = jnp.where(
        n_new > 0,
        (memory.dist_mean * memory.dist_n + sum_new) / jnp.maximum(total, 1.0),
        memory.dist_mean,
    )
    d_n = nn_d2 / jnp.maximum(dist_mean, 1e-8)
 
    # 4. treat very close states as the same state
    d_n = jnp.maximum(d_n - xi, 0.0)
 
    # 5. kernel: 1 for an identical state, falling towards 0 with distance
    kernel = jnp.where(valid_nn, eps / (d_n + eps), 0.0)
 
    # 6. sum(kernel) is a soft visit count n, so the reward is the count-based
    #    bonus 1/sqrt(n)
    #    Note: each kernel value is at most 1 

    s = jnp.sqrt(jnp.sum(kernel, axis=-1)) + c
    reward = jnp.where(s > s_max, 0.0, 1.0 / s)
 
    # Implementation choice, not in the paper: an empty memory would give
    # s = c and a reward of 1/c = 1000 on every episode's first step, which would
    # dominate everything else. Return 0 until there is something to compare to.
    reward = jnp.where(count > 0, reward, 0.0)
 
    # 7. store the embedding in the ring buffer
    embeddings = jax.vmap(lambda m, w, x: m.at[w].set(x))(memory.embeddings, write, embedding)
    new_memory = EpisodicMemory(
        embeddings=embeddings,
        count=jnp.minimum(count + 1, size),
        write=(write + 1) % size,
        dist_mean=dist_mean,
        dist_n=total,
    )
    return reward, new_memory

# ---- NGU: lifelong novelty (RND) ---------------------------------------------
# A target network is initialised randomly and never trained; a predictor learns to copy its output. Where the predictor is still wrong, the state has rarely been
# seen during training. Unlike the episodic memory, this never resets.
 
 
class RNDNetwork(nn.Module):
    """Observation -> feature vector. Used twice: frozen target, trained predictor.
    """
    pixel_based: bool
    output_dim: int = 128
    mlp_width: int = 512
    mlp_depth: int = 2
 
    @nn.compact
    def __call__(self, obs):
        x = Torso(pixel_based=self.pixel_based, mlp_width=self.mlp_width, mlp_depth=self.mlp_depth)(obs)
        return nn.Dense(self.output_dim)(x)
 
 
def rnd_error(predictor_params, target_params, network, obs):
    """Per-observation squared prediction error, shape (B,)."""
    pred = network.apply(predictor_params, obs)
    target = jax.lax.stop_gradient(network.apply(target_params, obs))
    return jnp.mean(jnp.square(pred - target), axis=-1)
 
 
def rnd_loss(predictor_params, target_params, obs, network):
    """Train the predictor to copy the frozen target. `network` last for partial()."""
    return jnp.mean(rnd_error(predictor_params, target_params, network, obs))
 
 
@flax.struct.dataclass
class RunningStats:
    """Running mean and variance, updated a batch at a time (Chan et al.)."""
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray
 
 
def init_running_stats():
    return RunningStats(mean=jnp.array(0.0), var=jnp.array(1.0), count=jnp.array(1e-4))
 
 
def update_running_stats(stats, x):
    batch_mean, batch_var, n = jnp.mean(x), jnp.var(x), x.size
    delta = batch_mean - stats.mean
    total = stats.count + n
    mean = stats.mean + delta * n / total
    m2 = stats.var * stats.count + batch_var * n + jnp.square(delta) * stats.count * n / total
    return RunningStats(mean=mean, var=m2 / total, count=total)
 
 
def rnd_modulator(error, stats, cfg):
    """[NGU] alpha = 1 + (err - mean) / std, clipped to [1, L].
 
    Clipping below at 1 means a familiar state never REDUCES the episodic reward;
    lifelong novelty can only amplify it, by at most L = 5.
    """
    alpha = 1.0 + (error - stats.mean) / jnp.sqrt(stats.var + 1e-8)
    return jnp.clip(alpha, 1.0, cfg["INTRINSIC_CLIP_L"])
 
 
def intrinsic_reward(r_episodic, modulator):
    """[NGU] r_i = r_episodic * min(max(alpha, 1), L)."""
    return r_episodic * modulator
 
 
def mixed_reward(r_extrinsic, r_intrinsic, beta):
    """[NGU] the reward an arm learns from: r = r_e + beta * r_i.
 
    beta = 0 is the purely exploitative arm; with r_i present but beta = 0 the
    agent must behave exactly like plain R2D2.
    """
    return r_extrinsic + beta * r_intrinsic

# ---- NGU: policy family (arms) -----------------------------------------------
# NGU uses several policies, called arms, that share one network.
#
# Each arm differs by:
#   beta   weight of intrinsic reward
#   gamma  planning horizon
#
# arm 0:     beta = 0,        gamma = gamma_max  -> exploit
# arm N - 1: beta = beta_max, gamma = gamma_min  -> explore
#
# Intermediate beta values follow a sigmoid schedule:
#
#   beta_j = beta_max * sigmoid(10 * (2*j - (N - 2)) / (N - 2))
#
# with beta_0 = 0 and beta_{N-1} = beta_max.
# This places more arms near low and high curiosity, and fewer in the middle.

@flax.struct.dataclass
class BanditState:
    """[Agent57] sec. 4: one sliding-window UCB bandit per env (= per actor).
 
    Each env remembers the last W games it played: which arm it used and what
    unclipped game score it reached. A game ends at game over or truncation, not
    when a life is lost (that ends only the training episode). At the start of
    every game it picks the arm to play next from that window.
 
    arms/returns are ring buffers over games, not steps.
    """
    arms: jnp.ndarray       # (E, W) arm id of each remembered game
    returns: jnp.ndarray    # (E, W) score of that game
    write: jnp.ndarray      # (E,) next slot
    count: jnp.ndarray      # (E,) how many slots are filled, capped at W
 
 
def init_bandit(num_envs, window, num_arms):
    return BanditState(
        arms=jnp.zeros((num_envs, window), jnp.int32),
        returns=jnp.zeros((num_envs, window), jnp.float32),
        write=jnp.zeros((num_envs,), jnp.int32),
        count=jnp.zeros((num_envs,), jnp.int32),
    )
 
 
def bandit_select(bandit, num_arms, rng, cfg):
    """[Agent57] UCB over the window: score_k = mean return of arm k + beta * sqrt(1/N_k).
 
    An arm not present in the window scores +inf, so every arm is tried before
    the bandit starts exploiting. With probability UCB_EPSILON a random arm is
    chosen instead, as the paper prescribes. Returns one arm per env.
    """
    window = bandit.arms.shape[1]
    filled = jnp.arange(window)[None, :] < bandit.count[:, None]          # (E, W)
    is_arm = (bandit.arms[:, :, None] == jnp.arange(num_arms)[None, None, :]) & filled[:, :, None]
    n_k = jnp.sum(is_arm, axis=1)                                          # (E, A)
    sum_k = jnp.sum(bandit.returns[:, :, None] * is_arm, axis=1)
    mean_k = sum_k / jnp.maximum(n_k, 1)
    score = jnp.where(n_k == 0, jnp.inf, mean_k + cfg.get("UCB_BETA", 1.0) * jnp.sqrt(1.0 / jnp.maximum(n_k, 1)))
    greedy = jnp.argmax(score, axis=-1)
 
    k1, k2 = jax.random.split(rng)
    random_arm = jax.random.randint(k1, greedy.shape, 0, num_arms)
    explore = jax.random.uniform(k2, greedy.shape) < cfg.get("UCB_EPSILON", 0.5)
    return jnp.where(explore, random_arm, greedy).astype(jnp.int32)
 
 
def bandit_update(bandit, arm, game_return, done):
    """Record (arm, game score) for every env whose game just ended (done = game over or truncation)."""
    window = bandit.arms.shape[1]
    idx = bandit.write
    write_one = lambda buf, i, v, d: buf.at[i].set(jnp.where(d, v, buf[i]))
    arms = jax.vmap(write_one)(bandit.arms, idx, arm, done)
    returns = jax.vmap(write_one)(bandit.returns, idx, game_return, done)
    return BanditState(
        arms=arms, returns=returns,
        write=jnp.where(done, (idx + 1) % window, idx),
        count=jnp.where(done, jnp.minimum(bandit.count + 1, window), bandit.count),
    )
 


 
 
def arm_schedule(num_arms, beta_max, gamma_max, gamma_min):
    """[NGU] beta_j and gamma_j for every arm. Returns two (num_arms,) arrays."""
    j = jnp.arange(num_arms, dtype=jnp.float32)
    n = num_arms
 
    # beta: 0 for arm 0, beta_max for the last arm, sigmoid in between
    inner = beta_max * jax.nn.sigmoid(10.0 * (2.0 * j - (n - 2)) / (n - 2))
    betas = jnp.where(j == 0, 0.0, jnp.where(j == n - 1, beta_max, inner))
 
    # gamma: interpolate log(1 - gamma) linearly between the two ends
    log_1mg = ((n - 1 - j) * jnp.log(1.0 - gamma_max) + j * jnp.log(1.0 - gamma_min)) / (n - 1)
    gammas = 1.0 - jnp.exp(log_1mg)
    return betas, gammas




 
    




 

class Agent57TrainState(TrainState):
    target_params: flax.core.FrozenDict


@flax.struct.dataclass
class TimeStep:
    obs: jnp.ndarray            # observation BEFORE acting OC: (104,) float32. Pixel: (4,84,84) uint8.
    action: jnp.ndarray         # action index, 0..action_dim-1
    reward: jnp.ndarray         # extrinsic reward, clipped during training
    done: jnp.ndarray           # episode ended here; zeroes the future term  in the Q target

    prev_action: jnp.ndarray    # [R2D2] the LSTM input carries the previous
    prev_reward: jnp.ndarray    #   action and reward, not just the observation. Must be STORED: a sampled sequence has no access to the step before it.

    arm: jnp.ndarray            # which of the (beta, gamma) policies acted; 0 in r2d2
    intrinsic_reward: jnp.ndarray   # curiosity reward for this step; 0 in r2d2
    prev_intrinsic: jnp.ndarray     # [NGU] curiosity reward of the previous step; a network input, stored for the same reason as prev_reward
    # The LSTM state is not stored per step: sequences only ever start every
    # SEQUENCE_PERIOD steps, so single_run keeps one state per sequence start
    # (seq_carry) next to the buffer.


@flax.struct.dataclass
class PolicyState:
    """What the recurrent policy carries from one step to the next, per env."""
    lstm: tuple                 # (c, h), each (H, E, hidden): one LSTM state per head
    prev_action: jnp.ndarray    # (E,) -1 on the first step of an episode
    prev_reward: jnp.ndarray    # (E,)
    prev_intrinsic: jnp.ndarray # (E,)
    memory: EpisodicMemory      # NGU episodic memory; None in r2d2


@flax.struct.dataclass
class ActorState:
    env_state: object
    obs: jnp.ndarray
    policy: PolicyState
    arm: jnp.ndarray            # (E,) arm each env plays in its current game
    rnd_stats: RunningStats     # running statistics of the RND error; None in r2d2
    bandit: BanditState         # [Agent57] per-env meta-controller; None before stage agent57
    game_return: jnp.ndarray    # (E,) unclipped score of the current game, for the bandit
    global_step: jnp.ndarray


#---- R2D2 loss ---------------------------------------------------------------
""" Logic follows Acme's R2D2 learner and rlax's transformed n-step Q-learning
acme:  google-deepmind/acme, acme/agents/jax/r2d2/learning.py @ 89080fe  (Apache-2.0)
rlax:  google-deepmind/rlax, rlax/_src/multistep.py, transforms.py  (Apache-2.0)
Changed: plain JAX instead of rlax; stop_gradient on the burn-in state
(Acme lets gradients flow into the burn-in unroll)."""
def h(x, eps):
    """[R2D2]value rescaling : h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x"""
    return jnp.sign(x)*(jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + eps * x
def h_inv(x, eps):
    """[R2D2] Inverse value rescaling:h^-1(x) = sign(x) * (((sqrt(1 + 4*eps*(|x| + 1 + eps)) - 1) / (2*eps))^2 - 1 )
    """
    a= jnp.sqrt(1.0 + 4.0 * eps * (jnp.abs(x) + 1.0 + eps)) - 1.0
    b= 2.0 * eps
    return jnp.sign(x) * (jnp.square(a / b) - 1.0)
def n_step_targets(rewards, discounts,bootstrap , n ):
    """  G_t = r_t + d_t r_{t+1} + ... + d_t...d_{t+n-1} * bootstrap[t+n-1]
    The last n-1 steps have fewer rewards left and bootstrap from bootstrap[-1]. """
    t= rewards.shape[0]
    pad= n-1
    rewards= jnp.concatenate([rewards, jnp.zeros((pad,)+ rewards.shape[1:], dtype=rewards.dtype)])
    discounts= jnp.concatenate([discounts, jnp.ones((pad,)+ discounts.shape[1:], dtype=discounts.dtype)])
    bootstrap= jnp.concatenate([bootstrap, jnp.repeat(bootstrap[-1:], pad, axis=0)])
    g= bootstrap[n-1:n-1+t]
    for i in reversed(range(n)):
        g= rewards[i:i+t] + discounts[i:i+t] * g
    return g


def heads_apply(network, params, lstm, obs, prev_action, prev_reward, arm, prev_intrinsic):
    """One step of every head: params and lstm are stacked over heads, the inputs are shared.
    Returns (lstm, q of shape (H, B, A))."""
    return jax.vmap(network.apply, in_axes=(0, 0, None, None, None, None, None))(
        params, lstm, obs, prev_action, prev_reward, arm, prev_intrinsic)


def unroll_heads(network, params, carry, seq):
    """Run every head over a time-first sequence. Returns (last carry, q of shape (H, T, B, A))."""
    def unroll(params, carry):
        def step(carry, x):
            return network.apply(params, carry, *x)
        # [NGU] the arm and the previous curiosity reward are network inputs too
        # (ignored by the network when it has no arms, i.e. in r2d2)
        xs = (seq.obs, seq.prev_action, seq.prev_reward, seq.arm, seq.prev_intrinsic)
        return jax.lax.scan(step, carry, xs)
    return jax.vmap(unroll)(params, carry)


def combine_q(q, beta, eps):
    """The Q-values the policy acts on, from the heads' raw outputs q (H, ..., A).

    One head: that head. [SPLIT_Q] two heads: Q = h(h^-1(Q_e) + beta_j * h^-1(Q_i)),
    the combination of Agent57 sec. 3.1 in real-return space. beta: (...,).
    """
    if q.shape[0] == 1:
        return q[0]
    return h(h_inv(q[0], eps) + jnp.asarray(beta)[..., None] * h_inv(q[1], eps), eps)


def r2d2_loss(params, target_params, start_carry, batch, probabilities, network, cfg,
              arm_betas, arm_gammas, split_q, double_q=True):
    """Transformed n-step double Q-learning on stored sequences, for every head at once.

    params, target_params: stacked over heads (H = 1, or 2 with split_q).
    start_carry: (c, h), each (H, B, hidden), the stored LSTM state at each sequence start.
    batch: TimeStep with arrays shaped (B, BURN_IN + SEQ_LEN, ...).
    probabilities: (B,) sampling probabilities from the prioritised buffer.

    Rewards and discounts per head:
      r2d2     r_e, GAMMA (arm_betas = [0], arm_gammas = [GAMMA])
      ngu      r_e + beta_arm * r_i, gamma_arm
      split_q  head 0 learns r_e, head 1 learns r_i, both with gamma_arm. Both
               bootstrap from the SAME next action, argmax of the combined Q.

    Returns loss and (new priorities (B,), loss per head (H,), mean Q of taken actions per head (H,)).
    """
    burn_in = cfg["BURN_IN_LENGTH"]
    eps = cfg["VALUE_RESCALING_EPSILON"]

    data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), batch)      # (L, B, ...)
    burn = jax.tree.map(lambda x: x[:burn_in], data)
    learn = jax.tree.map(lambda x: x[burn_in:], data)

    # 1. burn-in: rebuild the memory from the stored state. No learning here.
    online_carry, _ = unroll_heads(network, params, start_carry, burn)
    target_carry, _ = unroll_heads(network, target_params, start_carry, burn)
    online_carry = jax.lax.stop_gradient(online_carry)

    # 2. Q-values on the part we learn from
    _, q_online = unroll_heads(network, params, online_carry, learn)          # (H, T, B, A)
    _, q_target = unroll_heads(network, target_params, target_carry, learn)
    q_target = jax.lax.stop_gradient(q_target)

    # 3. double Q: the online net picks the next action (on the combined Q), the target net scores it
    beta = arm_betas[learn.arm]                                               # (T, B)
    selector = q_online if double_q else q_target
    next_action = jnp.argmax(jax.lax.stop_gradient(combine_q(selector[:, 1:], beta[1:], eps)), axis=-1)
    next_value = jnp.take_along_axis(
        q_target[:, 1:], jnp.broadcast_to(next_action[None, ..., None], q_target[:, 1:, ..., :1].shape), axis=-1
    )[..., 0]                                                                 # (H, T-1, B)

    # 4. n-step target, computed in real-return space, then squashed with h
    r_e, r_i = learn.reward[:-1], learn.intrinsic_reward[:-1]
    if split_q:
        rewards = jnp.stack([r_e, r_i])
    else:
        # [NGU] each step learns from its own arm's reward mix: r = r_e + beta_arm * r_i.
        # Arm 0 has beta = 0, so it learns exactly what r2d2 learns.
        rewards = mixed_reward(r_e, r_i, beta[:-1])[None]
    gamma = arm_gammas[learn.arm[:-1]]
    discounts = gamma * (1.0 - learn.done[:-1].astype(jnp.float32))
    returns = jax.vmap(partial(n_step_targets, n=cfg["N_STEP"]), in_axes=(0, None, 0))(
        rewards, discounts, h_inv(next_value, eps))
    target = jax.lax.stop_gradient(h(returns, eps))

    actions = learn.action[:-1]
    q_taken = jnp.take_along_axis(
        q_online[:, :-1], jnp.broadcast_to(actions[None, ..., None], q_online[:, :-1, ..., :1].shape), axis=-1
    )[..., 0]
    td = target - q_taken                                                     # (H, T-1, B)

    # 5. loss: sum over time per sequence, importance-weighted mean over the batch, summed over heads
    per_seq = 0.5 * jnp.square(td).sum(axis=1)                                # (H, B)
    weights = (1.0 / (probabilities + 1e-10)) ** cfg["IMPORTANCE_SAMPLING_EXPONENT"]
    weights = weights / weights.max()
    head_losses = jnp.mean(weights * per_seq, axis=-1)                        # (H,)

    # 6. [R2D2] priority = eta * max|TD| + (1 - eta) * mean|TD| over the sequence.
    # [SPLIT_Q] combined into the one priority the buffer needs as
    # priority_e + beta_arm * priority_i (the project's rule, not a paper formula).
    abs_td = jnp.abs(td)
    eta = cfg["PRIORITY_ETA"]
    priorities = eta * abs_td.max(axis=1) + (1.0 - eta) * abs_td.mean(axis=1)  # (H, B)
    if split_q:
        priorities = priorities[0] + beta[0] * priorities[1]
    else:
        priorities = priorities[0]

    return head_losses.sum(), (jax.lax.stop_gradient(priorities), head_losses, q_taken.mean(axis=(1, 2)))


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    stage = config.get("STAGE", "r2d2")
    assert stage in ("r2d2", "ngu", "split_q", "agent57"), f"unknown STAGE: {stage}"

    use_intrinsic = stage != "r2d2"
    split_q = stage in ("split_q", "agent57")    # Agent57 uses split-Q underneath the meta-controller
    meta = stage == "agent57"
    num_heads = 2 if split_q else 1

    run_name = f"{config['ENV_ID']}_{config['EXP_NAME']}_{'pixel' if config['PIXEL_BASED'] else 'oc'}_{stage}_{config['SEED']}"

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

    env = make_env(
        config["ENV_ID"],
        list(config.get("TRAIN_MODS", [])),
        config["PIXEL_BASED"],
        config.get("NATIVE_DOWNSCALING", True),
        False,
    )()

    action_dim = env.action_space().n
    obs_shape = env.observation_space().shape
    # Pixel observations arrive with a trailing channel dimension that the
    # buffer does not store.
    if config["PIXEL_BASED"]:
        obs_shape = obs_shape[:-1]

    num_envs = config["NUM_ENVS"]
    hidden = config["HIDDEN_SIZE"]
    burn_in = config["BURN_IN_LENGTH"]
    seq_len = config["SEQUENCE_LENGTH"]
    period = config["SEQUENCE_PERIOD"]
    steps_per_call = config["TRAIN_FREQUENCY"]
    total_timesteps = config["TOTAL_TIMESTEPS"]
    eps_vr = config["VALUE_RESCALING_EPSILON"]
    sample_len = burn_in + seq_len

    # --- env helpers (copied from dqn.py) -----------------------------------
    @jax.jit
    def vmap_reset(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        # a lost life ends the episode (episodic life) but not the game
        game_over = jnp.logical_or(info["env_done"], truncated)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, next_done, game_over, info

    # --- arms -----------------------------------------------------------------
    # [NGU] a fixed (beta, gamma) per arm; env i starts on arm i mod NUM_ARMS and
    # keeps it, unless the agent57 bandit picks a new arm after each game.
    # r2d2 has a single arm with beta = 0 and gamma = GAMMA.
    if use_intrinsic:
        num_arms = config["NUM_ARMS"]
        arm_betas, arm_gammas = arm_schedule(num_arms, config["BETA_MAX"], config["GAMMA_MAX"], config["GAMMA_MIN"])
    else:
        num_arms = 1
        arm_betas, arm_gammas = jnp.zeros((1,)), jnp.full((1,), config["GAMMA"], jnp.float32)

    # --- network ------------------------------------------------------------
    network = RecurrentQNetwork(
        action_dim=action_dim,
        pixel_based=config["PIXEL_BASED"],
        hidden_size=hidden,
        dueling_units=config["DUELING_UNITS"],
        mlp_width=config.get("MLP_WIDTH", 512),
        mlp_depth=config.get("MLP_DEPTH", 2),
        # with num_arms = 0 the arm inputs do not exist and the network is exactly r2d2's
        num_arms=num_arms if use_intrinsic else 0,
    )

    def zero_lstm(n):
        """Zeroed (c, h) for every head, each (H, n, hidden)."""
        return (jnp.zeros((num_heads, n, hidden)), jnp.zeros((num_heads, n, hidden)))

    key, reset_key, net_key = jax.random.split(key, 3)
    dummy_obs, _ = env.reset(reset_key)
    dummy_obs = dummy_obs.reshape(obs_shape)

    def init_head(k):
        return network.init(
            k, RecurrentQNetwork.initial_carry(1, hidden), dummy_obs[None],
            jnp.zeros((1,), jnp.int32), jnp.zeros((1,), jnp.float32),
            jnp.zeros((1,), jnp.int32), jnp.zeros((1,), jnp.float32),
        )
    params = jax.vmap(init_head)(jax.random.split(net_key, num_heads))

    tx = optax.adam(
        learning_rate=config["LEARNING_RATE"],
        b1=config["ADAM_B1"],
        b2=config["ADAM_B2"],
        eps=config["ADAM_EPS"],
    )
    # Adam is elementwise, so one optimiser over the stacked heads is the same as one per head
    agent_state = Agent57TrainState.create(
        apply_fn=network.apply, params=params, target_params=jax.tree.map(jnp.copy, params), tx=tx)
    loss_fn = partial(r2d2_loss, network=network, cfg=config, arm_betas=arm_betas, arm_gammas=arm_gammas,
                      split_q=split_q, double_q=config.get("DOUBLE_Q", True))

    # --- replay buffer -----------------------------------------------------
    replay_buffer = fbx.make_prioritised_trajectory_buffer(
        add_batch_size=num_envs,
        sample_batch_size=config["BATCH_SIZE"],
        sample_sequence_length=sample_len,
        period=period,                                # stride between starts -> 40-step overlap
        min_length_time_axis=max(sample_len, config["LEARNING_STARTS"] // num_envs),
        max_length_time_axis=config["BUFFER_SIZE"] // num_envs,
        priority_exponent=config["PRIORITY_EXPONENT"],
    )
    dummy_timestep = TimeStep(
        obs=dummy_obs,
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
        done=jnp.zeros((), dtype=jnp.bool_),
        prev_action=jnp.zeros((), dtype=jnp.int32),
        prev_reward=jnp.zeros((), dtype=jnp.float32),
        arm=jnp.zeros((), dtype=jnp.int32),
        intrinsic_reward=jnp.zeros((), dtype=jnp.float32),
        prev_intrinsic=jnp.zeros((), dtype=jnp.float32),
    )
    buffer_state = replay_buffer.init(dummy_timestep)

    # Sequences can only start at buffer time positions that are multiples of
    # `period` (flashbax item i starts at env i // num_slots, position
    # (i % num_slots) * period), so the LSTM state is kept for those positions
    # only: seq_carry[env, slot] is the state before the step at slot * period.
    max_time = buffer_state.experience.done.shape[1]   # flashbax rounds it down to a multiple of period
    num_slots = max_time // period
    seq_carry = tuple(jnp.zeros((num_envs, num_slots, num_heads, hidden)) for _ in range(2))

    obs_bytes = int(np.prod(obs_shape)) * dummy_obs.dtype.itemsize
    buffer_gb = (num_envs * max_time * obs_bytes + seq_carry[0].size * 2 * 4) / 1e9

    # --- NGU: side networks ---------------------------------------------------
    #   embedding   observation -> 32-d controllable state, for episodic novelty
    #   RND         frozen random target + trained predictor, for lifelong novelty
    side = {}
    if use_intrinsic:
        emb_model = EmbeddingTrainer(
            action_dim=action_dim, pixel_based=config["PIXEL_BASED"],
            embedding_dim=config.get("EMBEDDING_DIM", 32),
            mlp_width=config.get("MLP_WIDTH", 512), mlp_depth=config.get("MLP_DEPTH", 2))
        rnd_net = RNDNetwork(
            pixel_based=config["PIXEL_BASED"], output_dim=config.get("RND_OUTPUT_DIM", 128),
            mlp_width=config.get("MLP_WIDTH", 512), mlp_depth=config.get("MLP_DEPTH", 2))
        key, k_emb, k_tgt, k_pred = jax.random.split(key, 4)
        # [NGU] lr 5e-4 and Adam eps 1e-4 for both side networks; L2 1e-5 on the embedding
        side_lr = config.get("NGU_LEARNING_RATE", 5e-4)
        side_eps = config.get("NGU_ADAM_EPS", 1e-4)
        side = {
            "emb": TrainState.create(
                apply_fn=emb_model.apply,
                params=emb_model.init(k_emb, dummy_obs[None], dummy_obs[None]),
                tx=optax.chain(optax.add_decayed_weights(config.get("EMBEDDING_L2", 1e-5)),
                               optax.adam(side_lr, eps=side_eps))),
            "rnd": TrainState.create(
                apply_fn=rnd_net.apply,
                params=rnd_net.init(k_pred, dummy_obs[None]),
                tx=optax.adam(side_lr, eps=side_eps)),
            "rnd_target": rnd_net.init(k_tgt, dummy_obs[None]),   # never trained
        }
    side_frames = config.get("NGU_TRAIN_FRAMES", 5)   # [NGU] side networks train on the last 5 frames

    # --- the policy, shared by collection and evaluation ------------------------
    def init_policy(n):
        return PolicyState(
            lstm=zero_lstm(n),
            prev_action=jnp.full((n,), -1, jnp.int32),
            prev_reward=jnp.zeros((n,), jnp.float32),
            prev_intrinsic=jnp.zeros((n,), jnp.float32),
            memory=init_episodic_memory(n, config["EPISODIC_MEMORY_SIZE"], config.get("EMBEDDING_DIM", 32))
            if use_intrinsic else None,
        )

    def policy_q(params, ps, obs, arm):
        """Q-values the policy acts on; the LSTM state is carried forward (the
        network wipes it itself wherever prev_action is -1)."""
        lstm, q = heads_apply(network, params, ps.lstm, obs, ps.prev_action, ps.prev_reward, arm, ps.prev_intrinsic)
        return combine_q(q, arm_betas[arm], eps_vr), ps.replace(lstm=lstm)

    def policy_observe(ps, side, rnd_stats, actions, rewards, next_obs, next_done, update_stats):
        """Network inputs for the next step. Where the episode just ended, -1 tells
        the network to start from a zero carry, and the episodic memory is wiped.
        Returns (policy state, intrinsic reward of reaching next_obs, rnd_stats)."""
        memory = ps.memory
        if use_intrinsic:
            emb = emb_model.apply(side["emb"].params, next_obs, method=EmbeddingTrainer.embed)
            r_episodic, memory = episodic_reward(memory, emb, next_done, config)
            err = rnd_error(side["rnd"].params, side["rnd_target"], rnd_net, next_obs)
            if update_stats:
                rnd_stats = update_running_stats(rnd_stats, err)
            r_int = intrinsic_reward(r_episodic, rnd_modulator(err, rnd_stats, config))
        else:
            r_int = jnp.zeros_like(rewards)
        ps = ps.replace(
            prev_action=jnp.where(next_done, -1, actions),
            prev_reward=jnp.where(next_done, 0.0, rewards),
            prev_intrinsic=jnp.where(next_done, 0.0, r_int),
            memory=memory,
        )
        return ps, r_int, rnd_stats

    # --- collection ---------------------------------------------------------
    # One call = TRAIN_FREQUENCY steps in every env. Unlike dqn, the whole chunk
    # goes into the buffer at once: the trajectory buffer keeps each env's steps
    # in order and cuts sequences out of them when sampling.
    def collect(params, side, actor, rng):
        def act_step(actor, rng):
            action_rng, explore_rng, bandit_rng = jax.random.split(rng, 3)
            # same schedule as dqn.py
            epsilon = jnp.interp(
                actor.global_step,
                jnp.array([0, config["EXPLORATION_FRACTION"] * total_timesteps]),
                jnp.array([config["START_E"], config["END_E"]]),
            )
            ps = actor.policy
            q_values, next_ps = policy_q(params, ps, actor.obs, actor.arm)
            greedy_actions = q_values.argmax(axis=-1)
            random_actions = jax.random.randint(action_rng, (num_envs,), 0, action_dim)
            explore_mask = jax.random.uniform(explore_rng, (num_envs,)) < epsilon
            actions = jnp.where(explore_mask, random_actions, greedy_actions)

            next_obs, env_state, rewards, next_done, game_over, info = vmap_step(actor.env_state, actions)
            rewards = rewards.astype(jnp.float32)   # the buffer and loss use float32
            next_ps, r_int, rnd_stats = policy_observe(
                next_ps, side, actor.rnd_stats, actions, rewards, next_obs, next_done, update_stats=True)

            timestep = TimeStep(
                obs=actor.obs,
                action=actions,
                reward=rewards,
                done=next_done,
                prev_action=ps.prev_action,
                prev_reward=ps.prev_reward,
                arm=actor.arm,
                intrinsic_reward=r_int,
                prev_intrinsic=ps.prev_intrinsic,
            )

            # [Agent57] the bandit scores whole games: its own running, unclipped
            # game score, recorded (and the arm switched) only at game over or
            # truncation, not when a life is lost.
            game_return = actor.game_return + info["env_reward"].astype(jnp.float32)
            arm, bandit = actor.arm, actor.bandit
            if meta:
                bandit = bandit_update(bandit, arm, game_return, game_over)
                new_arm = bandit_select(bandit, num_arms, bandit_rng, config)
                arm = jnp.where(game_over, new_arm, arm)
            actor = actor.replace(
                env_state=env_state, obs=next_obs, policy=next_ps, arm=arm, rnd_stats=rnd_stats,
                bandit=bandit, game_return=jnp.where(game_over, 0.0, game_return),
                global_step=actor.global_step + num_envs,
            )
            # the LSTM state BEFORE this step, kept if a sequence can start here
            return actor, (timestep, ps.lstm, info)

        actor, (traj, lstm, infos) = jax.lax.scan(act_step, actor, jax.random.split(rng, steps_per_call))
        # scan stacks time first (T, num_envs, ...); the buffer wants (num_envs, T, ...)
        traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        lstm = jax.tree.map(lambda x: jnp.transpose(x, (2, 0, 1, 3)), lstm)   # (T, H, E, hid) -> (E, T, H, hid)
        return actor, traj, lstm, infos

    # --- one learner step ----------------------------------------------------
    def update_side(side, exp):
        """[NGU] one step for the embedding and RND predictor, on the last frames only."""
        k = side_frames
        obs_t = exp.obs[:, -(k + 1):-1].reshape((-1,) + obs_shape)
        obs_tp1 = exp.obs[:, -k:].reshape((-1,) + obs_shape)
        act_t = exp.action[:, -(k + 1):-1].reshape(-1)
        valid = 1.0 - exp.done[:, -(k + 1):-1].reshape(-1).astype(jnp.float32)
        (emb_l, emb_acc), g = jax.value_and_grad(embedding_loss, has_aux=True)(
            side["emb"].params, obs_t, obs_tp1, act_t, emb_model, valid)
        emb = side["emb"].apply_gradients(grads=g)
        rnd_l, g = jax.value_and_grad(rnd_loss)(side["rnd"].params, side["rnd_target"], obs_tp1, rnd_net)
        rnd = side["rnd"].apply_gradients(grads=g)
        return {**side, "emb": emb, "rnd": rnd}, emb_acc, rnd_l

    def update(agent_state, buffer_state, seq_carry, side, rng):
        """Sample BATCH_SIZE sequences, take one gradient step, write priorities back.
        The same batch trains the NGU embedding and RND predictor."""
        batch = replay_buffer.sample(buffer_state, rng)
        env_idx, slot_idx = batch.indices // num_slots, batch.indices % num_slots
        start_carry = jax.tree.map(lambda c: jnp.swapaxes(c[env_idx, slot_idx], 0, 1), seq_carry)  # (H, B, hid)
        (_, (priorities, head_losses, q_means)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            agent_state.params, agent_state.target_params, start_carry, batch.experience, batch.probabilities)
        agent_state = agent_state.apply_gradients(grads=grads)
        buffer_state = replay_buffer.set_priorities(buffer_state, batch.indices, priorities)

        # [R2D2] target <- online every TARGET_NETWORK_FREQUENCY LEARNER steps (apply_gradients counts them)
        sync = (agent_state.step % config["TARGET_NETWORK_FREQUENCY"]) == 0
        target_params = jax.lax.cond(
            sync,
            lambda _: optax.incremental_update(agent_state.params, agent_state.target_params, 1.0),
            lambda _: agent_state.target_params,
            operand=None,
        )
        if use_intrinsic:
            side, emb_acc, rnd_l = update_side(side, batch.experience)
        else:
            emb_acc, rnd_l = jnp.float32(0.0), jnp.float32(0.0)
        return agent_state.replace(target_params=target_params), buffer_state, side, (head_losses, q_means, emb_acc, rnd_l)

    # --- play TRAIN_FREQUENCY steps, then learn once the buffer is warm ------
    def train_step(carry, _):
        agent_state, buffer_state, seq_carry, side, actor, rng = carry
        rng, collect_rng, update_rng = jax.random.split(rng, 3)
        actor, traj, lstm, infos = collect(agent_state.params, side, actor, collect_rng)

        # buffer positions this chunk is written to; keep the LSTM state where a sequence can start
        pos = (buffer_state.current_index + jnp.arange(steps_per_call)) % max_time
        slot = jnp.where(pos % period == 0, pos // period, num_slots)   # num_slots is out of range: dropped
        seq_carry = jax.tree.map(lambda store, c: store.at[:, slot].set(c, mode="drop"), seq_carry, lstm)
        buffer_state = replay_buffer.add(buffer_state, traj)

        zeros = (jnp.zeros((num_heads,)), jnp.zeros((num_heads,)), jnp.float32(0.0), jnp.float32(0.0))
        agent_state, buffer_state, side, metrics = jax.lax.cond(
            replay_buffer.can_sample(buffer_state),
            lambda a, b, s: update(a, b, seq_carry, s, update_rng),
            lambda a, b, s: (a, b, s, zeros),
            agent_state, buffer_state, side)
        r_int = traj.intrinsic_reward
        metrics = metrics + (r_int.mean(), r_int.max())
        return (agent_state, buffer_state, seq_carry, side, actor, rng), (infos, metrics)

    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))
    actor = ActorState(
        env_state=env_state,
        obs=obs,
        policy=init_policy(num_envs),
        arm=jnp.arange(num_envs, dtype=jnp.int32) % num_arms,
        rnd_stats=init_running_stats() if use_intrinsic else None,
        bandit=init_bandit(num_envs, config.get("UCB_WINDOW", 160), num_arms) if meta else None,
        game_return=jnp.zeros((num_envs,), jnp.float32),
        global_step=jnp.array(0, jnp.int32),
    )

    print(f"[agent57] stage={stage} env={config['ENV_ID']} {'pixel' if config['PIXEL_BASED'] else 'oc'}")
    print(f"[agent57] action_dim={action_dim} obs_shape={obs_shape} obs_bytes={obs_bytes}")
    print(f"[agent57] buffer: {num_envs * max_time} transitions + {num_envs * num_slots} sequence-start "
          f"LSTM states = {buffer_gb:.2f} GB, obs dtype {buffer_state.experience.obs.dtype}")

    # --- save and evaluate ---------------------------------------------------
    def save_and_eval(agent_state, side, rnd_stats, step_count):
        params = agent_state.params
        if config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_{step_count}_{int(time.time())}.cleanrl_model')
            saved = {"params": params}   # stacked over heads: [Q] or [Q_e, Q_i]
            if use_intrinsic:
                saved["side"] = {"emb": side["emb"].params, "rnd": side["rnd"].params,
                                 "rnd_target": side["rnd_target"], "rnd_stats": rnd_stats}
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes([config, saved]))
            print(f"model saved to {model_path}")

        # Greedy play as arm 0, the exploitative arm (beta = 0). The previous
        # intrinsic reward is a network input, so the curiosity reward keeps being
        # computed as in training (frozen side networks and RND statistics): fed as
        # a constant 0 instead, arm 0 falls from about -10 to -20 on Pong.
        def act(ps, obs):
            arm = jnp.zeros((obs.shape[0],), jnp.int32)
            q_values, ps = policy_q(params, ps, obs, arm)
            return q_values.argmax(axis=-1), ps

        def observe(ps, actions, rewards, next_obs, done):
            return policy_observe(ps, side, rnd_stats, actions, rewards, next_obs, done, update_stats=False)[0]

        print(f"running evaluation at step {step_count}...")
        metrics = {}
        for mods_cfg, mod_label in eval_mod_configs(config):
            episodic_returns, env_states = evaluate(
                (init_policy, act, observe),
                partial(
                    make_env,
                    mods=mods_cfg,
                    pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                    eval=True,
                ),
                config["ENV_ID"],
                eval_episodes=10,
                obs_shape=obs_shape,
                seed=config["SEED"] + 42,  # use a different seed for evaluation
            )
            metrics[mod_label] = float(np.mean(jax.device_get(episodic_returns)))
            wandb.log({f"eval/episodic_return_{mod_label}": metrics[mod_label]}, step=step_count)

            if config.get("CAPTURE_VIDEO", False):
                # Instantiate a clean renderer immune to the training env's downscaling
                clean_renderer = jaxtari.make(config["ENV_ID"], mods=mods_cfg).renderer
                frames = jnp.transpose(jax.vmap(clean_renderer.render)(env_states), (0, 3, 1, 2))
                video = wandb.Video(np.array(frames), fps=30, format="mp4")
                wandb.log({f"eval/video_{mod_label}": video}, step=step_count)
                print(f"Video (eval) logged to wandb with {frames.shape[0]} frames ({mod_label}).")
        return metrics

    # --- training loop -------------------------------------------------------
    # same budget rule for every agent: whole jitted chunks within TOTAL_TIMESTEPS
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(
        total_timesteps, num_envs * steps_per_call, config.get("SCAN_STEPS", 1000))

    def scanned_steps(carry):
        return jax.lax.scan(train_step, carry, None, length=scan_steps)

    print(f"[agent57] start compile...")
    start_compile = time.perf_counter()
    carry = (agent_state, buffer_state, seq_carry, side, actor, key)
    # donate the carry so XLA reuses the replay buffer's memory; lower/compile AOT so nothing is donated before the loop
    compiled = jax.jit(scanned_steps, donate_argnums=(0,)).lower(carry).compile()
    print(f"[agent57] compilation time: {time.perf_counter() - start_compile:.2f}s")

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks)
    rtpt.start()
    run_time = time.perf_counter()
    print(f"[agent57] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
    for chunk in range(num_chunks):
        rtpt.step()
        if config.get("EVAL_DURING_TRAIN", False) and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
            save_and_eval(carry[0], carry[3], carry[4].rnd_stats, int(carry[4].global_step))

        chunk_start = time.perf_counter()
        carry, (infos, (head_losses, q_means, emb_acc, rnd_l, r_int_mean, r_int_max)) = compiled(carry)
        global_step = int(carry[4].global_step)
        metrics = {
            "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
            "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
            # the extrinsic head (the only one before split_q)
            "losses/td_loss": float(head_losses[-1, 0]),
            "losses/q_values": float(q_means[-1, 0]),
            "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
            "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
            "charts/time": time.perf_counter() - run_time,
            "charts/global_step": global_step,
        }
        if split_q:
            metrics["losses/td_loss_intrinsic"] = float(head_losses[-1, 1])
            metrics["losses/q_values_intrinsic"] = float(q_means[-1, 1])
        if use_intrinsic:
            metrics.update({
                "ngu/intrinsic_reward_mean": float(r_int_mean[-1]),
                "ngu/intrinsic_reward_max": float(r_int_max[-1]),
                "ngu/embedding_accuracy": float(emb_acc[-1]),
                "ngu/rnd_loss": float(rnd_l[-1]),
            })
        if meta:
            metrics["agent57/mean_arm"] = float(carry[4].arm.mean())
        wandb.log(metrics, step=global_step)
        print(
            f"[agent57] chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
            f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
            f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
            f"td_loss {metrics['losses/td_loss']:.4f} | q_val {metrics['losses/q_values']:.4f} | "
            f"SPS {metrics['charts/SPS']}"
            + (f" | r_int {metrics['ngu/intrinsic_reward_mean']:.3f} | emb_acc {metrics['ngu/embedding_accuracy']:.2f}"
               if use_intrinsic else "")
        )

    eval_metrics = save_and_eval(carry[0], carry[3], carry[4].rnd_stats, global_step + 1)
    wandb.finish()
    return eval_metrics
