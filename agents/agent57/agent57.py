# ATTRIBUTION
#   acme      google-deepmind/acme, agents/jax/r2d2/learning.py @ 89080fe (Apache-2.0)
#             n-step transformed Q-loss, priority mixture, burn-in layout.
#             Changed: plain JAX instead of rlax; stop_gradient at the burn-in boundary.
#   rlax      google-deepmind/rlax, _src/multistep.py, transforms.py (Apache-2.0)
#             Value rescaling and n-step returns. Changed: rewritten to avoid the
#             dependency; h_inv rearranged for float32 precision.
#   flashbax  instadeepai/flashbax, prioritised trajectory buffer, used as a library.
#   this repo agents/dqn/dqn.py and dqn_eval.py: make_env, epsilon schedule and the
#             evaluation structure. Changed: recurrent evaluator, LSTM state threading.
#             agents/rainbow/rainbow.py: prioritised replay usage and dtype pattern.
#
#   Algorithms follow R2D2 (Kapturowski et al., ICLR 2019), NGU (Badia et al., ICLR
#   2020) and Agent57 (Badia et al., ICML 2020). Deviations are listed in the report.

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
import jaxatari
from jaxatari.wrappers import (
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    LogWrapper,
    FlattenObservationWrapper
)
from agents.agent57.agent57_eval import evaluate   
from rtpt import RTPT
def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None
    if not eval and mods is not None and len(mods) > 0:
        print(f"[WARNING] Training on mods {mods}!")

    def thunk():
        env = jaxatari.make(env_id, mods=mods)
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
    def __call__(self, carry, obs, prev_action, prev_reward, arm=None, prev_intrinsic=None):
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
        # these inputs do not exist and the network is exactly stage 1.

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
    agent must behave exactly like plain R2D2, which is stage 2's gate.
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
 
    Each env remembers the last W episodes it played: which arm it used and what
    undiscounted extrinsic return that episode scored. At the start of every
    episode it picks the arm to play next from that window.
 
    arms/returns are ring buffers over episodes, not steps.
    """
    arms: jnp.ndarray       # (E, W) arm id of each remembered episode
    returns: jnp.ndarray    # (E, W) extrinsic return of that episode
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
 
 
def bandit_update(bandit, arm, episode_return, done):
    """Record (arm, return) for every env whose episode just ended."""
    window = bandit.arms.shape[1]
    idx = bandit.write
    write_one = lambda buf, i, v, d: buf.at[i].set(jnp.where(d, v, buf[i]))
    arms = jax.vmap(write_one)(bandit.arms, idx, arm, done)
    returns = jax.vmap(write_one)(bandit.returns, idx, episode_return, done)
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

    
    carry_e: tuple             # LSTM state at this step, used for burn-in. In r2d2/ngu this is the network's only carry. [SPLIT_Q] specifically Q_e's carry.
    carry_i: tuple             # [SPLIT_Q] Q_i keeps its own carry since it has separate weights, so it can diverge from carry_e after the first step. In r2d2/ngu, this is just set to carry_e as unused padding.


    arm: jnp.ndarray            # which of the 8 (beta, gamma) policies acted
    intrinsic_reward: jnp.ndarray   # curiosity reward for this step
    prev_intrinsic: jnp.ndarray     # [NGU] curiosity reward of the previous step; a network input, stored for the same reason as prev_reward


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
        
def unroll(network, params, carry, seq):
    """Run the network over a time-first sequence. Returns (last carry, q of shape (T, B, A))."""
    def step(carry, x):
        return network.apply(params, carry, *x)

    xs = (seq.obs, seq.prev_action, seq.prev_reward)
    if network.num_arms > 0:
        # [NGU] the arm and the previous curiosity reward are network inputs too
        xs = xs + (seq.arm, seq.prev_intrinsic)
    return jax.lax.scan(step, carry, xs)


def r2d2_loss(params, target_params, batch, probabilities, network, cfg,              
              arm_betas=None, arm_gammas=None,
              next_action=None, reward_key="reward", carry_key="carry_e",double_q=True):

    """batch: TimeStep with arrays shaped (B, BURN_IN + SEQ_LEN, ...).
     probabilities: (B,) sampling probabilities from the prioritised buffer.
     Returns loss and (new priorities (B,), mean Q of taken actions).
     Unchanged for r2d2/ngu when next_action=None, reward_key="reward",
     carry_key="carry_e" (all defaults).

     [SPLIT_Q] next_action: precomputed (T-1, B) greedy action from the
     COMBINED Q. Overrides the double-DQN self-argmax below, so Q_e/Q_i
     never pick their own target action independently.
     reward_key: "reward" or "intrinsic_reward" — which TimeStep field this
     call trains against.
     carry_key: "carry_e" or "carry_i" — which stored LSTM carry to burn in from."""

    burn_in = cfg["BURN_IN_LENGTH"]
    eps = cfg["VALUE_RESCALING_EPSILON"]
 
    data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), batch)      # (L, B, ...)
    

    start = jax.tree.map(lambda c: c[0], getattr(data, carry_key))   # stored state

    burn = jax.tree.map(lambda x: x[:burn_in], data)
    learn = jax.tree.map(lambda x: x[burn_in:], data)
 
    # 1. burn-in: rebuild the memory from the stored state. No learning here.
    online_carry, _ = unroll(network, params, start, burn)
    target_carry, _ = unroll(network, target_params, start, burn)
    online_carry = jax.lax.stop_gradient(online_carry)
 
    # 2. Q-values on the part we learn from
    _, q_online = unroll(network, params, online_carry, learn)          # (T, B, A)
    _, q_target = unroll(network, target_params, target_carry, learn)
    q_target = jax.lax.stop_gradient(q_target)
 
    # 3. double Q: the online net picks the next action, the target net scores it
   
    if next_action is None:
        if double_q:
            # Double-Q ON: online selects, target evaluates.
            next_action = jnp.argmax(q_online[1:], axis=-1)
        else:
            # Double-Q OFF: target selects and target evaluates.
            next_action = jnp.argmax(q_target[1:], axis=-1)

    next_value = jnp.take_along_axis(
        q_target[1:], next_action[..., None], axis=-1
    )[..., 0]
 
    # 4. n-step target, computed in real-return space, then squashed with h
    
    if reward_key == "reward":
        rewards = learn.reward[:-1]
    elif reward_key == "intrinsic_reward":
        rewards = learn.intrinsic_reward[:-1]
    else:
        raise ValueError(f"unknown reward_key: {reward_key}")

    gamma = cfg["GAMMA"]
    if arm_betas is not None:
        # [NGU] each step learns from its own arm's reward mix and discount:
        # r = r_e + beta_arm * r_i, gamma = gamma_arm. Arm 0 has beta = 0 and
        # gamma = GAMMA, so it learns exactly what stage 1 learns.
        arm = learn.arm[:-1]
        rewards = mixed_reward(rewards, learn.intrinsic_reward[:-1], arm_betas[arm])
        gamma = arm_gammas[arm]
        
        #Added elif
    elif arm_gammas is not None:
        # [SPLIT_Q] per-arm discount, same arm_schedule() as stage ngu, no reward mixing
        arm = learn.arm[:-1]
        gamma = arm_gammas[arm]

    
    discounts = gamma * (1.0 - learn.done[:-1].astype(jnp.float32))
    returns = n_step_targets(rewards, discounts, h_inv(next_value, eps), cfg["N_STEP"])
    target = jax.lax.stop_gradient(h(returns, eps))
 
    q_taken = jnp.take_along_axis(q_online[:-1], learn.action[:-1, :, None], axis=-1)[..., 0]
    td = target - q_taken                                               # (T-1, B)
 
    # 5. loss: sum over time per sequence, importance-weighted mean over the batch
    per_seq = 0.5 * jnp.square(td).sum(axis=0)                          # (B,)
    weights = (1.0 / (probabilities + 1e-10)) ** cfg["IMPORTANCE_SAMPLING_EXPONENT"]
    weights = weights / weights.max()
    loss = jnp.mean(weights * per_seq)
 
    # 6. [R2D2] priority = eta * max|TD| + (1 - eta) * mean|TD| over the sequence
    abs_td = jnp.abs(td)
    eta = cfg["PRIORITY_ETA"]
    priorities = eta * abs_td.max(axis=0) + (1.0 - eta) * abs_td.mean(axis=0)
 
    return loss, (jax.lax.stop_gradient(priorities), q_taken.mean())
 





# ---- Split-Q (Agent57 Sec 3.1) -----------------------------------------------

def combine_q(q_e, q_i, beta, eps):
    """Q = Q_e + beta_j * Q_i, on RAW network outputs (h-space). h/h_inv stay
    exactly as in r2d2_loss — applied only to the bootstrap and TD target,
    never here. q_e, q_i: (..., A). beta: (...,) broadcastable against A."""
    return h(h_inv(q_e, eps) + jnp.asarray(beta)[..., None] * h_inv(q_i, eps), eps)


def split_q_priority_from(priorities_e, priorities_i, arm, arm_betas):
    """[SPLIT_Q] priority = priority_e + beta_arm * priority_i.

    The per-network R2D2 priority (eta * max|TD| + (1-eta) * mean|TD|,
    computed inside r2d2_loss) is specified by the paper; combining the two
    networks' priorities into the single value the shared replay buffer
    needs is NOT — this is the project's chosen rule, not a paper formula.

    priorities_e, priorities_i: (B,) — one priority per sampled sequence.
    arm: (B,) — each sequence's fixed arm id (arm is constant across the
    whole sequence, see TimeStep.arm / env_arms in single_run).
    arm_betas: (NUM_ARMS,) from arm_schedule(), same schedule stage ngu uses."""
    beta = arm_betas[arm]
    return priorities_e + beta * priorities_i


def _unroll_online_q(network, params, batch, cfg, carry_key="carry_e"):
    """Burn-in + online unroll only — no target net, no TD/loss/priority math.
    Deliberately duplicates the opening of r2d2_loss (r2d2_loss itself is
    left structurally untouched); used only for the shared-action pre-pass."""
    burn_in = cfg["BURN_IN_LENGTH"]
    data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), batch)
    start = jax.tree.map(lambda c: c[0], getattr(data, carry_key))
    burn = jax.tree.map(lambda x: x[:burn_in], data)
    learn = jax.tree.map(lambda x: x[burn_in:], data)
    online_carry, _ = unroll(network, params, start, burn)
    online_carry = jax.lax.stop_gradient(online_carry)
    _, q_online = unroll(network, params, online_carry, learn)
    return q_online, learn.arm


def compute_shared_next_action(network, params_e, params_i, batch, cfg, arm_betas):
    """[SPLIT_Q] The ONE target action both losses use: argmax(Q_e + beta_j *
    Q_i). Computed once per update, outside any grad tape. Relies on
    env_arms being fixed per env slot for the whole run (single_run:
    env_arms = arange(num_envs) % NUM_ARMS, never resampled)."""
    q_online_e, arm = _unroll_online_q(network, params_e, batch, cfg, carry_key="carry_e")
    q_online_i, _ = _unroll_online_q(network, params_i, batch, cfg, carry_key="carry_i")
    beta = arm_betas[arm]
    q_combined = combine_q(q_online_e, q_online_i, beta, cfg["VALUE_RESCALING_EPSILON"])
    return jax.lax.stop_gradient(jnp.argmax(q_combined[1:], axis=-1))


def update_split_q(agent_state_e, agent_state_i, buffer_state, batch, network, cfg, arm_betas, arm_gammas, replay_buffer):
    """[SPLIT_Q] Reuses r2d2_loss once for Q_e and once for Q_i.
    Replay priority is combined as priority_e + beta_arm * priority_i.
    The replay batch is sampled by the caller so the same batch can also train
    the NGU embedding and RND predictor.
    """
    exp = batch.experience

    shared_next_action = compute_shared_next_action(
        network, agent_state_e.params, agent_state_i.params, exp, cfg, arm_betas)

    def loss_e_fn(params_e):
        return r2d2_loss(params_e, agent_state_e.target_params, exp, batch.probabilities,
                          network, cfg, arm_gammas=arm_gammas, next_action=shared_next_action,
                          reward_key="reward", carry_key="carry_e")

    def loss_i_fn(params_i):
        return r2d2_loss(params_i, agent_state_i.target_params, exp, batch.probabilities,
                          network, cfg, arm_gammas=arm_gammas, next_action=shared_next_action,
                          reward_key="intrinsic_reward", carry_key="carry_i")

    (loss_e, (priorities_e, q_mean_e)), grads_e = jax.value_and_grad(loss_e_fn, has_aux=True)(agent_state_e.params)
    (loss_i, (priorities_i, q_mean_i)), grads_i = jax.value_and_grad(loss_i_fn, has_aux=True)(agent_state_i.params)

    agent_state_e = agent_state_e.apply_gradients(grads=grads_e)
    agent_state_i = agent_state_i.apply_gradients(grads=grads_i)

    priorities = split_q_priority_from(priorities_e, priorities_i, exp.arm[:, 0], arm_betas)
    buffer_state = replay_buffer.set_priorities(buffer_state, batch.indices, priorities)

    sync = (agent_state_e.step % cfg["TARGET_NETWORK_FREQUENCY"]) == 0
    target_params_e = jax.lax.cond(
        sync, lambda _: optax.incremental_update(agent_state_e.params, agent_state_e.target_params, 1.0),
        lambda _: agent_state_e.target_params, operand=None)
    target_params_i = jax.lax.cond(
        sync, lambda _: optax.incremental_update(agent_state_i.params, agent_state_i.target_params, 1.0),
        lambda _: agent_state_i.target_params, operand=None)

    agent_state_e = agent_state_e.replace(target_params=target_params_e)
    agent_state_i = agent_state_i.replace(target_params=target_params_i)
    return agent_state_e, agent_state_i, buffer_state, loss_e, loss_i, q_mean_e, q_mean_i




def single_run(config:dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    stage = config.get("STAGE", "r2d2")
    
    

    assert stage in ("r2d2", "ngu", "split_q", "agent57"), f"unknown STAGE: {stage}"

    ngu = stage == "ngu"
    meta = stage == "agent57"
    # Agent57 uses the split-Q architecture underneath the meta-controller.
    split_q = stage in ("split_q", "agent57")
    intrinsic_enabled = ngu or split_q

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

    sample_len = burn_in + seq_len

    # Getting this dtype wrong is the classic silent failure: a uint8 buffer
    # rounds a normalised 0.47 to 0 and the agent trains on nothing.
    obs_bytes = int(np.prod(obs_shape)) * (1 if config["PIXEL_BASED"] else 4)
    carry_bytes = 2 * hidden * 4
    buffer_gb = config["BUFFER_SIZE"] * (obs_bytes + carry_bytes) / 1e9


    # --- env helpers (copied from dqn.py) -----------------------------------
    @jax.jit
    def vmap_reset(rng):
        obs, state = jax.vmap(env.reset)(rng)
        return obs.reshape(rng.shape[0], *obs_shape), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.reshape(action.shape[0], *obs_shape), state, reward, next_done, info



    # --- network ------------------------------------------------------------

    network = RecurrentQNetwork(
        action_dim=action_dim,
        pixel_based=config["PIXEL_BASED"],
        hidden_size=hidden,
        dueling_units=config["DUELING_UNITS"],
        mlp_width=config.get("MLP_WIDTH", 512),
        mlp_depth=config.get("MLP_DEPTH", 2),
       
        num_arms=config["NUM_ARMS"] if intrinsic_enabled else 0,
    )
    

    ngu_inputs = lambda n: (jnp.zeros((n,), jnp.int32), jnp.zeros((n,), jnp.float32)) if intrinsic_enabled else ()
    
    key, reset_key, net_key = jax.random.split(key, 3)
    dummy_obs, _ = env.reset(reset_key)
    dummy_obs = dummy_obs.reshape(obs_shape)

    
    def _init_params(k):
        return network.init(
            k, RecurrentQNetwork.initial_carry(1, hidden), dummy_obs[None],
            jnp.zeros((1,), jnp.int32),
            jnp.zeros((1,), jnp.float32),
            *ngu_inputs(1),
        )

    if split_q:
        net_key_e, net_key_i = jax.random.split(net_key)
        params_e = _init_params(net_key_e)
        params_i = _init_params(net_key_i)
    else:
        params = _init_params(net_key)    


    # --- replay buffer -----------------------------------------------------
    replay_buffer = fbx.make_prioritised_trajectory_buffer(
        add_batch_size=num_envs,
        sample_batch_size=config["BATCH_SIZE"],       # 64 sequences per update
        sample_sequence_length=sample_len,              
        period=config["SEQUENCE_PERIOD"],             # stride between starts -> 40-step overlap
        min_length_time_axis=max(sample_len, config["LEARNING_STARTS"]// num_envs),                 
        max_length_time_axis=config["BUFFER_SIZE"] // num_envs,
        priority_exponent=config["PRIORITY_EXPONENT"],
    )
    key, reset_key = jax.random.split(key)
    dummy_obs, _ = env.reset(reset_key)
    dummy_obs = dummy_obs.reshape(obs_shape)
    dummy_timestep = TimeStep(
        obs=dummy_obs,
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros((), dtype=jnp.float32),
        done=jnp.zeros((), dtype=jnp.bool_),
        prev_action=jnp.zeros((), dtype=jnp.int32),
        
        prev_reward=jnp.zeros((), dtype=jnp.float32),
        carry_e=jax.tree.map(lambda c: c[0], RecurrentQNetwork.initial_carry(1, hidden)),
        carry_i=jax.tree.map(lambda c: c[0], RecurrentQNetwork.initial_carry(1, hidden)),
        

        arm=jnp.zeros((), dtype=jnp.int32),
        intrinsic_reward=jnp.zeros((), dtype=jnp.float32),
        prev_intrinsic=jnp.zeros((), dtype=jnp.float32),
    )
    buffer_state = replay_buffer.init(dummy_timestep)
 # --- NGU: side networks and arms ------------------------------------------
    # Stage 2 adds three things to stage 1, all created only when ngu is True:
    #   arms        a fixed (beta, gamma) per env: env i plays arm i mod NUM_ARMS
    #   embedding   observation -> 32-d controllable state, for episodic novelty
    #   RND         frozen random target + trained predictor, for lifelong novelty
    ngu_state = {}
   
    if intrinsic_enabled:

        arm_betas, arm_gammas = arm_schedule(
            config["NUM_ARMS"], config["BETA_MAX"], config["GAMMA_MAX"], config["GAMMA_MIN"])
        env_arms = jnp.arange(num_envs, dtype=jnp.int32) % config["NUM_ARMS"]
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
        ngu_state = {
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
     # --- collection ---------------------------------------------------------
     # One call = TRAIN_FREQUENCY steps in every env. Unlike dqn, the whole chunk
     # goes into the buffer at once: the trajectory buffer keeps each env's steps
     # in order and cuts 120-step sequences out of them when sampling.
    total_timesteps = config["TOTAL_TIMESTEPS"]
    steps_per_call = config["TRAIN_FREQUENCY"]
    
    
    def collect(params, act_state, rng, side=None, params_i=None):

        def act_step(act_state, rng):
            if meta:
                env_state, obs, lstm_e, lstm_i, prev_action, prev_reward, ngu_act, bandit, arms, global_step = act_state
            elif split_q:
                env_state, obs, lstm_e, lstm_i, prev_action, prev_reward, ngu_act, global_step = act_state
                arms=env_arms
            else:
                env_state, obs, lstm, prev_action, prev_reward, ngu_act, global_step = act_state
                arms=env_arms if intrinsic_enabled else None


            action_rng, explore_rng, bandit_rng = jax.random.split(rng,3)
            # same schedule as dqn.py
            epsilon = jnp.interp(
                global_step,
                jnp.array([0, config["EXPLORATION_FRACTION"] * total_timesteps]),
                jnp.array([config["START_E"], config["END_E"]]),
            )

            
            if intrinsic_enabled:
                memory, prev_intrinsic, rnd_stats = ngu_act
                net_extra = (arms, prev_intrinsic)
            else:
                net_extra = ()
           
            

            
            if split_q:
                next_lstm_e, q_e = network.apply(params, lstm_e, obs, prev_action, prev_reward, *net_extra)
                next_lstm_i, q_i = network.apply(params_i, lstm_i, obs, prev_action, prev_reward, *net_extra)
                q_values = combine_q(q_e, q_i, arm_betas[arms], config["VALUE_RESCALING_EPSILON"])
            else:
                next_lstm, q_values = network.apply(params, lstm, obs, prev_action, prev_reward, *net_extra)            
            

            
            greedy_actions = q_values.argmax(axis=-1)
            random_actions = jax.random.randint(action_rng, (num_envs,), 0, action_dim)
            explore_mask = jax.random.uniform(explore_rng, (num_envs,)) < epsilon
            actions = jnp.where(explore_mask, random_actions, greedy_actions)
            next_obs, env_state, rewards, next_done, info = vmap_step(env_state, actions)
            rewards = rewards.astype(jnp.float32)   # env clips to int32; the buffer and loss use float32
            
            
            if intrinsic_enabled:

                emb = emb_model.apply(side["emb"].params, next_obs, method=EmbeddingTrainer.embed)
                r_episodic, memory = episodic_reward(memory, emb, next_done, config)
                err = rnd_error(side["rnd"].params, side["rnd_target"], rnd_net, next_obs)
                rnd_stats = update_running_stats(rnd_stats, err)
                r_int = intrinsic_reward(r_episodic, rnd_modulator(err, rnd_stats, config))
                arm_used, prev_int_used = arms, prev_intrinsic
            else:
                r_int = jnp.zeros_like(rewards)
                arm_used, prev_int_used = jnp.zeros_like(actions), jnp.zeros_like(rewards)
            
            timestep = TimeStep(
                obs=obs,
                action=actions,
                reward=rewards,
                done=next_done,
                prev_action=prev_action,
                prev_reward=prev_reward,

                
                carry_e=(lstm_e if split_q else lstm),
                carry_i=(lstm_i if split_q else lstm),


                arm=arm_used,        
                intrinsic_reward=r_int,
                prev_intrinsic=prev_int_used,

            )
            # Inputs for the next step. Where the episode just ended, -1 tells
            # the network to start from a zero carry.
            prev_action = jnp.where(next_done, -1, actions)
            prev_reward = jnp.where(next_done, 0.0, rewards)
            
            
            if intrinsic_enabled:
                ngu_act = (memory, jnp.where(next_done, 0.0, r_int), rnd_stats)

            if meta:
                bandit = bandit_update(bandit,arms, info["returned_episode_returns"].astype(jnp.float32),next_done,)
                new_arms = bandit_select(bandit, config["NUM_ARMS"], bandit_rng, config,)
                # Only environments whose episode ended switch arm.
                arms = jnp.where(next_done, new_arms, arms)

            if meta:
                act_state = (env_state, next_obs, next_lstm_e, next_lstm_i, prev_action, prev_reward, ngu_act, bandit, arms, global_step + num_envs, )
            elif split_q:
                act_state = (env_state, next_obs, next_lstm_e, next_lstm_i, prev_action, prev_reward, ngu_act, global_step + num_envs,)
            else:
                act_state = (env_state, next_obs, next_lstm, prev_action, prev_reward, ngu_act, global_step + num_envs, )
            
            return act_state, (timestep, info)
        
        rngs = jax.random.split(rng, steps_per_call)
        act_state, (traj, infos) = jax.lax.scan(act_step, act_state, rngs)
        # scan stacks time first (T, num_envs, ...); the buffer wants (num_envs, T, ...)
        traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)
        return act_state, traj, infos
    
    key, reset_key = jax.random.split(key)
    obs, env_state = vmap_reset(jax.random.split(reset_key, num_envs))

    _ngu_act0 =(
        init_episodic_memory(num_envs, config["EPISODIC_MEMORY_SIZE"], config.get("EMBEDDING_DIM", 32)),
        jnp.zeros((num_envs,), jnp.float32), init_running_stats(),
    ) if intrinsic_enabled else ()

    if meta:
        bandit0 = init_bandit(num_envs, config.get("UCB_WINDOW", 160), config["NUM_ARMS"],)
        
        arms0 = jnp.zeros((num_envs,), dtype=jnp.int32)
        act_state = (env_state, obs, RecurrentQNetwork.initial_carry(num_envs, hidden),RecurrentQNetwork.initial_carry(num_envs, hidden), jnp.full((num_envs,), -1, jnp.int32), jnp.zeros((num_envs,), jnp.float32), _ngu_act0, bandit0, arms0, jnp.array(0, jnp.int32), )
    elif split_q:
        act_state = (env_state, obs, RecurrentQNetwork.initial_carry(num_envs, hidden), RecurrentQNetwork.initial_carry(num_envs, hidden), jnp.full((num_envs,), -1, jnp.int32), jnp.zeros((num_envs,), jnp.float32), _ngu_act0, jnp.array(0, jnp.int32), )
    else:
        act_state = (env_state, obs, RecurrentQNetwork.initial_carry(num_envs, hidden), jnp.full((num_envs,), -1, jnp.int32), jnp.zeros((num_envs,), jnp.float32), _ngu_act0, jnp.array(0, jnp.int32), )



    

    print(f"[agent57] stage={stage} env={config['ENV_ID']} "f"{'pixel' if config['PIXEL_BASED'] else 'oc'}")
    print(f"[agent57] action_dim={action_dim} obs_shape={obs_shape} " f"obs_bytes={obs_bytes}")
    print(f"[agent57] buffer: {config['BUFFER_SIZE']} transitions = {buffer_gb:.2f} GB")
    print(f"[agent57] buffer storage: {buffer_state.experience.obs.shape} "f"dtype={buffer_state.experience.obs.dtype}")

        # main.py expects a dict back from every agent. nan means "no score yet";

    # --- optimizer and train state ------------------------------------------
    tx = optax.adam(
        learning_rate=config["LEARNING_RATE"],
        b1=config["ADAM_B1"],
        b2=config["ADAM_B2"],
        eps=config["ADAM_EPS"],
    )
    
    if split_q:
        agent_state_e = Agent57TrainState.create(
            apply_fn=network.apply, params=params_e, target_params=jax.tree.map(jnp.copy, params_e), tx=tx)
        agent_state_i = Agent57TrainState.create(
            apply_fn=network.apply, params=params_i, target_params=jax.tree.map(jnp.copy, params_i), tx=tx)
    else:
        agent_state = Agent57TrainState.create(
            apply_fn=network.apply, params=params, target_params=jax.tree.map(jnp.copy, params), tx=tx)
        loss_fn = partial(r2d2_loss, network=network, cfg=config,double_q=config.get("DOUBLE_Q", True),
                           **({"arm_betas": arm_betas, "arm_gammas": arm_gammas} if ngu else {}))


    side_frames = config.get("NGU_TRAIN_FRAMES", 5)   # [NGU] side networks train on the last 5 frames

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

    def update(agent_state, buffer_state, rng, side):
        """Sample 64 sequences, take one gradient step, write priorities back."""
        batch = replay_buffer.sample(buffer_state, rng)
        (loss, (priorities, q_mean)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            agent_state.params,
            agent_state.target_params,
            batch.experience,
            batch.probabilities,
        )
        agent_state = agent_state.apply_gradients(grads=grads)
        buffer_state = replay_buffer.set_priorities(buffer_state, batch.indices, priorities)

        # [R2D2] target <- online every 2500 LEARNER steps (apply_gradients counts them)
        sync = (agent_state.step % config["TARGET_NETWORK_FREQUENCY"]) == 0
        target_params = jax.lax.cond(
            sync,
            lambda _: optax.incremental_update(agent_state.params, agent_state.target_params, 1.0),
            lambda _: agent_state.target_params,
            operand=None,
        )
        if ngu:
            side, emb_acc, rnd_l = update_side(side, batch.experience)
        else:
            emb_acc, rnd_l = jnp.float32(0.0), jnp.float32(0.0) 
        return (
            agent_state.replace(target_params=target_params),
            buffer_state,
            side,
            (loss, q_mean, emb_acc, rnd_l),
        )
    # --- play TRAIN_FREQUENCY steps, then learn once the buffer is warm ------
    def train_step(carry, _):
        if split_q:
            agent_state_e, agent_state_i, buffer_state, act_state, rng, side = carry
            rng, collect_rng, update_rng = jax.random.split(rng, 3)
            act_state, traj, infos = collect(agent_state_e.params,act_state,collect_rng,side,params_i=agent_state_i.params,)
            buffer_state = replay_buffer.add(buffer_state, traj)
            
            def do_update(a_e, a_i, b, s):
                # Sample ONCE. The same replay batch trains:
                #   1. extrinsic Q network
                #   2. intrinsic Q network
                #   3. NGU embedding
                #   4. RND predictor
                batch = replay_buffer.sample(b, update_rng)

                a_e, a_i, b, loss_e, loss_i, q_mean_e, q_mean_i = update_split_q(
                    a_e,
                    a_i,
                    b,
                    batch,
                    network,
                    config,
                    arm_betas,
                    arm_gammas,
                    replay_buffer,
                )

                s, emb_acc, rnd_l = update_side(s, batch.experience)

                return (a_e,a_i,b,s,loss_e,loss_i,q_mean_e,q_mean_i,emb_acc,rnd_l,)

            def no_update(a_e, a_i, b, s):
                z = jnp.float32(0.0)
                return (a_e, a_i, b, s, z, z, z, z, z, z,)
            (agent_state_e, agent_state_i, buffer_state, side, loss_e, loss_i, q_mean_e, q_mean_i, emb_acc, rnd_l,) = jax.lax.cond(replay_buffer.can_sample(buffer_state), do_update, no_update, agent_state_e, agent_state_i, buffer_state, side,
            )
            
            r_int = traj.intrinsic_reward
            metrics = (loss_e, loss_i, q_mean_e, q_mean_i, emb_acc, rnd_l, r_int.mean(), r_int.max(), )
            return (agent_state_e, agent_state_i, buffer_state, act_state, rng, side,), (infos, metrics)
        else:
            agent_state, buffer_state, act_state, rng, side = carry
            rng, collect_rng, update_rng = jax.random.split(rng, 3)
            act_state, traj, infos = collect(agent_state.params, act_state, collect_rng, side)
            buffer_state = replay_buffer.add(buffer_state, traj)
            zeros = (jnp.float32(0.0),) * 4
            agent_state, buffer_state, side, metrics = jax.lax.cond(
                replay_buffer.can_sample(buffer_state),
                lambda a, b, s: update(a, b, update_rng, s),
                lambda a, b, s: (a,b,s,zeros),
                agent_state, buffer_state, side)
            r_int = traj.intrinsic_reward
            metrics = metrics + (r_int.mean(), r_int.max())
            return (agent_state, buffer_state, act_state, rng, side), (infos, metrics)

    @jax.jit
    def scanned_steps(carry):
        return jax.lax.scan(train_step, carry, None, length=config["SCAN_STEPS"])

    # --- training loop -------------------------------------------------------
    run_name = f"{config['ENV_ID']}_{config.get('EXP_NAME', 'agent57')}_{stage}_{'pixel' if config['PIXEL_BASED'] else 'oc'}_{config['SEED']}"
    wandb.init(
        project=config.get("PROJECT", "jaxtari-blines"),
        entity=config.get("ENTITY", None),
        config=config,
        name=run_name, 
        save_code=True,
    )
    wandb.define_metric("*", step_metric="charts/global_step")

    steps_per_iteration = num_envs * steps_per_call * config["SCAN_STEPS"]
    
    if split_q:
        carry = (agent_state_e, agent_state_i, buffer_state, act_state, key, ngu_state)
    else:
        carry = (agent_state, buffer_state, act_state, key, ngu_state)

    print(f"[agent57] compiling ({config['SCAN_STEPS']} scan steps)...")
    compile_start = time.perf_counter()
    _ = jax.block_until_ready(scanned_steps(carry))
    print(f"[agent57] compile time: {time.perf_counter() - compile_start:.1f}s")

    rtpt = RTPT(
        name_initials=config["NAME_INITIALS"],
        experiment_name=run_name,
        max_iterations=max(1, config["TOTAL_TIMESTEPS"] // steps_per_iteration),
    )
    rtpt.start()

    global_step, avg_return = 0, float("nan")
    run_start = time.perf_counter()
    print(f"[agent57] training for {config['TOTAL_TIMESTEPS']} steps...")
    act_state_idx = 3 if split_q else 2
    while global_step < config["TOTAL_TIMESTEPS"]:
        rtpt.step()
        iteration_start = time.perf_counter()
        carry, (infos, metrics) = jax.block_until_ready(scanned_steps(carry))
        global_step = int(carry[act_state_idx][-1])

        avg_return = float(infos["returned_episode_returns"][-1].mean())
        avg_length = float(infos["returned_episode_lengths"][-1].mean())
        sps = int(steps_per_iteration / (time.perf_counter() - iteration_start))
        if split_q:
            loss_e, loss_i, q_mean_e, q_mean_i, emb_acc, rnd_l, r_int_mean, r_int_max = metrics
            td_loss_e, td_loss_i = float(loss_e[-1]), float(loss_i[-1])
            print(
                f"[agent57][split_q] step {global_step} | return {avg_return:.2f} | length {avg_length:.0f} "
                f"| loss_e {td_loss_e:.4f} | loss_i {td_loss_i:.4f} "
                f"| q_e {float(q_mean_e[-1]):.3f} | q_i {float(q_mean_i[-1]):.3f} | SPS {sps} "
                f"| total SPS {int(global_step / (time.perf_counter() - run_start))} "
                f"| r_int {float(r_int_mean[-1]):.3f} (max {float(r_int_max[-1]):.2f}) "
                f"| emb acc {float(emb_acc[-1]):.2f} | rnd {float(rnd_l[-1]):.4f}"
            )
            
            wandb.log({
                "charts/avg_episodic_return": avg_return,
                "charts/avg_episodic_length": avg_length,
                "losses/td_loss_e": td_loss_e,
                "losses/td_loss_i": td_loss_i,
                "losses/q_values_e": float(q_mean_e[-1]),
                "losses/q_values_i": float(q_mean_i[-1]),
                "charts/SPS": sps,
                "charts/global_step": global_step,
                "ngu/intrinsic_reward_mean": float(r_int_mean[-1]),
                "ngu/intrinsic_reward_max": float(r_int_max[-1]),
                "ngu/embedding_accuracy": float(emb_acc[-1]),
                "ngu/rnd_loss": float(rnd_l[-1]),
                }, step=global_step)
        else:
            loss, q_mean, emb_acc, rnd_l, r_int_mean, r_int_max = metrics
            td_loss, q_val = float(loss[-1]), float(q_mean[-1])
            print(
                f"[agent57] step {global_step} | return {avg_return:.2f} | length {avg_length:.0f} "
                f"| loss {td_loss:.4f} | q {q_val:.3f} | SPS {sps} "
                f"| total SPS {int(global_step / (time.perf_counter() - run_start))}"
                + (f" | r_int {float(r_int_mean[-1]):.3f} (max {float(r_int_max[-1]):.2f})"
                   f" | emb acc {float(emb_acc[-1]):.2f} | rnd {float(rnd_l[-1]):.4f}" if ngu else "")
            )
            wandb.log(
                {
                    "charts/avg_episodic_return": avg_return, "charts/avg_episodic_length": avg_length,
                    "losses/td_loss": td_loss, "losses/q_values": q_val,
                    "charts/SPS": sps, "charts/global_step": global_step,
                    **({"ngu/intrinsic_reward_mean": float(r_int_mean[-1]),
                        "ngu/intrinsic_reward_max": float(r_int_max[-1]),
                        "ngu/embedding_accuracy": float(emb_acc[-1]),
                        "ngu/rnd_loss": float(rnd_l[-1])} if ngu else {}),
                },
                step=global_step,
            )

    # --- save and evaluate ---------------------------------------------------
    # Training returns use clipped rewards; this is the unclipped number, the one
    # comparable to the DQN / Rainbow eval results.
    
    if split_q:
        model_path = (
            f'{config.get("SAVE_PATH", "./models")}/{run_name}/'
            f'{config["EXP_NAME"]}_{global_step}_{int(time.time())}.split_q_model'
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes({
                "config": config, "params_e": carry[0].params, "params_i": carry[1].params,
            }))
        print(f"[agent57][split_q] model saved to {model_path}")

        side_final = carry[5]
        rnd_stats_final = carry[3][6][2]   # act_state[6] = ngu_act tuple, [2] = rnd_stats
        side_path = model_path.replace(".split_q_model", ".ngu_side")
        with open(side_path, "wb") as f:
            f.write(flax.serialization.to_bytes({
                "emb": side_final["emb"].params, "rnd": side_final["rnd"].params,
                "rnd_target": side_final["rnd_target"], "rnd_stats": rnd_stats_final,
            }))
        print(f"[agent57][split_q] NGU side networks saved to {side_path}")

        def eval_intrinsic_fn(memory, obs, done):
            emb = emb_model.apply(side_final["emb"].params, obs, method=EmbeddingTrainer.embed)
            r_episodic, memory = episodic_reward(memory, emb, done, config)
            err = rnd_error(side_final["rnd"].params, side_final["rnd_target"], rnd_net, obs)
            return intrinsic_reward(r_episodic, rnd_modulator(err, rnd_stats_final, config)), memory

        eval_intrinsic = (
            eval_intrinsic_fn,
            lambda n: init_episodic_memory(n, config["EPISODIC_MEMORY_SIZE"], config.get("EMBEDDING_DIM", 32)),
        )

        episodic_returns, _ = evaluate(   # SAME call as r2d2/ngu — evaluate() detects split_q itself
            model_path,
            partial(make_env, mods=list(config.get("TRAIN_MODS", [])), pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config.get("NATIVE_DOWNSCALING", True), eval=True),
            config["ENV_ID"], eval_episodes=config.get("EVAL_EPISODES", 10), Model=RecurrentQNetwork,
            network_kwargs=dict(
                pixel_based=config["PIXEL_BASED"], hidden_size=hidden, dueling_units=config["DUELING_UNITS"],
                mlp_width=config.get("MLP_WIDTH", 512), mlp_depth=config.get("MLP_DEPTH", 2),
                num_arms=config["NUM_ARMS"],
            ),
            seed=config["SEED"] + 42, intrinsic=eval_intrinsic,
        )
        eval_return = float(jnp.mean(episodic_returns))
        print(f"[agent57][split_q] eval return {eval_return:.2f} (train return {avg_return:.2f})")
        wandb.log({"eval/episodic_return": eval_return}, step=global_step)
        wandb.finish()
        return {"default": eval_return}
    else:
        model_path = (
            f'{config.get("SAVE_PATH", "./models")}/{run_name}/'
            f'{config["EXP_NAME"]}_{global_step}_{int(time.time())}.cleanrl_model'
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes([config, carry[0].params]))
        print(f"[agent57] model saved to {model_path}")

        eval_intrinsic = None
        if ngu:
            side, rnd_stats_final = carry[4], carry[2][5][2]
            side_path = model_path.replace(".cleanrl_model", ".ngu_side")
            with open(side_path, "wb") as f:
                f.write(flax.serialization.to_bytes({
                    "emb": side["emb"].params, "rnd": side["rnd"].params,
                    "rnd_target": side["rnd_target"], "rnd_stats": rnd_stats_final,
                }))
            print(f"[agent57] NGU side networks saved to {side_path}")

            def eval_intrinsic_fn(memory, obs, done):
                emb = emb_model.apply(side["emb"].params, obs, method=EmbeddingTrainer.embed)
                r_episodic, memory = episodic_reward(memory, emb, done, config)
                err = rnd_error(side["rnd"].params, side["rnd_target"], rnd_net, obs)
                return intrinsic_reward(r_episodic, rnd_modulator(err, rnd_stats_final, config)), memory

            eval_intrinsic = (
                eval_intrinsic_fn,
                lambda n: init_episodic_memory(n, config["EPISODIC_MEMORY_SIZE"], config.get("EMBEDDING_DIM", 32)),
            )

        episodic_returns, _ = evaluate(
            model_path,
            partial(make_env, mods=list(config.get("TRAIN_MODS", [])), pixel_based=config["PIXEL_BASED"],
                    native_downscaling=config.get("NATIVE_DOWNSCALING", True), eval=True),
            config["ENV_ID"], eval_episodes=config.get("EVAL_EPISODES", 10), Model=RecurrentQNetwork,
            network_kwargs=dict(
                pixel_based=config["PIXEL_BASED"], hidden_size=hidden, dueling_units=config["DUELING_UNITS"],
                mlp_width=config.get("MLP_WIDTH", 512), mlp_depth=config.get("MLP_DEPTH", 2),
                num_arms=config["NUM_ARMS"] if ngu else 0,
            ),
            seed=config["SEED"] + 42, intrinsic=eval_intrinsic,
        )
        eval_return = float(jnp.mean(episodic_returns))
        print(f"[agent57] eval return {eval_return:.2f} (train return {avg_return:.2f})")
        wandb.log({"eval/episodic_return": eval_return}, step=global_step)
        wandb.finish()
        return {"default": eval_return}
