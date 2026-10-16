# PPO for continual reinforcement learning: one agent trained on a sequence of
# tasks (a game plus mods), with one of
#   none     plain fine-tuning (the baseline)
#   ewc      online Elastic Weight Consolidation (Kirkpatrick et al., 2017;
#            online variant: Schwarz et al., 2018, https://arxiv.org/abs/1805.06370)
#   agem     Averaged Gradient Episodic Memory (Chaudhry et al., 2019)
#   packnet  iterative pruning, one weight subset per task (Mallya & Lazebnik, 2018)
# After each task, every task seen so far is evaluated, giving average performance,
# backward transfer and forgetting.
#
# PPO itself is agents/ppo/ppo.py (networks, rollout, GAE, loss, chunked training).
import os
import random
import time
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flashbax as fbx
import wandb
from flax.training.train_state import TrainState
from jax.flatten_util import ravel_pytree
import jaxtari
from jaxtari.wrappers import NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper, AtariWrapper, LogWrapper, FlattenObservationWrapper
from jaxtari import spaces
from agents.ppo.ppo import Network, MLP_Network, Actor, Critic, AgentParams, Storage
from agents.ppo_crl.ppo_crl_eval import make_evaluator
from agents.run_utils import plan_chunks
from rtpt import RTPT

CL_METHODS = ("none", "ewc", "agem", "packnet")


def make_env(env_id, mods=[], pixel_based=True, native_downscaling=True, eval=False, full_action_space=False):
    assert mods is None or isinstance(mods, list), "mods must be None or a list of strings"
    if mods is not None and len(mods) == 0:
        mods = None

    def thunk():
        env = jaxtari.make(env_id, mods=mods)
        env = AtariWrapper(
                env,
                sticky_actions=0.0,
                episodic_life=not eval, # only active during training
                first_fire=True,
                noop_max=30,
                # several games share one actor head: all 18 actions
                full_action_space=full_action_space,
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


# ---------------------------------------------------------
# CONTINUAL LEARNING FUNCTIONS
# ---------------------------------------------------------

def init_cl_state(cl_method, params):
    """Arrays the CL method carries across tasks (a pytree, passed into the jitted chunk)."""
    restricted = {"network": params.network_params, "actor": params.actor_params}
    zeros = lambda: jax.tree.map(jnp.zeros_like, restricted)
    if cl_method == "ewc":
        return {"fisher": zeros(), "params": zeros()}
    if cl_method == "packnet":
        # all parameters are free initially (task id 0)
        return {
            "param_task_ids": jax.tree.map(lambda p: jnp.zeros_like(p, dtype=jnp.int32), restricted),
            "current_task_idx": jnp.array(1, dtype=jnp.int32),
            "is_retraining": jnp.array(False),
        }
    return {}


def ewc_penalty(params, cl_state):
    """[EWC] 0.5 * sum F * (theta - theta*)^2 over the shared network and the actor,
    averaged over the number of restricted parameters (hence the large EWC_LAMBDA)."""
    current = {"network": params.network_params, "actor": params.actor_params}
    penalties = jax.tree.map(lambda p, p_star, f: (f * (p - p_star) ** 2).sum(),
                             current, cl_state["params"], cl_state["fisher"])
    num_restrict_params = sum(x.size for x in jax.tree.leaves(cl_state["params"]))
    return 0.5 * sum(jax.tree.leaves(penalties)) / num_restrict_params


def agem_project(grads, ref_grads):
    """[A-GEM] if the update would increase the memory loss (g . g_ref < 0), project it:
    g - (g . g_ref) / (g_ref . g_ref) * g_ref."""
    flat_current, unravel_fn = ravel_pytree(grads)
    flat_ref, _ = ravel_pytree(ref_grads)
    dot_product = jnp.dot(flat_current, flat_ref)
    ref_mag = jnp.dot(flat_ref, flat_ref) + 1e-12
    projected = flat_current - (dot_product / ref_mag) * flat_ref
    return unravel_fn(jnp.where(dot_product < 0, projected, flat_current))


def packnet_grad_mask(grads, cl_state):
    """[PackNet] only weights of the current task (or still free ones, before pruning)
    are trained; weights owned by earlier tasks are frozen. The critic is not masked."""
    current_idx = cl_state["current_task_idx"]

    def apply_grad_mask(grad, task_ids):
        active_mask = jnp.logical_or(
            task_ids == current_idx,
            jnp.logical_and(task_ids == 0, jnp.logical_not(cl_state["is_retraining"]))
        )
        return grad * active_mask

    return grads._replace(
        network_params=jax.tree.map(apply_grad_mask, grads.network_params, cl_state["param_task_ids"]["network"]),
        actor_params=jax.tree.map(apply_grad_mask, grads.actor_params, cl_state["param_task_ids"]["actor"]),
    )


def packnet_prune(params, task_ids, current_task_idx, prune_fraction):
    """[PackNet] layer by layer, prune (zero) the smallest prune_fraction of the free
    weights; the free weights that are kept now belong to the current task. Biases
    are never pruned and all belong to task 1."""
    flat_params_with_paths, tree_def = jax.tree_util.tree_flatten_with_path(params)
    flat_ids, _ = jax.tree_util.tree_flatten(task_ids)

    new_flat_params, new_flat_ids = [], []
    for (path, param_layer), id_layer in zip(flat_params_with_paths, flat_ids):
        path_str = "".join([str(p.key) for p in path])
        if "bias" in path_str.lower():
            new_flat_params.append(param_layer)
            new_flat_ids.append(jnp.where(id_layer == 0, 1, id_layer))
            continue

        free_mask = (id_layer == 0)
        if not jnp.any(free_mask) or prune_fraction == 0.0:
            # nothing to prune (always the case on the last task): the free weights go to the current task
            new_flat_params.append(param_layer)
            new_flat_ids.append(jnp.where(free_mask, current_task_idx, id_layer))
            continue

        # threshold over the free weights only (frozen ones are NaN for nanquantile)
        threshold = jnp.nanquantile(jnp.where(free_mask, jnp.abs(param_layer), jnp.nan), prune_fraction)
        # strict: weights that never received a gradient (e.g. of always-zero OC inputs)
        # stay exactly 0; with >= they would all be kept whenever they make up more than
        # prune_fraction of the free weights, leaving nothing for the next tasks
        keep_mask = jnp.abs(param_layer) > threshold
        new_flat_ids.append(jnp.where(jnp.logical_and(free_mask, keep_mask), current_task_idx, id_layer))
        new_flat_params.append(jnp.where(jnp.logical_and(free_mask, ~keep_mask), 0.0, param_layer))

    return jax.tree_util.tree_unflatten(tree_def, new_flat_params), jax.tree_util.tree_unflatten(tree_def, new_flat_ids)


def packnet_inference_mask(params, cl_state, eval_task_idx):
    """[PackNet] evaluate task j with the weights of tasks 1..j only (plus the free
    weights while the current task has not been pruned yet)."""
    def mask(param, task_ids):
        valid_mask = jnp.logical_and(task_ids > 0, task_ids <= eval_task_idx)
        if eval_task_idx == int(cl_state["current_task_idx"]) and not bool(cl_state["is_retraining"]):
            valid_mask = jnp.logical_or(valid_mask, task_ids == 0)
        return param * valid_mask

    return (jax.tree.map(mask, params.network_params, cl_state["param_task_ids"]["network"]),
            jax.tree.map(mask, params.actor_params, cl_state["param_task_ids"]["actor"]))

# ---------------------------------------------------------


def single_run(config: dict):
    config = {k.upper(): v for k, v in config.items() if k != "alg"}
    pixel_based = config["PIXEL_BASED"]
    cl_method = config["CL_METHOD"].lower()
    assert cl_method in CL_METHODS, f"unknown CL_METHOD {cl_method}, expected one of {CL_METHODS}"

    tasks = list(config["TASKS"])
    num_tasks = len(tasks)
    # mods per task: TASK_TRAIN_MODS[i] is the list trained on in task i,
    # TASK_EVAL_MODS[i] the list of mod lists it is evaluated on (default: its train mods)
    task_train_mods = [list(m) for m in (config.get("TASK_TRAIN_MODS") or [])]
    task_train_mods += [[] for _ in range(num_tasks - len(task_train_mods))]
    task_eval_mods = [[list(m) for m in mods] for mods in (config.get("TASK_EVAL_MODS") or [])]
    task_eval_mods += [[task_train_mods[i]] for i in range(len(task_eval_mods), num_tasks)]

    run_name = f"{'-'.join(tasks)}_{config['EXP_NAME']}_{'pixel' if pixel_based else 'oc'}_{cl_method}_{config['SEED']}"
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
    num_steps = config["NUM_STEPS"]
    num_minibatches = config["NUM_MINIBATCHES"]
    update_epochs = config["UPDATE_EPOCHS"]
    batch_size = num_envs * num_steps
    scan_steps, num_chunks, steps_per_chunk = plan_chunks(config["TIMESTEPS_PER_TASK"], batch_size, config.get("SCAN_STEPS", 1000))
    num_updates = scan_steps * num_chunks   # per task

    # several games: one actor head over the full action space
    multi_game = len(set(tasks)) > 1
    env_kwargs = dict(pixel_based=pixel_based, native_downscaling=config.get("NATIVE_DOWNSCALING", True),
                      full_action_space=multi_game)
    train_envs = [make_env(t, task_train_mods[i], eval=False, **env_kwargs)() for i, t in enumerate(tasks)]
    for env in train_envs:
        assert isinstance(env.action_space(), spaces.Discrete), "only discrete action space is supported"
    action_dim = train_envs[0].action_space().n

    # Object-centric observations differ in size between games: zero-pad all of
    # them to the largest one, so one network fits every task.
    if pixel_based:
        obs_shape = (4, 84, 84)
    else:
        obs_shape = (max(env.observation_space().shape[0] for env in train_envs),)

    def prepare_obs(obs):
        """One env's observation -> network input (drops the pixel channel, pads OC)."""
        if pixel_based:
            return obs.reshape(obs_shape)
        return jnp.pad(obs, (0, obs_shape[0] - obs.shape[0]))

    def linear_schedule(count):
        # anneal learning rate linearly over each task (the optimizer is reset per task)
        # after one update which contains (NUM_MINIBATCHES * UPDATE_EPOCHS) gradient steps
        frac = 1.0 - (count // (num_minibatches * update_epochs)) / num_updates
        return config["LEARNING_RATE"] * frac

    network = Network() if pixel_based else MLP_Network()
    actor = Actor(action_dim=action_dim)
    critic = Critic()

    # same root split as ppo
    key, init_key = jax.random.split(key, 2)
    network_key, actor_key, critic_key = jax.random.split(init_key, 3)
    network_params = network.init(network_key, jnp.zeros((1, *obs_shape)))
    hidden_example = network.apply(network_params, jnp.zeros((1, *obs_shape)))
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params=network_params,
            actor_params=actor.init(actor_key, hidden_example),
            critic_params=critic.init(critic_key, hidden_example),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LEARNING_RATE"], eps=1e-5
            ),
        ),
    )
    cl_state = init_cl_state(cl_method, agent_state.params)

    # [A-GEM] episodic memory: SAMPLES_PER_TASK (obs, action, return) per finished task
    if cl_method == "agem":
        samples_per_task = min(batch_size, config["SAMPLES_PER_TASK"])
        agem_buffer = fbx.make_item_buffer(
            max_length=num_tasks * samples_per_task,
            min_length=config["AGEM_MINIBATCH_SIZE"],
            sample_batch_size=config["AGEM_MINIBATCH_SIZE"],
            add_batches=True,
        )
        cl_state = {"buffer_state": agem_buffer.init({
            "observations": jnp.zeros(obs_shape, jnp.uint8 if pixel_based else jnp.float32),
            "actions": jnp.zeros((), dtype=jnp.int32),
            "value_targets": jnp.zeros((), dtype=jnp.float32),
        })}
        print(f"[ppo_crl] A-GEM buffer length: {num_tasks * samples_per_task}")

    def get_action_and_value(params, obs, key):
        """sample action, calculate value and logprob"""
        hidden = network.apply(params.network_params, obs)
        logits = actor.apply(params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(params.critic_params, hidden)
        return action, logprob, value.squeeze(1), key

    def get_action_and_value2(params, x, action):
        """calculate value, logprob of supplied `action`, and entropy"""
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, cl_state):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if config["NORM_ADV"]:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - config["CLIP_COEF"], 1 + config["CLIP_COEF"])
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - config["ENT_COEF"] * entropy_loss + v_loss * config["VF_COEF"]
        cl_loss = ewc_penalty(params, cl_state) if cl_method == "ewc" else jnp.float32(0.0)
        loss = loss + config.get("EWC_LAMBDA", 0.0) * cl_loss
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl), cl_loss)

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    def agem_ref_grads(params, mem_obs, mem_actions, mem_returns):
        """[A-GEM] reference gradient on a memory batch: behavioural cloning of the stored
        actions plus regression of the critic on the stored returns."""
        def ref_loss_fn(params):
            hidden = network.apply(params.network_params, mem_obs)
            logits = actor.apply(params.actor_params, hidden)
            values = critic.apply(params.critic_params, hidden).squeeze(-1)
            logprob = jax.nn.log_softmax(logits)[jnp.arange(mem_actions.shape[0]), mem_actions]
            actor_loss = -logprob.mean()
            critic_loss = 0.5 * ((values - mem_returns) ** 2).mean()
            return actor_loss + config["VF_COEF"] * critic_loss
        return jax.grad(ref_loss_fn)(params)

    def modify_gradients(grads, params, cl_state, key):
        """A-GEM and PackNet change the gradient before the optimizer step."""
        if cl_method == "agem":
            buffer_state = cl_state["buffer_state"]
            # only once the memory holds data, i.e. after the first task
            has_data = jnp.logical_or(buffer_state.is_full, buffer_state.current_index > 0)

            def project(g):
                batch = agem_buffer.sample(buffer_state, key).experience
                ref = agem_ref_grads(params, batch["observations"], batch["actions"], batch["value_targets"])
                return agem_project(g, ref)

            return jax.lax.cond(has_data, project, lambda g: g, grads)
        if cl_method == "packnet":
            return packnet_grad_mask(grads, cl_state)
        return grads

    def make_task_fns(env):
        """Rollout + GAE and the PPO update for one task's training env."""

        def reset_envs(rng):
            obs, state = jax.vmap(env.reset)(rng)
            return jax.vmap(prepare_obs)(obs), state

        def step_envs(state, action):
            obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
            done = jnp.logical_or(terminated, truncated)
            return jax.vmap(prepare_obs)(obs), state, reward, done, info

        def rollout(params, env_state, obs, last_done, key):
            """num_steps per env, then GAE computed backwards through the rollout."""
            def rollout_step(c, _):
                env_state, last_obs, last_done, key = c
                action, logprob, value, key = get_action_and_value(params, last_obs, key)
                next_obs, next_state, reward, next_done, info = step_envs(env_state, action)
                storage = Storage(
                    obs=last_obs,
                    actions=action,
                    logprobs=logprob,
                    dones=last_done,
                    values=value,
                    rewards=reward,
                    returns=jnp.zeros_like(reward),
                    advantages=jnp.zeros_like(reward),
                )
                return (next_state, next_obs, next_done, key), (storage, info)

            (env_state, next_obs, next_done, key), (storage, infos) = jax.lax.scan(
                rollout_step, (env_state, obs, last_done, key), None, length=num_steps
            )

            next_value = critic.apply(params.critic_params, network.apply(params.network_params, next_obs)).squeeze(1)
            dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0).astype(jnp.float32)
            values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)

            def gae_step(advantages, inp):
                nextdone, nextvalues, curvalues, reward = inp
                nextnonterminal = 1.0 - nextdone
                delta = reward + config["GAMMA"] * nextvalues * nextnonterminal - curvalues
                advantages = delta + config["GAMMA"] * config["GAE_LAMBDA"] * nextnonterminal * advantages
                return advantages, advantages

            _, advantages = jax.lax.scan(
                gae_step, jnp.zeros((num_envs,)), (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
            )
            storage = storage.replace(advantages=advantages, returns=advantages + storage.values)
            return env_state, next_obs, next_done, key, storage, infos

        def ppo_update(carry, cl_state):
            """One rollout, GAE, and update_epochs of minibatch SGD."""
            agent_state, env_state, obs, last_done, key, global_step = carry
            env_state, next_obs, next_done, key, storage, infos = rollout(agent_state.params, env_state, obs, last_done, key)

            def update_epoch(carry, _):
                agent_state, key = carry
                key, perm_key = jax.random.split(key)
                flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), storage)
                # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
                shuffled = jax.tree.map(
                    lambda x: jnp.reshape(jax.random.permutation(perm_key, x), (num_minibatches, -1) + x.shape[1:]),
                    flat,
                )

                def update_minibatch(carry, mb):
                    agent_state, key = carry
                    key, cl_key = jax.random.split(key)
                    (loss, (pg_loss, v_loss, entropy_loss, approx_kl, cl_loss)), grads = ppo_loss_grad_fn(
                        agent_state.params, mb.obs, mb.actions, mb.logprobs, mb.advantages, mb.returns, cl_state
                    )
                    grads = modify_gradients(grads, agent_state.params, cl_state, cl_key)
                    agent_state = agent_state.apply_gradients(grads=grads)
                    return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, cl_loss)

                (agent_state, key), losses = jax.lax.scan(update_minibatch, (agent_state, key), shuffled)
                return (agent_state, key), losses

            (agent_state, key), losses = jax.lax.scan(update_epoch, (agent_state, key), None, length=update_epochs)
            losses = jax.tree.map(lambda x: x[-1, -1], losses)
            learning_rate = agent_state.opt_state[1].hyperparams["learning_rate"]

            new_carry = (agent_state, env_state, next_obs, next_done, key, global_step + batch_size)
            return new_carry, (infos, losses, learning_rate)

        def train_chunk(carry, cl_state):
            return jax.lax.scan(lambda c, _: ppo_update(c, cl_state), carry, None, length=scan_steps)

        return jax.jit(reset_envs), jax.jit(rollout), train_chunk

    @partial(jax.jit, static_argnames="minibatch_size")
    def compute_fisher(params, obs_batch, action_batch, old_fisher, ewc_decay, minibatch_size):
        """[EWC] diagonal Fisher information: mean squared gradient of the log-probability of
        the actions the current policy took, in minibatches; blended with the previous Fisher
        (online EWC: F = decay * F_old + (1 - decay) * F_new)."""
        def log_prob(net_params, act_params, obs, action):
            logits = actor.apply(act_params, network.apply(net_params, obs[None]))
            return jax.nn.log_softmax(logits)[0, action]

        per_sample_grad_fn = jax.vmap(jax.grad(log_prob, argnums=(0, 1)), in_axes=(None, None, 0, 0))

        num_batches = obs_batch.shape[0] // minibatch_size
        obs_batch = obs_batch[:num_batches * minibatch_size].reshape((num_batches, minibatch_size) + obs_batch.shape[1:])
        action_batch = action_batch[:num_batches * minibatch_size].reshape((num_batches, minibatch_size))

        def scan_fn(fisher_sum, minibatch):
            grads_network, grads_actor = per_sample_grad_fn(params.network_params, params.actor_params, *minibatch)
            squared = {"network": jax.tree.map(lambda g: jnp.sum(g ** 2, axis=0), grads_network),
                       "actor": jax.tree.map(lambda g: jnp.sum(g ** 2, axis=0), grads_actor)}
            return jax.tree.map(jnp.add, fisher_sum, squared), None

        fisher_sum, _ = jax.lax.scan(scan_fn, jax.tree.map(jnp.zeros_like, old_fisher), (obs_batch, action_batch))
        new_fisher = jax.tree.map(lambda f: f / (num_batches * minibatch_size), fisher_sum)
        return jax.tree.map(lambda f_old, f_new: ewc_decay * f_old + (1. - ewc_decay) * f_new, old_fisher, new_fisher)

    def on_task_end(task_idx, cl_state, params, storage, key):
        """Update the CL state from a fresh rollout of the finished task (EWC, A-GEM)."""
        if cl_method in ("ewc", "agem"):
            flat_obs = storage.obs.reshape((-1,) + storage.obs.shape[2:])
            flat_actions = storage.actions.reshape(-1)
        if cl_method == "ewc":
            ewc_decay = 0.0 if task_idx == 0 else config["EWC_DECAY"]
            start = time.perf_counter()
            fisher = compute_fisher(params, flat_obs, flat_actions, cl_state["fisher"], jnp.float32(ewc_decay),
                                    minibatch_size=min(256, flat_obs.shape[0]))
            print(f"[ppo_crl] task {task_idx + 1}: Fisher computation time {time.perf_counter() - start:.2f}s")
            # copies: the next task's training chunk donates the live parameters
            anchor = jax.tree.map(jnp.copy, {"network": params.network_params, "actor": params.actor_params})
            cl_state = {"fisher": fisher, "params": anchor}
        elif cl_method == "agem":
            indices = jax.random.choice(key, flat_obs.shape[0], (samples_per_task,), replace=False)
            batch = {"observations": flat_obs[indices], "actions": flat_actions[indices],
                     "value_targets": storage.returns.reshape(-1)[indices]}
            cl_state = {"buffer_state": agem_buffer.add(cl_state["buffer_state"], batch)}
            print(f"[ppo_crl] task {task_idx + 1}: A-GEM memory updated")
        elif cl_method == "packnet":
            # unlock the free (pruned) weights for the next task
            cl_state = {**cl_state, "current_task_idx": cl_state["current_task_idx"] + 1,
                        "is_retraining": jnp.array(False)}
        return cl_state

    # --- evaluation on every task seen so far ---------------------------------
    seen_tasks = []          # (task index (1-based), game, train mods, eval mods)
    evaluators = {}          # compiled once per eval env
    crl_state = {"max_performances": {}, "a_j_j": {}}

    def task_key(game, train_mods, eval_mods):
        return f"{game}+tr:{train_mods}+ev:{eval_mods}"

    def save_and_eval(params, cl_state, task_idx, step_count, is_end_of_task):
        game, train_mods = tasks[task_idx], task_train_mods[task_idx]
        if is_end_of_task and config.get("SAVE_PATH") is not None:
            save_dir = os.path.join(config["SAVE_PATH"], run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'{config["EXP_NAME"]}_task{task_idx + 1}_{step_count}_{int(time.time())}.cleanrl_model')
            # PackNet also needs the weight -> task assignment to evaluate a task
            task_ids = cl_state["param_task_ids"] if cl_method == "packnet" else None
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(
                    [config, [params.network_params, params.actor_params, params.critic_params], task_ids]))
            print(f"model saved to {model_path}")

        print(f"running evaluation at step {step_count}...")
        performances = {}
        for eval_task_idx, eval_game, eval_train_mods, eval_mods in seen_tasks:
            eval_key = task_key(eval_game, eval_train_mods, eval_mods)
            if eval_key not in evaluators:
                eval_env = make_env(eval_game, eval_mods, eval=True, **env_kwargs)()
                evaluators[eval_key] = (make_evaluator(eval_env, network, actor, config["EVAL_EPISODES"], prepare_obs),
                                        jaxtari.make(eval_game, mods=eval_mods or None).renderer)
            evaluate, renderer = evaluators[eval_key]

            net_params, act_params = params.network_params, params.actor_params
            if cl_method == "packnet":
                net_params, act_params = packnet_inference_mask(params, cl_state, eval_task_idx)
            episodic_returns, env_states = evaluate(net_params, act_params, seed=config["SEED"] + 42)
            performances[eval_key] = float(np.mean(jax.device_get(episodic_returns)))
            wandb.log({f"{eval_key}/eval/episodic_return": performances[eval_key]}, step=step_count)
            print(f"[ppo_crl] eval {eval_key}: {performances[eval_key]:.2f}")

            # videos only of the current task's envs
            capture_video = config.get("CAPTURE_VIDEO", False) and eval_game == game and eval_train_mods == train_mods
            if capture_video and (is_end_of_task or not config.get("VIDEO_AT_END", True)):
                frames = jnp.transpose(jax.vmap(renderer.render)(env_states), (0, 3, 1, 2))
                wandb.log({f"{eval_key}/eval/video": wandb.Video(np.array(frames), fps=30, format="mp4")}, step=step_count)
                print(f"Video ({eval_key}/eval) logged to wandb with {frames.shape[0]} frames.")

        if not is_end_of_task:
            return performances

        # CRL metrics: a_k_j = performance on task j after training task k
        metrics = {"crl_metrics/Average_Performance": sum(performances.values()) / len(performances)}
        is_current = lambda t: t[1] == game and t[2] == train_mods
        bwt, fm = [], []
        for t in seen_tasks:
            eval_key = task_key(*t[1:])
            if not is_current(t) and eval_key in crl_state["a_j_j"]:
                bwt.append(performances[eval_key] - crl_state["a_j_j"][eval_key])             # backward transfer
                fm.append(crl_state["max_performances"][eval_key] - performances[eval_key])  # forgetting
        if bwt:
            metrics["crl_metrics/Backward_Transfer"] = sum(bwt) / len(bwt)
            metrics["crl_metrics/Forgetting_Measure"] = sum(fm) / len(fm)
        wandb.log(metrics, step=step_count)
        print(f"[ppo_crl] after task {task_idx + 1}: " + " | ".join(f"{k.split('/')[-1]} {v:.2f}" for k, v in metrics.items()))

        # update max performances and lock in the baseline a_j_j of the current task
        for t in seen_tasks:
            eval_key = task_key(*t[1:])
            crl_state["max_performances"][eval_key] = max(crl_state["max_performances"].get(eval_key, -np.inf), performances[eval_key])
            if is_current(t):
                crl_state["a_j_j"][eval_key] = performances[eval_key]
        return {**performances, **{k.split("/")[-1]: v for k, v in metrics.items()}}

    # --- train the tasks in sequence -----------------------------------------
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=run_name, max_iterations=num_chunks * num_tasks)
    rtpt.start()
    run_time = time.perf_counter()
    global_step = 0
    for i, game in enumerate(tasks):
        for eval_mods in task_eval_mods[i]:
            if not any(t[1:] == (game, task_train_mods[i], eval_mods) for t in seen_tasks):
                seen_tasks.append((i + 1, game, task_train_mods[i], eval_mods))
        if task_train_mods[i]:
            print(f"[WARNING] Training on mods {task_train_mods[i]}!")

        # reset the optimizer state (step count & momentum) for every new task
        if i > 0:
            agent_state = agent_state.replace(opt_state=agent_state.tx.init(agent_state.params))

        reset_envs, rollout, train_chunk = make_task_fns(train_envs[i])
        key, reset_key = jax.random.split(key)
        obs, env_state = reset_envs(jax.random.split(reset_key, num_envs))
        carry = (agent_state, env_state, obs, jnp.zeros(num_envs, dtype=jnp.bool_), key, jnp.array(global_step, dtype=jnp.int32))

        print(f"[ppo_crl] task {i + 1}/{num_tasks}: {game} {task_train_mods[i]} | start compile...")
        compile_start = time.perf_counter()
        # donate the carry so XLA reuses its memory; lower/compile AOT so nothing is donated before the loop
        compiled = jax.jit(train_chunk, donate_argnums=(0,)).lower(carry, cl_state).compile()
        print(f"[ppo_crl] compilation time: {time.perf_counter() - compile_start:.2f}s")

        prune_at = int(config.get("PRUNE_ITERATION_FRAC", 0.66) * num_updates)
        pruned = False
        print(f"[ppo_crl] starting training: {num_chunks} chunks x {steps_per_chunk} env steps")
        for chunk in range(num_chunks):
            rtpt.step()
            if config["EVAL_DURING_TRAIN"] and chunk > 0 and chunk % config["EVAL_EVERY"] == 0:
                save_and_eval(carry[0].params, cl_state, i, int(carry[-1]), is_end_of_task=False)

            # [PackNet] prune at the first chunk boundary past PRUNE_ITERATION_FRAC of the task's updates
            if cl_method == "packnet" and not pruned and chunk * scan_steps >= prune_at:
                pruned = True
                num_tasks_left = num_tasks - int(cl_state["current_task_idx"])
                prune_fraction = num_tasks_left / (num_tasks_left + 1) if num_tasks_left > 0 else 0.0
                print(f"[ppo_crl] PackNet pruning, fraction {prune_fraction:.4f}")
                params = carry[0].params
                task_idx_now = cl_state["current_task_idx"]
                pruned_net, net_ids = packnet_prune(params.network_params, cl_state["param_task_ids"]["network"], task_idx_now, prune_fraction)
                pruned_act, act_ids = packnet_prune(params.actor_params, cl_state["param_task_ids"]["actor"], task_idx_now, prune_fraction)
                cl_state = {**cl_state, "param_task_ids": {"network": net_ids, "actor": act_ids}}
                if prune_fraction > 0.0:
                    # retrain the kept weights with a fresh optimizer; the pruned ones stay locked
                    params = params._replace(network_params=pruned_net, actor_params=pruned_act)
                    agent_state = carry[0].replace(params=params, opt_state=carry[0].tx.init(params))
                    carry = (agent_state,) + carry[1:]
                    cl_state = {**cl_state, "is_retraining": jnp.array(True)}

            chunk_start = time.perf_counter()
            carry, (infos, (loss, pg_loss, v_loss, entropy_loss, approx_kl, cl_loss), learning_rate) = compiled(carry, cl_state)
            global_step = int(carry[-1])
            task_prefix = f"{game}+{task_train_mods[i]}"
            metrics = {
                "charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
                "charts/avg_episodic_length": float(infos["returned_episode_lengths"][-1].mean()),
                f"{task_prefix}/charts/avg_episodic_return": float(infos["returned_episode_returns"][-1].mean()),
                "charts/learning_rate": learning_rate[-1].item(),
                "charts/task": i + 1,
                "losses/value_loss": v_loss[-1].item(),
                "losses/policy_loss": pg_loss[-1].item(),
                "losses/entropy": entropy_loss[-1].item(),
                "losses/approx_kl": approx_kl[-1].item(),
                "losses/cl_loss": cl_loss[-1].item(),
                "losses/loss": loss[-1].item(),
                "charts/SPS": int(global_step / (time.perf_counter() - run_time)),
                "charts/SPS_update": int(steps_per_chunk / (time.perf_counter() - chunk_start)),
                "charts/time": time.perf_counter() - run_time,
                "charts/global_step": global_step,
            }
            wandb.log(metrics, step=global_step)
            print(
                f"[ppo_crl] task {i + 1} chunk {chunk + 1}/{num_chunks} | global_step {global_step} | "
                f"avg_return {metrics['charts/avg_episodic_return']:.2f} | "
                f"avg_length {metrics['charts/avg_episodic_length']:.2f} | "
                f"loss {metrics['losses/loss']:.4f} | cl_loss {metrics['losses/cl_loss']:.4g} | "
                f"SPS {metrics['charts/SPS']}"
            )

        agent_state, env_state, obs, done, key = carry[:5]
        eval_metrics = save_and_eval(agent_state.params, cl_state, i, global_step, is_end_of_task=True)

        # a fresh rollout of the finished task for the Fisher information / the A-GEM memory
        key, rollout_key, task_end_key = jax.random.split(key, 3)
        storage = None
        if cl_method in ("ewc", "agem"):
            *_, storage, _ = rollout(agent_state.params, env_state, obs, done, rollout_key)
        cl_state = on_task_end(i, cl_state, agent_state.params, storage, task_end_key)

    wandb.finish()
    return eval_metrics
