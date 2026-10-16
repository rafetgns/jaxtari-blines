from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp


def impala_cnn(
    x,
    channels: Sequence[int] = (16, 32, 32),
    kernel_init: Callable = nn.initializers.lecun_normal(),
    norm: Optional[Callable] = None,
):
    """IMPALA encoder (Espeholt et al., 2018): per stage a conv, a 3x3/2 max-pool and two residual blocks.

    Drop-in replacement for the Nature conv stack: takes (B, frames, 84, 84) uint8
    frames and returns the flattened, ReLU'd conv features (B, 11*11*channels[-1]);
    each network keeps its own dense layers on top. Call it from inside an
    `@nn.compact` method so the layers register in the calling module. `norm`
    (e.g. `lambda h: nn.LayerNorm()(h)`) is applied after every conv, for agents
    whose Nature stack is normalised too (PQN, CrossQ, ...).
    """
    norm = norm or (lambda h: h)

    def conv(h, ch):
        return norm(nn.Conv(ch, kernel_size=(3, 3), padding="SAME", kernel_init=kernel_init)(h))

    x = jnp.transpose(x, (0, 2, 3, 1))  # (batch, frames, H, W) -> (batch, H, W, frames)
    x = x.astype(jnp.float32) / 255.0
    for ch in channels:
        x = conv(x, ch)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        for _ in range(2):  # residual blocks
            h = conv(nn.relu(x), ch)
            h = conv(nn.relu(h), ch)
            x = x + h
    x = nn.relu(x)
    return x.reshape((x.shape[0], -1))
