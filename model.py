from flax import linen as nn
from typing import Sequence, Callable

import jax

class Model(nn.Module):
    model_layout: Sequence[int] = (20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20)
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    # train: bool = False

    @nn.compact
    def __call__(self, x):
        for layer_width in self.model_layout:
            x = nn.Dense(layer_width,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            # x = nn.BatchNorm(use_running_average=not self.train)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return nn.selu(x)
