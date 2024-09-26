from flax.serialization import from_bytes

import jax
import jax.numpy as jnp
import jax.random as random

from typing import Callable, Generator, Sequence

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from flax import linen as nn


class Model(nn.Module):
    model_layout: Sequence[int] = (20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20)
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-3)
    # train: bool = False

    @nn.compact
    def __call__(self, x):
        for layer_width in self.model_layout:
            x = nn.Dense(layer_width,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            # x = nn.BatchNorm(use_running_average=not self.train)(x)
            x = nn.selu(x)
        x = nn.Dense(1)(x)
        return nn.selu(x)


def visualize_3d(u: Callable, *args, defenition: int = 1000, title="U", xlab="x", ylab="t"):

    # a not on u: Callabale, has to be a lambda function with one jax input
    xs = list()
    x1_domains = args[0]
    x2_domains = args[1]

    for x1 in x1_domains:
        for x2 in x2_domains:
            xs.append((x1, x2))

    xs = jnp.array(xs)
    print(xs)
    ys = jax.vmap(lambda x: u(x))(xs)
    data = jnp.concat((xs, ys), axis=1)
    print(data)

    x = data[:, 0]
    t = data[:, 1]
    u = data[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, t, u, c=u, cmap='viridis', marker='o')


    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel('U')
    ax.set_title(title)

    # Optional: Add a color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('U')

    plt.show()


def load_model(path: str):
    with open(path, "rb") as raw:
        byte_data = raw.read()
        model_structure = (20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20)
        model = Model(model_structure)
        params = from_bytes(model, byte_data)
    return model, params


if __name__ == '__main__':
    model, params = load_model("params.bin")
    x = random.normal(random.key(0), (2,))

    print(params)
    y = Model.apply(params, x)
    print(y)

    visualize_3d(lambda x: model.apply(params, x),
                 jnp.linspace(-5, 5, 100),
                 jnp.linspace(0, 10, 100),
                 defenition=100)
