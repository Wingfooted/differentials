import jax
import jax.numpy as jnp
import jax.random as random

from typing import Callable, Generator

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def approx_domain_limits(domain: Generator, n=10000):
    vector = jnp.array([next(domain) for _ in range(n)])
    return jnp.min(vector), jnp.max(vector)

def visualize_3d(u: Callable, expression, defenition: int = 1000, title="U"):

    # a not on u: Callabale, has to be a lambda function with one jax input
    assert len(expression.domains) == len(expression.domains_raw.keys()), "bruh"
    approximated_domains = list()

    for domain in expression.domains:
        minimum, maximum = approx_domain_limits(domain)
        approximated_domains.append(jnp.linspace(minimum, maximum, num=defenition))
    
    xs = list()
    # since it is assumed that there are only two iterable domains 3d
    assert len(approximated_domains) == 2, "two dependant variables for 3d graph"
    x1_domains = approximated_domains[0]
    x2_domains = approximated_domains[1]

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

    variable_names = expression.variables

    ax.set_xlabel(variable_names[0])
    ax.set_ylabel(variable_names[1])
    ax.set_zlabel('U')
    ax.set_title(title)

    # Optional: Add a color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('U')

    plt.show()


def _make_loss_map_data(f: Callable, *args, defenition=100):
    empty = list()
    for x in args[0]:
        for y in args[1]:
            empty.append((x, y))
    xs = jnp.array(empty)
    ys = jax.vmap(lambda x: f(x))(xs)
    return jnp.concat((xs,ys))

    xs = jnp.array(xs)
    print(xs)
    ys = jax.vmap(lambda x: f(x))(xs)

def plot_3d(data):
    x = data[:, 0]
    t = data[:, 1]
    l = data[:, 2] # l for loss

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, t, u, c=u, cmap='viridis', marker='o')

    variable_names = expression.variables

    ax.set_zlabel('loss')

    # Optional: Add a color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('loss')

    plt.show()

def loss_map(f: Callable, *args, defenition=50):
    # args assumed to be linsapce
    # make loss data
    data = _make_loss_map_data(f, *args, defenition=defenition)
    plot_3d(data)
