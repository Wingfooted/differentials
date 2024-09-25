import jax
import jax.numpy as jnp
import jax.random as random

from typing import Sequence

def make_loss(expression, n=100, struct=(1, 1)):
    u_hat, _ = expression.u(struct=struct)

    # make a xs matrix quickly
    xs_rng = random.key(1)
    # moddify this process
    xs = random.uniform(xs_rng,
                        shape=(n, 2),
                        minval=-10,
                        maxval=10)

    def pde_loss(params):
        def loss_unit(x):
            loss_val = expression.loss(
                    lambda x: u_hat.apply(params, x),
                    x)
            return loss_val
        return jnp.mean(jax.vmap(loss_unit)(xs))
    return jax.jit(pde_loss)


def train(expression,
          struct: Sequence[int] = (20, 20, 20, 20, 20, 20),
          epochs: int = 100,
          epoch_log: int = 5,
          lr: float = 0.001,
          gamma: float = 0.99,
          epsilon: float = 1e-6,
          # HYPERPARAMS GO HERE
          loss_alg=make_loss
          ):


    u_hat, params = expression.u(struct=struct)
    velocity = jax.tree.map(lambda p: jnp.zeros_like(p), params)  # empty params


    @jax.jit
    def RMSProp(params, grads, velocity):
        velocity = jax.tree.map(lambda v, g: gamma * v + (1-gamma) * jnp.square(g), velocity, grads)
        params = jax.tree.map(lambda p, g, v: p - (lr / (jnp.sqrt(v) + epsilon)) * g, params, grads, velocity)
        return velocity, params

    for epoch in range(epochs+1):
        loss = loss_alg(expression)
        loss_val, grads = jax.value_and_grad(loss)(params)

        velocity, params = RMSProp(params, grads, velocity)

        if epoch % epoch_log == 0:
            print(f"epoch: {epoch}, loss: {loss_val}")

    return model, params

    return lambda x: u_hat.apply(params, x)
