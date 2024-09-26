import jax
import jax.random as random
import jax.numpy as jnp

import flax
from flax import serialization

from differentials import expression, domain, boundary, initial
from tools import visualize_3d


# struct = (4, 4, 4)
# makes loss from a PDE expression
# assumes u_hat

    #model_layout: Sequence[int] = (20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20)

def make_loss(expression, n=1000, struct=(20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20)):
    u_hat, _ = expression.u(struct=struct)
    # hyper param, num of samples per loss
    def loss(params):
        xs = expression.matrix(n)
        def loss_unit(x):
            error = expression.loss(
                lambda x, t: u_hat.apply(params, jnp.array((x, t)))[0],
                x[0], x[1])
            return error
        return jnp.mean(jax.vmap(loss_unit)(xs))
    return jax.jit(loss)


if __name__ == '__main__':

    # FORMAT
    # u(x, t)

    dx = lambda u: jax.grad(u, argnums=0)
    dt = lambda u: jax.grad(u, argnums=1)

    alpha = 0.8

    heat = expression(
        lambda u: lambda x, t: dt(u)(x, t) + 6 * u(x, t) * dx(u)(x, t) + dx(dx(dx(u)))(x, t),
        var=("x", "t"),
        boundaries=(
            # insulated ends u_x(0, t) = 0
            boundary(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: 0,
                con=(0, "t")
            ),
            # insulated end u_x(L, t) = 0
            boundary(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: 0,
                con=(50, "t")
            ),
            # inital function. u(x, 0) = sin(x)
            initial(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: jnp.exp(- jnp.square(x-30)),
                con=("x", 0.0)
            )
        ),
        x=domain(0, 50),
        t=domain(0, 100)
    )

    '''initial(
        LHS=lambda u: lambda x, t: u(x, t),
            RHS=lambda u: lambda x, t: 0.5,
            con=(0.1, 0.5)
    )'''
    # initial visualize_3d(lambda x: u_hat.apply(params, x), heat, defenition=80)

    model_structure = (20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20)

    epochs = 2000
    epoch_logs = 1  # how often to log loss
    lr = 0.00007
    gamma = 0.99
    epsilon = 1e-6

    # initializing model / params
    u_hat, params = heat.u(struct=model_structure)
    velocity = jax.tree.map(lambda p: jnp.zeros_like(p), params)  # empty params

    def param_update(params, grads, velocity):
        velocity = jax.tree.map(lambda v, g: gamma * v + (1-gamma) * jnp.square(g), velocity, grads)
        params = jax.tree.map(lambda p, g, v: p - (lr / (jnp.sqrt(v) + epsilon)) * g, params, grads, velocity)
        return velocity, params

    heat_loss = make_loss(heat, n=200, struct=model_structure)
    for epoch in range(epochs):
        loss, grads = jax.value_and_grad(heat_loss)(params)
        # gradient descent component
        velocity, params = param_update(params, grads, velocity)
        # print(params)
        # velocity = jax.tree.map(lambda v, g: gamma * v + (1-gamma) * jnp.square(g), velocity, grads)
        # params = jax.tree.map(lambda p, g, v: p - (lr / (jnp.sqrt(v) + epsilon)) * g, params, grads, velocity)

        if epoch % epoch_logs == 0:
            print(f"epoch: {epoch}, loss: {loss}")

    print("saving")
    bytes_output = serialization.to_bytes(params)
    with open("models/bins/kdv.bin", "wb") as f:
        f.write(bytes_output)

    pass
