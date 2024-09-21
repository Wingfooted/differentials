import jax
import jax.random as random
import jax.numpy as jnp

import flax

from differentials import expression, domain, boundary, initial
from tools import visualize_3d


# struct = (4, 4, 4)
# makes loss from a PDE expression
# assumes u_hat


def make_loss(expression, n=100):
    u_hat, _ = expression.u()
    # hyper param, num of samples per loss
    xs = expression.matrix(n)
    def loss(params):
        def loss_unit(x):
            error = expression.loss(
                lambda x, t: u_hat.apply(params, jnp.array((x, t)))[0],
                x[0], x[1]  # this is for x and t. No better way exists to do this
            )
            return error
        return jnp.max(jax.vmap(loss_unit)(xs))
        # here there is a contention. What loss is better, the worst point tested, or the average point tested
    return jax.jit(loss)

# TRAINING A MODEL on an Expression

# Solving the PDE Heat Equation

if __name__ == '__main__':

    # FORMAT
    # u(x, t)

    dx = lambda u: jax.grad(u, argnums=0)
    dt = lambda u: jax.grad(u, argnums=1)

    heat = expression(
        lambda u: lambda x, t: dt(u)(x, t) + 10 * dx(dx(u))(x, t),
        var=("x", "t"),
        boundaries=(
            # insulated ends u_x(0, t) = 0
            boundary(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: 0.0,
                con=(0.0, "t")
            ),
            # insulated end u_x(L, t) = 0
            boundary(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: 0.0,
                con=(1.0, "t")
            ),
            # inital function. u(x, 0) = sin(x)
            initial(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: jnp.sin(x * jnp.pi) * jnp.exp(x), 
                con=("x", 0.0)
            )
        ),
        x=domain(-1, 1),
        t=domain(0, 1)
    )

    '''initial(
        LHS=lambda u: lambda x, t: u(x, t),
            RHS=lambda u: lambda x, t: 0.5,
            con=(0.1, 0.5)
    )'''
    # initial visualize_3d(lambda x: u_hat.apply(params, x), heat, defenition=80)

    model_structure = (8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8)
    epochs = 50
    epoch_logs = 1  # how often to log loss
    lr = 0.01
    gamma = 0.999
    epsilon = 1e-6

    # initializing model / params
    u_hat, params = heat.u()
    velocity = jax.tree.map(lambda p: jnp.zeros_like(p), params) # empty params

    @jax.jit
    def param_update(params, grads, velocity):
        velocity = jax.tree.map(lambda v, g: gamma * v + (1-gamma) * jnp.square(g), velocity, grads)
        params = jax.tree.map(lambda p, g, v: p - (lr / (jnp.sqrt(v) + epsilon)) * g, params, grads, velocity)
        return velocity, params

    for epoch in range(epochs):
        heat_loss = make_loss(heat)
        loss, grads = jax.value_and_grad(heat_loss)(params)
        # gradient descent component
        velocity, params = param_update(params, grads, velocity)
        # velocity = jax.tree.map(lambda v, g: gamma * v + (1-gamma) * jnp.square(g), velocity, grads)
        # params = jax.tree.map(lambda p, g, v: p - (lr / (jnp.sqrt(v) + epsilon)) * g, params, grads, velocity)

        if epoch % epoch_logs == 0:
            print(f"epoch: {epoch}, loss: {loss}")

    visualize_3d(lambda x: u_hat.apply(params, x), heat, defenition=150)
