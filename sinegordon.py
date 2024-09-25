import jax
import jax.random as random
import jax.numpy as jnp

import flax

from diff import expression, domain, boundary, initial
from tools import visualize_3d


# struct = (4, 4, 4)
# makes loss from a PDE expression
# assumes u_hat


def make_loss(expression, n=100, struct=(1, 1)):
    u_hat, _ = expression.u(struct=struct)
    # hyper param, num of samples per loss
    def loss(params):
        xs = expression.matrix(n)
        def loss_unit(x):
            error = expression.loss(
                lambda x, t: u_hat.apply(params, jnp.array((x, t)))[0],
                x[0], x[1]  # this is for x and t. No better way exists to do this
            )
            return error
        return jnp.mean(jax.vmap(loss_unit)(xs))
        # here there is a contention. What loss is better, the worst point tested, or the average point tested
    return jax.jit(loss)

# TRAINING A MODEL on an Expression

# Solving the PDE Heat Equation

if __name__ == '__main__':

    # FORMAT
    # u(x, t)

    dx = lambda u: jax.grad(u, argnums=0)
    dt = lambda u: jax.grad(u, argnums=1)

    sine_gordon = expression(
        lambda u: lambda x, t: dt(dt(u))(x, t) - dx(dx(u))(x, t) - jnp.sin(u(x, t)),
        var=("x", "t"),
        boundaries=(
            # u(-10, t) = 0 (Dirichlet boundary condition)
            boundary(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: 0,
                con=(-10.0, "t")
            ),
            # u(10, t) = 0 (Dirichlet boundary condition)
            boundary(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: 0,
                con=(10.0, "t")
            ),
            # u(x, 0) = 4 * arctan(exp(x)) (initial condition)
            # u_t(x, 0) = 0 (initial velocity)
            initial(
                LHS=lambda u: lambda x, t: dt(u)(x, t),
                RHS=lambda u: lambda x, t: 0,
                con=("x", 0.0)
            ),
            initial(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: 4 * jnp.arctan(x), 
                con=("x", 0.0)
            ),
        ),
        x=domain(-10, 10),
        t=domain(0, 10)
    )

    '''initial( LHS=lambda u: lambda x, t: u(x, t),
        RHS=lambda u: lambda x, t: 4 * jnp.arctan(jnp.exp(x)),
        con=("x", 0.0)
    ),'''
    '''initial(
        LHS=lambda u: lambda x, t: u(x, t),
            RHS=lambda u: lambda x, t: 0.5,
            con=(0.1, 0.5)
    )'''
    # initial visualize_3d(lambda x: u_hat.apply(params, x), heat, defenition=80)

    model_structure = (30, 30, 30, 30, 30, 30, 30, 30, 30, 30)
    epochs = 500
    epoch_logs = 1  # how often to log loss
    lr = 0.0001
    gamma = 0.999
    epsilon = 1e-10

    # initializing model / params
    u_hat, params = sine_gordon.u(struct=model_structure)
    velocity = jax.tree.map(lambda p: jnp.zeros_like(p), params)  # empty params

    @jax.jit
    def param_update(params, grads, velocity):
        velocity = jax.tree.map(lambda v, g: gamma * v + (1-gamma) * jnp.square(g), velocity, grads)
        params = jax.tree.map(lambda p, g, v: p - (lr / (jnp.sqrt(v) + epsilon)) * g, params, grads, velocity)
        return velocity, params

    for epoch in range(epochs):
        heat_loss = make_loss(sine_gordon, struct=model_structure, n=1000)
        loss, grads = jax.value_and_grad(heat_loss)(params)
        # gradient descent component
        velocity, params = param_update(params, grads, velocity)
        # print(params)
        # velocity = jax.tree.map(lambda v, g: gamma * v + (1-gamma) * jnp.square(g), velocity, grads)
        # params = jax.tree.map(lambda p, g, v: p - (lr / (jnp.sqrt(v) + epsilon)) * g, params, grads, velocity)

        if epoch % epoch_logs == 0:
            print(f"epoch: {epoch}, loss: {loss}")

    visualize_3d(lambda x: u_hat.apply(params, x), sine_gordon, defenition=70)
