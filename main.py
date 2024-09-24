import jax
import jax.random as random
import jax.numpy as jnp

import flax

from differentials import expression, domain, boundary, initial
from tools import visualize_3d, loss_map


# struct = (4, 4, 4)
# makes loss from a PDE expression
# assumes u_hat
def raw_loss(x, y, params, model, expression):
    error = expression.loss(
        lambda x, t: u_hat.apply(params, jnp.array((x, t)))[0],
        x[0], x[1]  # this is for x and t. No better way exists to do this
    )
    return error

def make_loss(expression, n=100, struct=(1, 1)):
    u_hat, _ = expression.u(struct=struct)
    # hyper param, num of samples per loss
    xs = expression.matrix(n)
    def loss(params):
        def loss_unit(x):
            error = expression.loss(
                lambda x, t: u_hat.apply(params, jnp.array((x, t)))[0],
                x[0], x[1]  # this is for x and t. No better way exists to do this
            )
            return error
        return jnp.mean(jax.vmap(loss_unit)(xs))
        # here there is a contention. What loss is better, the worst point tested, or the average point tested
    return loss
    return jax.jit(loss)

# TRAINING A MODEL on an Expression

# Solving the PDE Heat Equation

if __name__ == '__main__':

    # FORMAT
    # u(x, t)

    dx = lambda u: jax.grad(u, argnums=0)
    dt = lambda u: jax.grad(u, argnums=1)

    c = lambda x, t: jax.array((x, t))

    heat = expression(
        lambda u: lambda x, t: dt(u)(c(x, t)) + 0.001 * dx(dx(u))(c(x, t)),
        var=("x", "t"),
        boundaries=(
            # insulated ends u_x(0, t) = 0
            boundary(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: 1,
                con=(0.0, "t")
            ),
            # insulated end u_x(L, t) = 0
            boundary(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: -1,
                con=(1.0, "t")
            ),
            # inital function. u(x, 0) = sin(x)
            initial(
                LHS=lambda u: lambda x, t: u(c(x, t)),
                RHS=lambda u: lambda x, t: 2 * jnp.exp(x)-1, 
                con=("x", 0.0)
            )
        ),
        x=domain(-1, 1),
        t=domain(0, 29)
    )

    '''initial(
        LHS=lambda u: lambda x, t: u(x, t),
            RHS=lambda u: lambda x, t: 0.5,
            con=(0.1, 0.5)
    )'''
    # initial visualize_3d(lambda x: u_hat.apply(params, x), heat, defenition=80)

    model_structure = (30, 30, 30, 30, 30, 30, 30, 30, 30, 30)
    epochs = 1000
    epoch_logs = 1  # how often to log loss
    lr = 0.001
    gamma = 0.99
    epsilon = 1e-6

    # initializing model / params
    u_hat, params = heat.u(struct=model_structure)
    velocity = jax.tree.map(lambda p: jnp.zeros_like(p), params)  # empty params

    @jax.jit
    def param_update(params, grads, velocity):
        velocity = jax.tree.map(lambda v, g: gamma * v + (1-gamma) * jnp.square(g), velocity, grads)
        params = jax.tree.map(lambda p, g, v: p - (lr / (jnp.sqrt(v) + epsilon)) * g, params, grads, velocity)
        return velocity, params

    for epoch in range(epochs):
        heat_loss = make_loss(heat, struct=model_structure)
        loss, grads = jax.value_and_grad(heat_loss)(params)
        # gradient descent component
        velocity, params = param_update(params, grads, velocity)
        # print(params)
        # velocity = jax.tree.map(lambda v, g: gamma * v + (1-gamma) * jnp.square(g), velocity, grads)
        # params = jax.tree.map(lambda p, g, v: p - (lr / (jnp.sqrt(v) + epsilon)) * g, params, grads, velocity)

        if epoch % epoch_logs == 0:
            print(f"epoch: {epoch}, loss: {loss}")
            # loss map
            loss_fn = lambda x, t: raw_loss(x, t, params, u_hat, heat)
            xt_domains = (jnp.linspace(0, 10), jnp.linspace(0, 5))
            loss_map(loss_fn, *xt_domains)


    visualize_3d(lambda x: u_hat.apply(params, x), heat, defenition=200)
