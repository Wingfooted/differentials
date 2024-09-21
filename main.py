import jax
import jax.random as random
import jax.numpy as jnp

import flax

from differentials import expression, domain, boundary, initial
from tools import visualize_3d


# struct = (4, 4, 4)
# makes loss from a PDE expression
# assumes u_hat


def make_loss(expression, n=40):
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

    heat = expression(
        lambda u: lambda x, t: dt(u)(x, t) + dx(dx(u))(x, t),
        var=("x", "t"),
        boundaries=(
            # insulated ends u_x(0, t) = 0
            boundary(
                LHS=lambda u: lambda x, t: dx(u)(x, t),
                RHS=lambda u: lambda x, t: 0.0,
                con=(0.0, "t")
            ),
            # insulated end u_x(L, t) = 0
            boundary(
                LHS=lambda u: lambda x, t: dx(u)(x, t),
                RHS=lambda u: lambda x, t: 0.0,
                con=(1.0, "t")
            ),
            # inital function. u(x, 0) = sin(x)
            initial(
                LHS=lambda u: lambda x, t: u(x, t),
                RHS=lambda u: lambda x, t: jnp.sin(x),
                con=("x", 0.0)
            )
        ),
        x=domain(0, 1),
        t=domain(0, 1)
    )

    u_hat, params = heat.u()
    visualize_3d(lambda x: u_hat.apply(params, x), heat)
