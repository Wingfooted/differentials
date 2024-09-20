import jax
import jax.random as random
import jax.numpy as jnp

import flax

from differentials import expression, domain, boundary, initial


# struct = (4, 4, 4)
# makes loss from a PDE expression
# assumes u_hat


def make_pde_loss(expression):
    u_hat, _ = expression.u((4, 4, 4))
    def loss(params):
        # make values
        n = 30  # HYPER PARAMETER, mean samples taken
        xs_matrix = expression.matrix(n)
        def rnd_instance_val(x):
            print(x)
            error = expression.loss(
                    lambda x, t: u_hat.apply(params, jnp.array((x, t))),
                    x
            )
            print("HERE", error)
            return error

        return jnp.mean(jax.vmap(rnd_instance_val, in_axes=0)(xs_matrix))
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
    test_loss = make_pde_loss(heat)
    value_and_grad_fn = jax.value_and_grad
    print(test_loss(params))

