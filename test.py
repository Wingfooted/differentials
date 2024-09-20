import jax
import jax.numpy as jnp

from differentials import expression, boundary, initial

if __name__ == '__main__':

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
        x=jnp.linspace(0, 1, num=100),
        t=jnp.linspace(0, 1, num=100)
    )

    u_model, params = heat.u(struct=(2, 3))
    print(str(params)[-50:])
    u = lambda x, t: u_model.apply(params, jnp.array((x, t)))[0] # this makes this a scalar func

    print(u(0.1, 0.1))
    print(dt(u)(100, 0.1))
    print(dx(u)(0.1, 0.1))
    print(dx(dx(u))(0.1, 0.1))

    loss = heat.loss(u, 0.1, 1.0)
    print(loss)
    print("ended")
