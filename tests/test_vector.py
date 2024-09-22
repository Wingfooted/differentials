from differentials import *
import jax
import jax.numpy as jnp

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



