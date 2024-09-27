import jax
import jax.numpy as jnp

from differentials import function, dx, dt, expression, boundary, term

u = function()
x = jnp.array((1.0, 4.0))
sinx = term(lambda u: lambda x: jnp.sin(x[0])) # defining sinx

f = lambda x: jnp.sin(x[0]) * jnp.cos(x[1]) + jnp.exp(x[0])

heat = expression(
    dx(dx(u)) == dt(u),
    boundary(
            u == sinx,
            t=0
    ),
    x=jnp.linspace(0, 1, 50),
    t=jnp.linspace(0, 1, 50)
)
