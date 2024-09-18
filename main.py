import jax
import jax.numpy as jnp

from model import Model
from differentials import expression, forward

# let u be approximated by u^

# u = Model() # u is fit by the expression

heat = expression(
    lambda u, x, t: u.d(1) + u.d(0).dx(0), #x = 0, t = 1
    var = ("x", "t"),
    boundary = {
        (0, "t"): lambda x, t: 0,  # U(0, t) = 0
        (1, "t"): lambda x, t: 0,  # U(1, t) = 0
        ("x", 0): lambda x, t: jnp.sin(3.14 * x)  # U(x, 0) = sin(pi x)
    },
    x = jnp.linspace(-1, 1, num=100),  # for x domain
    t = jnp.linspace(0, 1, num=100)    # for t domain
)

u, params = heat.u()
x = jnp.array((0, 1))
print(u.apply(params, x))
ux = jax.grad(u, argnums=0)
print(ux)
print(ux.__code__.co_varnames)

print(forward(u, params, x))
print(jax.grad(forward(u, params), argnums=1)(x))
