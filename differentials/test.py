import jax
import jax.numpy as jnp

from boundary import function, dx, dt

# dx = lambda u: jax.grad(u, argnums=0)

u = function()
x = jnp.array((0.1, 0.1))

f = lambda x: jnp.sin(x[0]) * jnp.cos(x[1]) + jnp.exp(x[0])

ux = dx(u)
uxx = dx(ux)
ut = dt(u)

print(ut(f)(x))
print(ux(f)(x))
print(uxx(f)(x))

expr = dt(u) == 4
print(expr)
print(expr(f)(x))

# manual diff


