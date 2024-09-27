import jax
import jax.numpy as jnp

from .sampledxs import function, dx, dt
from .expression import expression
from .boundary import boundary

u = function()
x = jnp.array((1.0, 4.0))

f = lambda x: jnp.sin(x[0]) * jnp.cos(x[1]) + jnp.exp(x[0])

heat = expression(
    dx(dx(u)) == dt(u),
    x=jnp.linspace(0, 1, 50),
    t=jnp.linspace(0, 1, 50)

)

print(heat.shape)
print(heat.kwargs)
print(heat.loss_fns)
print(heat.domains)
