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
x = jnp.array([float(0), float(1)])
f = lambda x, t, params: u.apply(params, jnp.array([x, t]))[0]  # Ensure scalar output (e.g., by taking [0])
print(f(0, 1, params))  # Should print a scalar output
f_x = jax.grad(f, argnums=0)  # Differentiating w.r.t. x
f_xx = jax.grad(f_x, argnums=0)
print(f_x(0.0, 1.0, params))  # Should print the gradient w.r.t. x at (0, 1)
print(f_xx(0.0, 1.0, params))  # Should print the gradient w.r.t. x at (0, 1)
hessian_f = jax.hessian(f)
print(hessian_f(0.0, 1.0, params))  # Should print the Hessian matrix at (0, 1)
