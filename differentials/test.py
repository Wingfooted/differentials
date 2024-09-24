import jax
import jax.numpy as jnp

from boundary import function, dx, dt

# dx = lambda u: jax.grad(u, argnums=0)

u = function()
x = jnp.array((1.0, 4.0))

f = lambda x: jnp.sin(x[0]) * jnp.cos(x[1]) + jnp.exp(x[0])

# term adding
expr_add = u + dx(u) == dt(u) + dx(dx(u)) + dx(u)
LHS = u + dx(u)
RHS = dt(u) + dx(dx(u)) + dx(u)
print(expr_add(f)(x))
print(LHS(f)(x))
print(RHS(f)(x))

# testing addition of numbers
print("NEGATIVE DEALING ---------------")
neg_expr = dx(u) - u
cost_expr = dx(u) - 4
print(dx(u)(f)(x))
print("here", cost_expr(f)(x))

# testing addition of numbers
print("MULTIPLICATION DEALING ---------------")
div_expr = dx(u)*2
print(div_expr(f)(x))

# testing the divison of terms
print("Divison ---------------")
div_expr = dx(u)/dt(u)
print(dx(u)(f)(x), dt(u)(f)(x))
print(div_expr(f)(x))

print("exponention")
exp_epr = dx(u) ** dt(u)
print(exp_epr(f)(x))

print("ARRAY UFUNC - sin")
ufunc_expr = jnp.sin(u)
print(ufunc_expr(f)(x))


expr = u == 4
expr1 = -4 == u
print(type(expr(f)(x)), expr(f)(x))
print(type(expr1(f)(x)), expr1(f)(x))

# print(u(f)(x))
# print(const(f)(x))
# print(u(f)(x)+const(f)(x))

# print(expr(f)(x))


# manual diff


