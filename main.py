import jax
import jax.numpy as jnp

from model import Model
from differentials import expression, forward

# let u be approximated by u^

# u = Model() # u is fit by the expression

dx = lambda u: jax.grad(u, argnums=1)
dt = lambda u: jax.grad(u, argnums=2)

heat = expression(
		lambda u: lambda x, t: dt(u)(x, t) + dx(dx(u))(x, t),
		var = ("x", "t"),
		x = jnp.linspace(-1, 1, num=100),  # for x domain
		t = jnp.linspace(0, 1, num=100)    # for t domain
		)

if __name__ == '__main__':
	# test
	import inspect
	u_model, params = heat.u()
	x = jnp.array([float(0), float(1)])
	u = lambda x, t: u_model.apply(params, jnp.array((x, t)))[0] # this makes this a scalar func
 
	#test 2
	print(u(0.0, 1.0))
	#print(inspect.getargspec(ux))
	ux = dx(u)
	print(dx(dx(u))(0.0, 1.0))

	dx = lambda f: jax.grad(f, argnums=0)
	dy = lambda f: jax.grad(f, argnums=1)

	f = lambda x, y: x * y * 2 + y * 2

	print(dx(f)(0.0, 1.0))
	print(dx(dx(f))(0.0, 1.0))
	print(dy(dx(f))(0.0, 1.0))
	print(dy(f)(0.0, 1.0))
	
