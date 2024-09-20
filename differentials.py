import jax
import jax.random as random
import jax.numpy as jnp

from typing import Callable, Tuple, Dict, Sequence

from model import Model


# want to have a function for u

# desired syntax

# expression(x1, x2, ..., xn) = 0 __call__
# essential for loss function.
# expression.dependants -> tuple
# expression(FUNCTION, args[ivp], kwargs[domains])
# expression.u()

class expression:
    def __init__(self, function: Callable,
                 var: Tuple[str],
                 *args, **kwargs):
        # heat = expression(
        #     lambda u, x, t: u.dt() + u.dx().dx(),
        #     var = ("x", "t"),
        #     boundary = {
        #         (0, "t"): lambda x, t: 0,  # U(0, t) = 0
        #         (1, "t"): lambda x, t: 0,  # U(1, t) = 0
        #         ("x", 0): lambda x, t: jnp.sin(3.14 * x)  # U(x, 0) = sin(pi x)
        #     },
        #     x = jnp.linspace(-1, 1, num=100),  # for x domain
        #     t = jnp.linspace(0, 1, num=100)    # for t domain
        # )

        self.function = function
        self.variables = var
        self.domains = list()
        for key, value in kwargs.items():
            if str(key) in self.variables:
                self.domains.append(value)

    def loss(self, 
                 U: Callable = lambda *args: None,
		*args) -> float:
        # expression(x1, x2, ... , xn) -> float
        # expression(x1, x2, ... , xn, U=U_validation) -> float

        value: jax.Array = jnp.array((0))

        value += self.function(U, *args)

        return value

    def u(self,
          struct: Sequence[int] = (4, 5, 5, 4)
            ) -> Tuple:
        schema = (len(self.variables), *struct)
        u_hat = Model(schema)
        forward_rng, model_rng = random.split(random.key(0), (2,))
        x = list()
        for domain in self.domains:
            element = random.choice(forward_rng, domain)
            x.append(element)
        params = u_hat.init(model_rng, jnp.array(x))
        return u_hat, params


def forward(u, params,  x):
    return u.apply(params, x)
