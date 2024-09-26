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
                 boundaries=None,
                 *args, **kwargs):

        self.function = function
        self.variables = var
        self.domains_raw = kwargs
        self.domains = list()
        for key, value in kwargs.items():
            if str(key) in self.variables:
                self.domains.append(value)
        
        self.boundaries_defined = True
        if boundaries:
            self.boundaries = boundaries
        else:
            self.boundaries_defined = False

    def loss(self,
             U: Callable = lambda *args: None,
             *args) -> float:

        # expression(x1, x2, ... , xn) -> float
        # expression(x1, x2, ... , xn, U=U_validation) -> float

        value = jnp.square(self.function(U)(*args))

        boundary_loss = value
        if self.boundaries_defined:
            for boundary in self.boundaries:
                instance_loss = boundary(U, *args)
                boundary_loss += jnp.square(instance_loss)
        return boundary_loss + value


    def u(self,
          struct: Sequence[int] = (20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20)
          ) -> Tuple:
<<<<<<< HEAD
        # schema = (len(self.variables), *struct)
=======
>>>>>>> new
        u_hat = Model(struct)
        forward_rng, model_rng = random.split(random.key(1), (2,))
        x = list()
        for domain in self.domains:
            element = next(domain)
            x.append(element)
        params = u_hat.init(model_rng, jnp.array(x))
        return u_hat, params

    def vector(self) -> jnp.array:
        # retuns a vector domain
        vector_internal = [next(domain) for domain in self.domains]
        return jnp.array(vector_internal)

    def matrix(self, n: int = 1) -> jnp.array:
        # returns a matrix of domain # traversal row wise
        matrix_internal = [(self.vector()) for _ in range(n)]
        return jnp.array(matrix_internal)


class boundary:
    def __init__(self, LHS: Callable,
                 RHS: Callable,
                 con: Tuple):

        self.RHS = RHS
        self.LHS = LHS
        self.con = con

    def __call__(self, u, *args):
        con = self.con

        def vals(con, args):
            output = []
            if len(con) != len(args):
                print(f"unamtching lens {len(con)}, {len(args)}")
            for i in range(len(con)):
                if isinstance(con[i], float):
                    output.append(con[i])
                else:
                    output.append(args[i])
            # print(output, self.con, args)
            return output

        inputs = vals(con, args)
        rhs_value = self.RHS(u)(*inputs)
        lhs_value = self.LHS(u)(*inputs)
        return jnp.square(rhs_value - lhs_value)


class initial(boundary):
    # alias for boundary for redability
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def domain(a: float = 0.0, b: float = 1.0,
           open_a: bool = True,
           open_b: bool = True,
           rng=random.key(1)):

    while True:
        rng, subkey = random.split(rng)
        sample = random.uniform(subkey, shape=(), minval=a, maxval=b)
        if not open_a and jnp.isclose(sample, jnp.array((a))):
            pass
        if not open_b and jnp.isclose(sample, jnp.aray((b))):
            pass
        else:
            yield sample
