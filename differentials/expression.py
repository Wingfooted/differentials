import jax
import jax.numpy as jnp
import jax.random as random

from typing import Callable, Tuple, Sequence
from .model import Model

from .domain import domain


class term:
    #function(u) -> function(x). 
    def __init__(self, func):
        self.func = func

    def __call__(self, u):
        return lambda x: self.func(u)(x)

    def __eq__(self, other):
        # evaluated RHS - LHS
        if isinstance(other, term) and isinstance(self, term):
            # self works, problem term is other
            return lambda u: lambda x: other(u)(x) - self(u)(x)
            return lambda u: lambda x: other(u)(x)
            return lambda u: lambda x: self(u)(x)
        elif not isinstance(other, term):
            # turn other into a term
            const_term = term(lambda u: lambda x: other)
            return self.__eq__(const_term)

    def __add__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) + corrected_other(u)(x))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) - corrected_other(u)(x))

    def __rsub__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) - corrected_other(u)(x))

    def __pow__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) ** corrected_other(u)(x))

    def __rpow__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) ** corrected_other(u)(x))

    def __mul__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) * corrected_other(u)(x))

    def __rmul__(self, other):
        return self.__add__(other) # abelian

    def __truediv__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) / corrected_other(u)(x))

    def __rtruediv__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) / corrected_other(u)(x))

    def convert_other(self, other):
        if isinstance(other, term):
            return other
        else:
            const_term = term(lambda u: lambda x: other)
            return const_term

class derivative:
    # function. term -> term
    def __init__(self, arg):
        self.arg = arg

    def __call__(self, target):

        # subscript when give x
        return term(lambda u:
                    lambda x: jax.grad(u)(x)[self.arg]
                    )

        return term(lambda u: jax.grad(target(u), argnums=self.arg))


class manual_function:
    # term -> term
    def __init__(self, operation: Callable):
        self.operation = operation

    def __call__(self, target):
        return term(lambda u:
                    lambda x: self.operation(u(x))
                    )

class expression:
    def __init__(self,
                 *loss_fns,
                 **domains):
        self.number_independent = len(domains.keys())
        self.loss_conditions = loss_fns
        self.kwargs = domains
        self.domains = [domain("XER") if d == True else d for d in domains.values()]
        self.vars = domains.keys()

    def loss(self, u: Callable, x: jax.Array):
        # where x is a vector
        total_loss = jnp.array((0))
        for fn in self.loss_conditions:
            total_loss += jnp.abs(fn(u)(x))
        return total_loss

    def u(self, struct: Sequence[int] = (8, 8, 8, 8), start_point: int = 0) -> Tuple:
        schema = (len(self.vars), *struct)
        u_hat = Model(schema)
        forward_rng, model_rng = random.split(random.key(start_point))
        x = random.uniform(forward_rng, (self.number_independent,))
        params = u_hat.init(model_rng, x)
        return u_hat, params
