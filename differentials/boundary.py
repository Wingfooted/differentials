import jax
import jax.numpy as jnp

from typing import Callable

class function:
    def __new__(self):
        # this has to be a term -> function
        return term(lambda u: lambda x: u(x))
        pass

    def __call__(self):
        pass

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

    def __sub__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) - corrected_other(u)(x))

    def __pow__(self, other):
        pass

    def __mul__(self, other):
        corrected_other = self.convert_other(other)
        return term(lambda u: lambda x: self(u)(x) * corrected_other(u)(x))

    def __truediv__(self, other):
        corrected_other = self.convert_other(other)
        print("true_div")
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
    def __new__(self, operation):
        return term(lambda u: lambda x: operation(x))


dx = derivative(0)
dt = derivative(1)
sin = manual_function(lambda x: jnp.sin(x))
