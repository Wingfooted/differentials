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
        if isinstance((self, other), term):
            return lambda u: lambda x: other(u)(x) - self(u)(x)

        else:
            if isinstance(other, int) or isinstance(other, float):
                self.__eq__(term(lambda u: lambda x: jax.array((other))), self)


    def __add__(self, other):
        pass


class derivative:
    def __init__(self, arg):
        self.arg = arg

    def __call__(self, target):

        # subscript when give x
        return term(lambda u:
                    lambda x: jax.grad(u)(x)[self.arg]
                    )

        return term(lambda u: jax.grad(target(u), argnums=self.arg))

class function:
    def __init__(self):
        pass


dx = derivative(0)
dt = derivative(1)
sin = function()

