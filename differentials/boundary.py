from jax.numpy import jnp
from typing import List


class boundary:
    def __init__(self, *expressions, **fixed):
        # want to return a boundary class
        #loaded with a set of expressions, that collapse into functions
        
        # boundary class needs to have input of shape.
        # shape can be inherited from parent function
        #initials 
        self.expressions = expressions
        self.fixed = fixed

    def make_loss_fn(self, vars):
        coefficients_vector = list()
        value_vector = list()
        for xi in vars:
            if xi in list(self.fixed.keys()):
                coefficients_vector.append(0)
                value_vector.append(self.fixed[xi])
            else:
                coefficients_vector.append(1)

        # defined in fixed

        

        # self.expressions, are functions of the form
        # fn(u)(x)
        expr = self.expressions[0]
        return lambda u: lambda x: expr(u)(jnp.dot(jnp.array(coefficients_vector), x)+jnp.array(value_vector))

            


