from expression import term, derivative, manual_function

import jax
import jax.numpy as jnp


class function:
    def __new__(self):
        # this has to be a term -> function
        return term(lambda u: lambda x: u(x))
        pass

    def __call__(self):
        pass


dx = derivative(0)
dt = derivative(1)

abs = manual_function(jnp.abs)
absolute = manual_function(jnp.abs)
acos = manual_function(jnp.acos)
acosh = manual_function(jnp.acosh)
arccos = manual_function(jnp.arccos)
arccosh = manual_function(jnp.arccosh)
arcsin = manual_function(jnp.arcsin)
arcsinh = manual_function(jnp.arcsinh)
arctan = manual_function(jnp.arctan)
arctanh = manual_function(jnp.arctanh)
asin = manual_function(jnp.asin)
asinh = manual_function(jnp.asinh)
atan = manual_function(jnp.atan)
atanh = manual_function(jnp.atanh)
cbrt = manual_function(jnp.cbrt)
conj = manual_function(jnp.conj)
conjugate = manual_function(jnp.conjugate)  # Fixed spelling
cos = manual_function(jnp.cos)
cosh = manual_function(jnp.cosh)
exp = manual_function(jnp.exp)
exp2 = manual_function(jnp.exp2)
expm1 = manual_function(jnp.expm1)
fabs = manual_function(jnp.fabs)
floor = manual_function(jnp.floor)
log = manual_function(jnp.log)
log10 = manual_function(jnp.log10)
log1p = manual_function(jnp.log1p)
log2 = manual_function(jnp.log2)
negative = manual_function(jnp.negative)
positive = manual_function(jnp.positive)
sign = manual_function(jnp.sign)
sin = manual_function(jnp.sin)
sinc = manual_function(jnp.sinc)
sinh = manual_function(jnp.sinh)
sqrt = manual_function(jnp.sqrt)
square = manual_function(jnp.square)
tan = manual_function(jnp.tan)
tanh = manual_function(jnp.tanh)

sin = manual_function(jnp.sin)




