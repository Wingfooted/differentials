import jax
import jax.random as random
import jax.numpy as jnp


def uniform(minval, maxval, key):
    key, subkey = random.split(key)
    yield random.uniform(subkey, (1,), minval, maxval)


def normal(key):
    key, subkey = random.split(key)
    yield random.normal(subkey)


class return_domain:
    def __init__(self, minval, maxval, key=random.key(1), sampling_method="UNIFORM"):
        self.minval = minval
        self.maxval = maxval
        if sampling_method == "UNIFORM":
            self.gen = uniform(minval, maxval, key)
        if sampling_method == "NORMAL":
            self.gen = normal(key)
    def __call__(self):
        return next(self.gen)


class domain:
    def __new__(self, *args):
        # ARG SORTING
        minval = False
        maxval = False

        for arg in args:
            if isinstance(arg, str):
                return return_domain(-jnp.inf, jnp.inf, key=random.key(1), sampling_method= "NORMAL")
            else:
                vals = jnp.array(args)
                return return_domain(jnp.min(vals),
                                     jnp.max(vals))



