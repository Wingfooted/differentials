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
    def __new__(self, minval, maxval, key=random.key(1), sampling_method="UNIFORM"):
        if sampling_method == "UNIFORM":
            return uniform(minval, maxval, key)
        if sampling_method == "NORMAL":
            return normal(key)


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



