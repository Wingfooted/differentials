import jax.numpy as jnp
import jax.random as random

import load

if __name__ == '__main__':
    model, params = load.load_model("bins/heat.bin")
    bruh = load.Model((20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20))

    test = bruh.init(random.key(0), jnp.array((1, 1)))

    x = random.normal(random.key(0), (2,))
    y = model.apply(params, x)

    load.visualize_3d(lambda x: model.apply(params, x),
                 jnp.linspace(-1, 1, 100),
                 jnp.linspace(0, 3, 100),
                 defenition=100)

