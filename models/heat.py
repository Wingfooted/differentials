import jax.numpy as jnp
import jax.random as random

import load

if __name__ == '__main__':
    model, params = load.load_model("bins/heat.bin")
    print(params)
    print(model.apply(params, jnp.array((0.0, 0.0))))
    #bruh = load.Model((20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20))

    #test = bruh.init(random.key(0), jnp.array((1, 1)))

    x = random.normal(random.key(0), (2,))
    y = model.apply(params, x)

    load.visualize_3d(lambda x: model.apply(params, x),
<<<<<<< HEAD
                 jnp.linspace(-5, 5, 100),
                 jnp.linspace(0, 5, 100),
=======
                 jnp.linspace(-1, 1, 100),
                 jnp.linspace(0, 1, 100),
>>>>>>> new
                 defenition=100)

