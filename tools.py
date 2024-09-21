import jax
import jax.numpy as jnp
import jax.random as random

from typing import Callable, Generator

from differentials import domain

def approx_domain_limits(domain: Generator, n=10000):
    vector = jnp.array([next(domain) for _ in range(n)])
    return jnp.min(vector), jnp.max(vector)

def visualize_3d(u: Callable, expression, defenition: int = 1000):

    assert len(expression.domains) == len(expression.domains_raw.keys())
    approximated_domains = list()

    for domain in expression.domains:
        minimum, maximum = approx_domain_limits(domain)
        approximated_domains.append(jnp.linspace(minimum, maximum, num=defenition))
    
    xs = list()
    for domain in approximated_domains: # iterates over j
        vector = []
        for ij in domain:  # iterates over i, so ij
            vector.append(ij)
        xs.append(vector)
    xs = jnp.array(xs).T
    ys = jax.vmap(lambda x: u(x))(xs)
    data = jnp.concat((xs, ys), axis=1)
    print(data)
