import jax.numpy as jnp
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np



def make_discrete_laplacian(size, dt):
    return (-jnp.eye(size)*2 + jnp.eye(size, k=1) + jnp.eye(size, k=-1))/(dt**2)

def step(xts, u, key, ds, hyperparams):
    dt = hyperparams['dt']
    discrete_laplacian = make_discrete_laplacian(hyperparams['num_steps'], dt)

    I = jnp.eye(hyperparams['num_steps'])
    L = (I - 0.25*ds*discrete_laplacian)
    R = (I + 0.25*ds*discrete_laplacian)
    L_inv = jnp.linalg.inv(L)

    jacobian_u = jax.jacfwd(u)

    M_part_1 = -0.5*ds*jax.vmap(lambda k: jacobian_u(k) @ u(k))(xts)
    M_part_2 = -0.5*ds*jax.vmap(jax.grad(lambda k: jnp.trace(jacobian_u(k))) )(xts)

    noise = jnp.sqrt(2 * (ds/dt))*jax.random.normal(key, shape=xts.shape)

    xts_ds = L_inv @ (R @ xts + M_part_1 + M_part_2 + noise)
    xts_ds = xts_ds.at[0].set(-1)
    xts_ds = xts_ds.at[-1].set(1)
    return xts_ds

def refine_spde_brownian(xts, V, s, num_steps, key, ds, hyperparams):

    u = lambda x: -s*jax.grad(V)(x)


    for i in range(num_steps):
        key = jax.random.fold_in(key, i)
        xts = step(
            xts=xts,
            u=u,
            key=key,
            ds=ds,
            hyperparams=hyperparams
        )

    return xts