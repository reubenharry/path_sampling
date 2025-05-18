import jax.numpy as jnp
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np


## this vesion works specifically for Brownian motion -> Brownian Bridge transport


# define discrete Laplacian matrix
def make_discrete_laplacian(size, dt):
    return (-jnp.eye(size)*2 + jnp.eye(size, k=1) + jnp.eye(size, k=-1))/(dt**2)

# used for MH
J = lambda xts, grad_V, dt: jnp.sum(-dt*jax.vmap(lambda xt: 0.25*grad_V(xt).dot(grad_V(xt)) - 0.5*jnp.trace(jax.jacfwd(lambda k: grad_V(k))(xt)) )(xts))

def pi(xts, grad_V, s, hyperparams):
    dt = hyperparams['dt']
    A = make_discrete_laplacian(hyperparams['num_steps'], dt)
    part1 = s*J(xts, grad_V, dt)
    part2 = jnp.sum(0.25*dt*jax.vmap(lambda xt: xt @ A @ xt , in_axes=-1)(xts))
    return part1 + part2

# main update function
def step(xts, potential, s, ds, A, key, hyperparams, mh=False, prior= 'sde_prior'):

    grad_V = lambda x: jax.grad(potential)(x)   # gradient of the prior potential

    u = lambda x: -grad_V(x)

    # define the main quantities
    dt = hyperparams['dt']
    discrete_laplacian = make_discrete_laplacian(hyperparams['num_steps'], dt)

    I = jnp.eye(hyperparams['num_steps'])
    L = I 
    R = (I + 0.5*ds*discrete_laplacian)
    L_inv = jnp.linalg.inv(L)

    jacobian_u = jax.jacfwd(u)
    M_part_1 = -0.5*ds*jax.vmap(lambda k: jacobian_u(k) @ u(k))(xts)
    M_part_2 = -0.5*ds*jax.vmap(jax.grad(lambda k: jnp.trace(jacobian_u(k))))(xts)

    noise = jnp.sqrt(2 * (ds/dt))*jax.random.normal(key, shape=xts.shape)

    # updated path
    #xts_ds = L_inv @ (R @ xts + M_part_1 + M_part_2 + noise)
    sigma = 0.1  
    likelihood = ((s*ds)/(sigma**2)) *(2.0 - xts[-2])   # constraint on the second-to-last point
    #xts_ds = R @ xts + M_part_1 + M_part_2 + likelihood + noise
    xts_ds = R @ xts + likelihood + noise
    
    
    # impose the boundary condition
    if prior=='brownian':
        xts_ds = xts_ds.at[0].set(-1)
        xts_ds = xts_ds.at[-1].set(1)
    elif prior=='sde_prior':
        #sigma = 0.1   
        # change the initial point to -2
        xts_ds = xts_ds.at[0].set(-2)   
        # change the traget point to +2
        #xts_ds = xts_ds.at[-1].set(xts_ds[-2] + dt*(u(xts_ds[-2]) + ((2.*s)/(sigma**2) )*((2.0 - xts_ds[-2])) )) 
        xts_ds = xts_ds.at[-1].set(xts_ds[-2])  # last 2 points are the same
        # xts_ds = xts_ds.at[-2].set(xts_ds[-2] + dt*(u(xts_ds[-1]) + ((2.*s)/(0.01**2) )*((1 - xts_ds[-1])) )) #todo: pass in sigma

    # MH adjustment
    def q(xts, xts_prime):
        return (-(dt/(4*ds))*jnp.linalg.norm(L@xts_prime - R@xts + (M_part_1 + M_part_2))**2)

    if mh:
        log_W = (pi(xts_ds, grad_V, s, hyperparams)+q(xts_ds, xts)) - (pi(xts, grad_V, s, hyperparams)+q(xts, xts_ds))
        accept_prob = jnp.clip(jnp.exp(log_W), 0., 1.)
        # print(accept_prob, "accept_prob")
        # print(f"pi(xts_ds, grad_V, s, hyperparams): {pi(xts_ds, grad_V, s, hyperparams)}")
        # print(f"pi(xts, grad_V, s, hyperparams): {pi(xts, grad_V, s, hyperparams)}")
        # print(f"q(xts_ds, xts): {q(xts_ds, xts)}")
        # print(f"q(xts, xts_ds): {q(xts, xts_ds)}")
        # raise Exception("stop")
        # print("pi", pi(xts_ds, u, hyperparams))
        accept = jax.random.uniform(key) < accept_prob
        # print(accept, "accept")
        xts_ds = xts_ds*accept + xts*(1-accept)


    A = A - ds*J(xts_ds, grad_V, dt)

    return xts_ds, A


def refine_spde(xts, V, s, A, num_steps, key, ds, hyperparams, mh, prior= 'sde_prior'):

    # u = lambda x: -s*jax.grad(V)(x)

    for i in range(num_steps):
        key = jax.random.fold_in(key, i)
        xts, A = step(
            xts=xts,
            # u=u,
            potential=V,   # prior potential
            A=A,
            s=s,
            key=key,
            ds=ds,
            hyperparams=hyperparams,
            mh=mh,
            prior=prior
        )

    return xts, A

