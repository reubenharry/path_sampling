import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

batch_size = 8
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)


def sample_sde(b, W, rho, key):

    init_key, run_key = jax.random.split(key)
    X_0 = rho(init_key)

    def f(dt, num_steps):

        output = lambda X, key: X + b(X) * dt + W(X_0, key) * jnp.sqrt(dt)
        body = lambda X, key: (output(X, key), output(X, key))

        keys = jax.random.split(run_key, num_steps)

        return jax.lax.scan(body, X_0, keys)[1]

    return f

def J(x,y):
    return x > 0

def E_J(paths, obs):
    return jnp.mean(J(paths, obs),axis=1)
    # return -J(paths) # todo: what is J in 7??

def h(b, x):
    dbds = jax.grad(b)(x)
    (b(x) - dxdt). dot(dbds) + 0.5*dbds
    pass

def loss(path):
    pass

def change_s(b_0, s):

    paths = jax.pmap(lambda key: sample_sde(
        b=lambda x: -jax.grad(potential)(x), 
        W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key),
        rho = lambda key: jax.random.normal(key),
        key=key,
        )(dt, 1000)
        )(jax.random.split(jax.random.key(0), batch_size))
    
    e_j = E_J(paths, None)
    e_h = None

    minimize = lambda s: e_j + e_h 


    dbds = find_dbds(s, b)
    return b + dbds



# double well potential
potential = lambda x: x**4 - 8 * x**2

# plot potential
dt = 0.01
num_steps = 50000
path  = sample_sde(
    b=lambda x: -jax.grad(potential)(x), 
    W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key),
    rho = lambda key: -2.,
    key=jax.random.key(0)
    )(dt, num_steps)

# paths = jax.pmap(lambda key: sample_sde(
#     b=lambda x: -jax.grad(potential)(x), 
#     W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key),
#     rho = lambda key: jax.random.normal(key),
#     key=key,
#     )(dt, 1000)
#     )(jax.random.split(jax.random.key(0), batch_size))


# print(E_J(paths, None))

x = np.linspace(-5, 5, 100)
y = potential(x)
plt.ylim(-10,10)
plt.plot(x, y)

time = np.arange(num_steps)*dt
# plt.scatter(path, time )
plt.hist(path, bins=30, density=True)
plt.savefig('potential.png')
