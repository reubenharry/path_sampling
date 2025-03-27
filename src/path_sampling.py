import jax
import jax.numpy as jnp
from nn import MLP, train
import numpy as np
import matplotlib.pyplot as plt
import optax 

batch_size = 8
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)

ndims = 1

def sample_sde(b, W, rho, dt, num_steps, key):

    init_key, run_key = jax.random.split(key)
    X_0 = rho(init_key)

    def body(X_and_t, key):
        X, time = X_and_t

        new_X = X + b(X, time) * dt + W(X_0, key) * jnp.sqrt(dt)
        new_time = time + dt
        return ((new_X, new_time), (new_X, new_time))

    keys = jax.random.split(run_key, num_steps)
    return jax.lax.scan(body, (X_0, 0), keys)[1]


def J(x,y):
    return (x[-1, :] > 2.).astype(jnp.float32)

def E_J(paths, obs):
    return jnp.mean(J(paths, obs),axis=1)
    # return -J(paths) # todo: what is J in 7??

def make_h(b, dbds):

    @jax.jit
    def h(x, time):
        # shapes
        # x: [steps, ndims]
        # time: [steps]

        # print(x.shape, time.shape, "x, time")

        dt = time[0]
        # print(dt, "dt")

        # dxdt = x[1:, :] - x[:-1, :]
        dxdts = jnp.concatenate([jnp.zeros((1, x.shape[1])) , x[1:, :] - x[:-1, :]], axis=0)
        # print("xdot shape", dxdts.shape)

        # calculate divergence of dbds
        div_dbdss = jax.vmap(lambda x, t: jnp.trace(jax.jacobian(lambda x: b(x,t))(x)))(x, time)
        # print(div_dbdss.shape)

        out = jax.vmap(lambda x, t, dxdt, div_dbds: 
                    
                    -0.5*dt*((b(x,t) - dxdt).dot(dbds(jnp.expand_dims(x,0),jnp.expand_dims(t,0))[0]) + 0.5*div_dbds)
                    )(x,time, dxdts, div_dbdss)
        return jnp.sum(out) 
        # raise Exception
        
        return -0.5*dt*jnp.sum(jnp.array([(b(t,x) - dxdt).dot(dbds(t, x[t]) + 0.5*div_dbds) for t in range(num_steps)]))


        # dbds = jax.grad(b)(x)
        # (b(x) - dxdt). dot(dbds) + 0.5*dbds
        pass

    return h

def make_h_loss(expectation_of_J, b):
    
    def h_loss(dbds, xs, times, ys):

        # jax.debug.print("xs {x}", x=(xs[0].shape, times[0].shape))
        # print(make_h(b,dbds)(xs[0], times[0]))
        # return 1.0

        h = make_h(b, dbds)

        expectation_of_h = jnp.mean(jax.vmap(h)(xs, times), axis=0)


        return jnp.mean(jax.vmap(lambda x,y, t: (-J(x, y)+expectation_of_J - h(x,t) + expectation_of_h)**2)(xs, ys, times))

    return h_loss        

def find_dbds(model_key, expectation_of_J, b, xs, times, ys, num_training_steps):

    model_key, data_key = jax.random.split(model_key)

    # initialize model
    dbds = MLP([2,20,20,1], key=model_key)
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)

    h_loss = make_h_loss(expectation_of_J, b)

    dbds = train(h_loss, dbds, optimizer, num_training_steps, xs, times, ys)

    return dbds

def update(b, hyperparams, key):

    model_key, init_key = jax.random.split(key)

    optimizer = optax.adam(1e-3)
    # Obtain the `opt_state` that contains statistics for the optimizer.
    params = {'w': jax.random.normal(key=init_key, shape=(hyperparams['num_steps'],ndims))}
    opt_state = optimizer.init(params)

    xs, times = jax.pmap(lambda key:sample_sde(
    b=b, 
    W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key, shape=(ndims,)),
    rho = lambda key: jnp.ones((ndims,))-2.,
    key=key, 
    dt=hyperparams['dt'], 
    num_steps=hyperparams['num_steps']))(jax.random.split(jax.random.key(0), batch_size))


    x = np.linspace(-5, 5, 100)
    y = (x**4 - 8 * x**2)
    plt.ylim(-10,10)
    plt.plot(x, y)

    # time = np.arange(num_steps)*dt
    # plt.scatter(path, time )
    plt.hist(xs[0], bins=100, density=True)
    plt.savefig('potential_new.png')



    # print("time", times.shape)
    # print("xs", xs.shape)
    
    ys = (xs > 0).astype(jnp.float32)

    expectation_of_J = E_J(xs, ys).mean(axis=0)
    # dbds = MLP([2,20,20,1], key=model_key)

    # print("input to thing", xs[0].shape, times[0].shape)
    # print(make_h(b, dbds)(xs[0], times[0]))
    # h = make_h(b, dbds)
    
    # print(expectation_of_J.shape)
    


    # print(b(xs[0][0],0.0))
    # print(dbds(jnp.expand_dims(xs[0][0],0), 0.0))


    
    


    # print(make_h_loss(expectation_of_J, b)(
    #     dbds, 
    #     xs, 
    #     times,
    #     ys,
    #     ))
    

    dbds = find_dbds(
        model_key,
        expectation_of_J,
        b,
        xs,
        times,
        ys,
        num_training_steps=hyperparams['num_training_steps']
        )
    
    print(dbds(xs[:, 0, :],0.1))
    print(dbds(xs[:, 0, :],0.01))

    ## TODO: you need to pass in the algorithmic time!!! or not?
    # raise Exception
    
    # print("dbds", dbds)

    
    





    # for i in range(2):

    #     # loss = lambda params: jnp.sum(params['w'])

    #     grads = jax.grad(loss)(params, b=b_tabulated, xs=x, ys=y)
    #     # print(grads)
    #     updates, opt_state = optimizer.update(grads, opt_state)
    #     print(params)
    #     params = optax.apply_updates(params, updates)

    return lambda x, t: b(x,t) + dbds(jnp.expand_dims(x, 0),t)[0]
 



    # paths = jax.pmap(lambda key: sample_sde(
    #     b=lambda x: -jax.grad(potential)(x), 
    #     W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key),
    #     rho = lambda key: jax.random.normal(key),
    #     key=key,
    #     )(dt, 1000)
    #     )(jax.random.split(jax.random.key(0), batch_size))
    
    # minimize loss

potential = lambda x: (x**4 - 8 * x**2)[0]
# path  = sample_sde(
#     b=lambda x: -jax.grad(potential)(x), 
#     W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key, shape=(1,)),
#     rho = lambda key: jnp.ones((1,))-2.,
#     key=jax.random.key(0)
#     )(0.01, num_steps)

# print(path.shape)

schedule = [0.0, 0.1]

b = lambda x, t: -jax.grad(potential)(x)
key = jax.random.key(0)
for i, _ in enumerate(schedule):

    key = jax.random.fold_in(key, i)

    b = update(b=b,
        hyperparams={'dt': 0.01, 'num_steps': 1000, 'num_training_steps' : 1000},
        key=key
    )
    print("NEXT ITER")


raise Exception



    # dbds = find_dbds(s, b)



# double well potential
potential = lambda x: (x**4 - 8 * x**2)[0]

# plot potential
dt = 0.01
path  = sample_sde(
    b=lambda x: -jax.grad(potential)(x), 
    W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key, shape=(1,)),
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
