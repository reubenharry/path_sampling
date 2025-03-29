import jax
import jax.numpy as jnp
from nn import MLP, train
import numpy as np
import matplotlib.pyplot as plt
import optax 

# number of paths sampled in one run of calculating dbds. We could handle with parallelization (pmap)
batch_size = 1000
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(batch_size)

# currently I'm just doing an SDE on \mathbb{R}
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


# currently set for TPS
def J(x,y):
    return (x[-1, :] > 0.).astype(jnp.float32)

def E_J(paths, obs):
    return jnp.mean(jax.vmap(J)(paths, obs),axis=0)

def make_h(b, dbds):

    @jax.jit
    def h(x, time):
        # x: [steps, ndims]
        # time: [steps]

        dt = time[0]
        # TODO: consider whether this treatment of the gradient is correct
        dxdts = jnp.concatenate([jnp.zeros((1, x.shape[1])) , x[1:, :] - x[:-1, :]], axis=0)

        # calculate divergence of dbds at all times, using the trace of the jacobian
        div_dbdss = jax.vmap(lambda xt, t: jnp.trace(jax.jacobian(lambda k: dbds(jnp.expand_dims(k,0),t)[0])(xt)))(x, time)

        # the discretized integral
        out = jax.vmap(lambda x, t, dxdt, div_dbds: 
                    
                    -0.5*dt*((b(x,t) - dxdt).dot(dbds(jnp.expand_dims(x,0),t)[0]) + 0.5*div_dbds)
                    )(x,time, dxdts, div_dbdss)
        
        return jnp.sum(out) 


    return h

# compute the loss function, which here is the square of the difference between the left and right of (7)
def make_h_loss(expectation_of_J, b):
    
    def h_loss(dbds, xs, times, ys):


        h = make_h(b, dbds)

        expectation_of_h = jnp.mean(jax.vmap(h)(xs, times), axis=0)


        return jnp.mean(jax.vmap(lambda x,y, t: (-J(x, y)+expectation_of_J - h(x,t) + expectation_of_h)**2)(xs, ys, times))

    return h_loss        

# parametrize dbds by a neural net, and minimize h_loss
def find_dbds(model_key, expectation_of_J, b, xs, times, ys, num_training_steps):

    # initialize model
    dbds = MLP([2,20,20,1], key=model_key)
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)

    h_loss = make_h_loss(expectation_of_J, b)

    dbds = train(h_loss, dbds, optimizer, num_training_steps, xs, times, ys)

    return dbds

# b \mapsto b + dbds
def update(b, hyperparams, key):

    # b : [ndims] -> [1]

    path_key, model_key = jax.random.split(key)

    xs, times = jax.pmap(lambda key:sample_sde(
    b=b, 
    W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key, shape=(ndims,)),
    rho = lambda key: jnp.ones((ndims,))-2.,
    key=key, 
    dt=hyperparams['dt'], 
    num_steps=hyperparams['num_steps']))(jax.random.split(path_key, batch_size))


    x = jnp.expand_dims(jnp.linspace(-5, 5, 100), 1)
    y = (x**4 - 8 * x**2)
    # y2 = jax.grad(lambda x: jnp.sum(x**4 - 8 * x**2))(x)
    y2 = jax.vmap(lambda k: b(k, 0.1))(x)
    plt.ylim(-15,15)
    plt.plot(x, y)
    plt.plot(x,y2)

    # time = np.arange(num_steps)*dt
    # plt.scatter(path, time )
    plt.hist(xs[0], bins=100, density=True)
    plt.savefig('potential_new.png')



    # print("time", times.shape)
    # print("xs", xs.shape)
    
    ys = (xs > 0).astype(jnp.float32)

    expectation_of_J = E_J(xs, ys) # .mean(axis=0)

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
    

    test_xs, test_times = jax.pmap(lambda key:sample_sde(
        b=b, 
        W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key, shape=(ndims,)),
        rho = lambda key: jnp.ones((ndims,))-2.,
        key=key, 
        dt=hyperparams['dt'], 
        num_steps=hyperparams['num_steps']))(jax.random.split(jax.random.key(500), batch_size))

    test_ys = (xs > 0).astype(jnp.float32)


    print(make_h_loss(expectation_of_J=expectation_of_J, b=b)(
        dbds,
        test_xs,
        test_times,
        test_ys
    ), "LOSS\n")

    ## TODO: you need to pass in the algorithmic time!!! or not?
    
    new_b =  lambda x, t: (b(x,t) + dbds(jnp.expand_dims(x, 0),t)[0])
    # new_b =  lambda x, t: (dbds(jnp.expand_dims(x, 0)[0],t))

    # print(b(xs[0, 0, :],0.1).shape)
    # print(new_b(xs[0, 0, :],0.1).shape)
    # print(jax.grad(lambda x: jnp.sum(x**4 - 8 * x**2,axis=0))(xs[0, 0, :]).shape)
    # print(dbds(jnp.expand_dims(xs[0, 0, :],0),0.1)[0].shape)

    # new_b(xs[:, 0, :],0.1)

    return new_b

   
potential = lambda x: jnp.sum(x**4 - 8 * x**2,axis=0)

schedule = [0.0, 0.1]


def b(x, t): 
    assert x.shape[0] == ndims
    return -jax.grad(potential)(x)


key = jax.random.key(0)
for i, _ in enumerate(schedule):



    key = jax.random.fold_in(key, i)

    b = update(b=b,
        hyperparams={'dt': 0.01, 'num_steps': 50, 'num_training_steps' : 1000},
        key=key
    )
    print("NEXT ITER")














#     # dbds = find_dbds(s, b)



# # double well potential
# potential = lambda x: (x**4 - 8 * x**2)[0]

# # plot potential
# dt = 0.01
# path  = sample_sde(
#     b=lambda x: -jax.grad(potential)(x), 
#     W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key, shape=(1,)),
#     rho = lambda key: -2.,
#     key=jax.random.key(0)
#     )(dt, num_steps)

# # paths = jax.pmap(lambda key: sample_sde(
# #     b=lambda x: -jax.grad(potential)(x), 
# #     W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key),
# #     rho = lambda key: jax.random.normal(key),
# #     key=key,
# #     )(dt, 1000)
# #     )(jax.random.split(jax.random.key(0), batch_size))


# # print(E_J(paths, None))

# x = np.linspace(-5, 5, 100)
# y = potential(x)
# plt.ylim(-10,10)
# plt.plot(x, y)

# time = np.arange(num_steps)*dt
# # plt.scatter(path, time )
# plt.hist(path, bins=30, density=True)
# plt.savefig('potential.png')
