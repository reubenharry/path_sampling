import jax
import jax.numpy as jnp
from nn import MLP, train
import numpy as np
import matplotlib.pyplot as plt
import optax 
import scipy
import os
from functools import partial
import itertools

from spde import refine_spde


def sample_sde(b, W, rho, dt, num_steps, key):

    """
    Sample a path of the SDE defined by b, W, and rho.
    b: drift term. A function from R^ndims x R -> R^ndims
    W: noise term
    rho: initial distribution to sample from
    dt: time step size
    num_steps: number of time steps to take along the path
    key: random key for sampling

    Returns:

    array of size [batch_size, num_steps, ndims] containing the sampled paths
    """

    init_key, run_key = jax.random.split(key)
    X_0 = rho(init_key)

    def body(X_and_t, key):
        X, time = X_and_t

        new_X = X + b(X, time) * dt + W(X_0, key) * jnp.sqrt(dt)
        new_time = time + dt
        return ((new_X, new_time), (new_X, new_time))

    keys = jax.random.split(run_key, num_steps-1)
    path, time =  jax.lax.scan(body, (X_0, 0), keys)[1]
    return jnp.concat((X_0[None], path), axis=0), jnp.concat((jnp.zeros(1,), time), axis=0)

def make_double_well_potential(v):
   
    return lambda x: jnp.sum(v*(x**2 - 1)**2, axis=-1)

# vmap is a jax function that allows us to apply a function to each element of an array
def E_J(J, paths, obs):
    """
    paths: array of shape [batch_size, num_steps, ndims]
    obs: here can just be None
    Returns:
        a scalar value representing the mean cost of the paths
    """
    return jnp.mean(jax.vmap(J)(paths, obs),axis=0)

def div_f(x,time, f):
    """
    x: path of the SDE, array of shape [num_steps, ndims]
    time: array of shape [num_steps]
    f: a function from R^ndims x R -> R^ndims
    Returns:
        an array of scalar values representing the divergence of f at x[i] and time[i]
    """
    return jax.vmap(lambda xt, t: jnp.trace(jax.jacobian(lambda k: f(k,t))(xt)))(x, time)

def dfdt(x, dt):
    """
    x: path of the SDE, array of shape [num_steps, ndims]
    dt: time step size
    Returns:
        an array of shape [num_steps, ndims] representing the derivative of x with respect to time
    """
    return jnp.concatenate([jnp.zeros((1, x.shape[1])) , x[1:, :] - x[:-1, :]], axis=0)/ dt




def make_h(b, dbds, s):
    """
    b: drift term. A function from R^ndims x R -> R^ndims
    dbds: also a function from R^ndims x R -> R^ndims
    Returns:
        a function h, as defined below
    """

    @jax.jit # this speeds up the function
    def h(x, time):
        """
        x: path of the SDE, array of shape [num_steps, ndims]
        time: array of shape [num_steps]
        Returns:
            a scalar value representing the cost of the path
        """
        dt = time[1]

        # a crude approximation of the derivative
        dxdts = dfdt(x, dt)

        # calculate divergence of dbds at all times, using the trace of the jacobian
        # div_dbdss = jax.vmap(lambda xt, t: jnp.trace(jax.jacobian(lambda k: dbds(k,t))(xt)))(x, time)
        div_dbdss = div_f(x,time,partial(dbds, s=s))

        # the discretized integral
        out = jax.vmap(lambda x, t, dxdt, div_dbds: 
                    
                    -0.5*dt*((b(x,t) - dxdt).dot(dbds(x,t, s)) + 0.5*div_dbds)
                    )(x,time, dxdts, div_dbdss)
        
        return jnp.sum(out) 

    return h

# compute the loss function, which here is the square of the difference between the left and right of (7)
def make_h_loss(expectation_of_J, J, b, s):
    """
    expectation_of_J: the expectation of J, computed from the paths
    b: drift term. A function from R^ndims x R -> R^ndims
    Returns:
        a function h_loss, which computes the loss
    """
    
    def h_loss(dbds, xs, times, ys):
        """
        xs: array of shape [batch_size, num_steps, ndims]
        times: array of shape [batch_size, num_steps]
        ys: for current purposes, just None
        Returns:
            a scalar value representing the loss
        """

        h = make_h(b, dbds, s)

        expectation_of_h = jnp.mean(jax.vmap(h)(xs, times), axis=0)

        return jnp.sum(jax.vmap(lambda x,y, t: (-J(x, y)+expectation_of_J - h(x,t) + expectation_of_h)**2)(xs, ys, times))

    return h_loss

def find_dbds(dbds, J, s, b, xs, times, ys, num_training_steps):
    """
    model_key: random key for the model
    expectation_of_J: the expectation of J, computed from the paths
    b: drift term. A function from R^ndims x R -> R^ndims
    xs: array of shape [batch_size, num_steps, ndims]
    times: array of shape [batch_size, num_steps]
    ys: for current purposes, just None
    num_training_steps: number of training steps to take
    Returns:
        dbds: the trained model, which is a function from R^ndims x R -> R^ndims
    """

    expectation_of_J = E_J(J, xs, ys)

    # print(s, "s")

    # dbds = MLP([3,20,20,1], key=model_key) # create an NN, with random weights
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    h_loss = make_h_loss(expectation_of_J, J, b, s)
    dbds = train(h_loss, dbds, optimizer, num_training_steps, xs, times, ys)
    return dbds

def make_b(schedule, uref, dbds):

    schedule_padded = np.concatenate([np.zeros((1,)), np.array(schedule)])

    # dss = np.concatenate([np.array([schedule[0]]), np.array(schedule)[1:] - np.array(schedule)[:-1]])
    dss = np.array(schedule_padded)[1:] - np.array(schedule_padded)[:-1]

    # print(dss, "dss")


    # print(list(zip(schedule, dss)), "schedule")

    b = lambda x, t: (uref(x,t) + sum([ds*dbds(x,t,s) for (s,ds) in zip(schedule, dss)]))

    return b 



# b \mapsto b + dbds
# Pseudo code:
# create `xs` of shape [batch_size, num_steps, ndims]
# calculate expectation of J
# initialize a neural net as function from R^ndims x R -> R^ndims
# fit weights of neural net according to the loss function
# calculate test loss
# return lambda x: b(x) + dbds(x)
# from mclmc import refine_path
from path_sampling import E_J, find_dbds, make_b, make_h_loss
import numpy as np

def plot_path(path, time, potential, label, i):

    if path.shape[1] != 2:

        plt.plot(path, time, label=label)
        x = jnp.expand_dims(jnp.linspace(-2, 2, 100), 1)
        y = potential(x)
        if i==0: plt.plot(x, y)

    else:

        # plot path in 2D
        plt.plot(path[:, 0], path[:, 1], label=label)
        plt.legend()
        
        
        # plot heatmap of the potential make_double_well_potential
        v = 1.0
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        # print(X.shape)
        # use vmap to apply the potential to the grid
        Z = jax.vmap(lambda x,y: potential(jnp.array([x,y])))(X.reshape(-1), Y.reshape(-1)).reshape(X.shape)
        plt.contourf(X, Y, Z, levels=50)

def update(V, uref, J, prior, dbds, hyperparams, key, schedule, i, A, rho = lambda key: jnp.zeros((1,))-1., refine=False, ndims=1):
    """
    b: drift term. A function from R^ndims x R -> R^ndims
    hyperparams: dictionary of hyperparameters
    key: random key for the model
    i: index for the current iteration (just for labelling plots)
    Returns:
        new_b: the updated drift term, which is a function from R^ndims x R -> R^ndims
    """

    

    new_s = schedule[i]
    old_s = schedule[i-1] if i>0 else 0.0
    ds = new_s - old_s

    b = make_b(schedule[:i], uref, dbds)

    path_key, model_key, refine_key = jax.random.split(key, 3)

    W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key, shape=(ndims,))
    

    # xs : [batch_size, num_steps, ndims]
    xs, times = jax.pmap(lambda key:sample_sde(
    b=b, 
    W = W,
    rho = rho,
    key=key, 
    dt=hyperparams['dt'], 
    num_steps=hyperparams['num_steps']))(jax.random.split(path_key, hyperparams['batch_size']))

    time = np.arange(0,hyperparams['num_steps'])*hyperparams['dt']
    
    print("old s", old_s)

    # path refinement
    if refine:
        # jax.debug.print("shapes 1 {xs.shape}", xs=xs.shape)
        # new_xs = jax.pmap(lambda key, p: refine_path(
        # x=p,
        # b=b,
        # s=old_s,
        # J=J,
        # I=I,
        # time=time,
        # rng_key=key,
        # num_steps=1000
        # ))(jax.random.split(refine_key, hyperparams['batch_size']), xs)[:,:, None]
        # xs = new_xs

        new_xs, _ = jax.pmap(lambda key, p: refine_spde(
        xts=p,
        V=V,
        s=old_s,
        ds=0.001,
        hyperparams=hyperparams,
        key=key,
        num_steps=100,
        prior=prior,
        mh=False,
        A=A,
        ))(jax.random.split(refine_key, hyperparams['batch_size']), xs)
        xs = new_xs
        # jax.debug.print("shapes 2 {xs.shape}", xs=xs.shape)

    # print(E_J(xs, None), "ej x")
    # print(E_J(new_xs, None), "ej new x")
    # print(times)

    # for path in xs:
    #     plt.plot(path,(time/hyperparams['dt'])/10, label='old')

    # plt.plot(xs[0],(time/hyperparams['dt'])/10, label='old')
    # for path in new_xs:
    #     plt.plot(path,(time/hyperparams['dt'])/5, label=f'annealed')
    # plt.plot(new_xs[0],(time/hyperparams['dt'])/10, label=f'annealed{i}')

    # xs = jnp.concatenate((xs[:10, :, :], new_xs[:30, :, :]), axis=0)

    expectation_of_J = E_J(J, xs, None)

    dbds = find_dbds(
        dbds=dbds,
        J=J,
        s=new_s,
        b=b,
        # xs[:90, :, :],
        # times[:90, :],
        xs=xs,
        times=times,
        ys=None,
        num_training_steps=hyperparams['num_training_steps']
        )
    
    ### calculate test loss
    test_xs, test_times = jax.pmap(lambda key:sample_sde(
        b=b, 
        W = W,
        rho = rho,
        key=key, 
        dt=hyperparams['dt'], 
        num_steps=hyperparams['num_steps']))(jax.random.split(jax.random.key(500), hyperparams['batch_size']))

    print(f"Test loss is {make_h_loss(expectation_of_J=expectation_of_J, J=J, b=b, s=new_s)(dbds, test_xs, test_times, None)}")

    # new version
    # new+b = b + [dbds(x,t,s)*delta_s for s in ...]

    # print(ds, "ds")
    
    # new_b =  lambda x, t: (b(x,t) + dbds(x,t, 0.0)*ds)

    # new_b =  lambda x, t: (b(x,t) + dbds(x,t, new_s))


    plot = True
    if plot:
        
        new_b = make_b(schedule[:i+1], b, dbds)
        

        
        potential = make_double_well_potential(v=5.0)
        
        

        path, time = sample_sde(
            b=new_b, 
            W = W,
            rho = rho,
            key=key, 
            dt=hyperparams['dt'], 
            num_steps=hyperparams['num_steps'])
                
        # refined_path = refine_path(
        #     x=path,
        #     s=new_s,
        #     J=J,
        #     b=
        #     I=I,
        #     time=time,
        #     rng_key=jax.random.key(2),
        #     num_steps=10000
        #     )

        # refined, _ = refine_spde(
        # xts=path,
        # V=V,
        # s=old_s,
        # ds=0.001,
        # hyperparams=hyperparams,
        # key=key,
        # num_steps=100,
        # prior=prior,
        # mh=False,
        # A=A,
        # )
        
        # path = new_xs
        

        # print("OM of path", I(path, time, b))
        # print("OM of refined path", I(refined_path, time, uref))

        plot_path(path, (time/hyperparams['dt'])/10, potential, label=f"s: {new_s}", i=i)
        # plt.plot(path,(time/hyperparams['dt'])/10, label=f"s: {new_s}")
        # plt.plot(refined_path,(time/hyperparams['dt'])/10, label=f'refined, s:{new_s}')
        # plt.savefig('potential_new.png')
        plt.legend()

    # return new_b, A - ds*expectation_of_J
    return dbds, A - ds*expectation_of_J

def update_non_amortized(V, b, J, prior, dbds, hyperparams, key, schedule, i, A, rho = lambda key: jnp.zeros((1,))-1., refine=False, ndims=1):
    """
    b: drift term. A function from R^ndims x R -> R^ndims
    hyperparams: dictionary of hyperparameters
    key: random key for the model
    i: index for the current iteration (just for labelling plots)
    Returns:
        new_b: the updated drift term, which is a function from R^ndims x R -> R^ndims
    """

    

    new_s = schedule[i]
    old_s = schedule[i-1] if i>0 else 0.0
    ds = new_s - old_s

    # b = make_b(schedule[:i], b, dbds)

    path_key, model_key, refine_key = jax.random.split(key, 3)

    W = lambda _, key: jnp.sqrt(2)*jax.random.normal(key, shape=(ndims,))
    

    # xs : [batch_size, num_steps, ndims]
    xs, times = jax.pmap(lambda key:sample_sde(
    b=b, 
    W = W,
    rho = rho,
    key=key, 
    dt=hyperparams['dt'], 
    num_steps=hyperparams['num_steps']))(jax.random.split(path_key, hyperparams['batch_size']))

    time = np.arange(0,hyperparams['num_steps'])*hyperparams['dt']
    
    print("old s", old_s)

    # path refinement
    if refine:
        # new_xs = jax.pmap(lambda key, p: refine_path(
        # x=p,
        # s=old_s,
        # b=b,
        # J=J,
        # I=I,
        # time=time,
        # rng_key=key,
        # num_steps=1000
        # ))(jax.random.split(refine_key, hyperparams['batch_size']), xs)[:,:, None]
        # xs = new_xs
        # for path in xs:
        #     plt.plot(path,(time/hyperparams['dt'])/10, label='old')

        # jax.debug.print("shapes 1 {x}", x=xs.shape)

        # plt.plot(xs[0],(time/hyperparams['dt'])/10, label=f'old{i}')
        new_xs, _ = jax.pmap(lambda key, p: refine_spde(
        xts=p,
        V=V,
        s=old_s,
        ds=0.001,
        hyperparams=hyperparams,
        key=key,
        num_steps=100,
        prior=prior,
        mh=False,
        A=A,
        ))(jax.random.split(refine_key, hyperparams['batch_size']), xs)
        xs = new_xs

        # jax.debug.print("shapes 2 {x}", x=xs.shape)

        # for path in xs:
        #     plt.plot(path,(time/hyperparams['dt'])/10, label='old')
        # plt.plot(xs[0],(time/hyperparams['dt'])/10, label=f'new{i}')

    # print(E_J(xs, None), "ej x")
    # print(E_J(new_xs, None), "ej new x")
    # print(times)


    # plt.plot(xs[0],(time/hyperparams['dt'])/10, label='old')
    # for path in new_xs:
    #     plt.plot(path,(time/hyperparams['dt'])/5, label=f'annealed')
    # plt.plot(new_xs[0],(time/hyperparams['dt'])/10, label=f'annealed{i}')

    # xs = jnp.concatenate((xs[:10, :, :], new_xs[:30, :, :]), axis=0)

    expectation_of_J = E_J(J, xs, None)

    dbds = find_dbds(
        dbds=dbds,
        J=J,
        s=new_s,
        b=b,
        # xs[:90, :, :],
        # times[:90, :],
        xs=xs,
        times=times,
        ys=None,
        num_training_steps=hyperparams['num_training_steps']
        )
    
    ### calculate test loss
    test_xs, test_times = jax.pmap(lambda key:sample_sde(
        b=b, 
        W = W,
        rho = rho,
        key=key, 
        dt=hyperparams['dt'], 
        num_steps=hyperparams['num_steps']))(jax.random.split(jax.random.key(500), hyperparams['batch_size']))

    print(f"Test loss is {make_h_loss(expectation_of_J=expectation_of_J, J=J, b=b, s=new_s)(dbds, test_xs, test_times, None)}")

    # new version
    # new+b = b + [dbds(x,t,s)*delta_s for s in ...]

    # print(ds, "ds")
    

    new_b =  lambda x, t: (b(x,t) + dbds(x,t, 0.0)*ds)

    plot = True
    if plot:
        
        
        # new_b = make_b(schedule[:i+1], b, dbds)

        
        # y = (x**4 - 8 * x**2)
        # y = make_double_well_potential(v=5.0)(x)
        # y2 = jax.vmap(lambda k, t: new_b(k, t))(x, times[0])
        # plt.ylim(0,15)
        # if i==0: plt.plot(x, y)
        # plt.plot(x, y2/50)
        # print(y2)

        path, time = sample_sde(
            b=new_b, 
            W = W,
            rho = rho,
            key=key, 
            dt=hyperparams['dt'], 
            num_steps=hyperparams['num_steps'])
        
        

        # print("OM of path", I(path, time, b))
        # print("OM of refined path", I(refined_path, time, uref))

        # plt.plot(path,(time/hyperparams['dt'])/10, label=f"s: {new_s}")
        if refine:

            # refined_path = refine_path(
            #     x=path,
            #     s=new_s,
            #     J=J,
            #     b=b,
            #     I=I,
            #     time=time,
            #     rng_key=jax.random.key(2),
            #     num_steps=10000
            #     )
            
            refined_path, _ = refine_spde(
                xts=path,
                V=V,
                s=new_s,
                ds=0.001,
                hyperparams=hyperparams,
                key=key,
                num_steps=100,
                A=A,
                prior=prior,
                mh=False,
                )
            # plt.plot(refined_path,(time/hyperparams['dt'])/10, label=f'refined, s:{new_s}')
            plot_path(refined_path, (time/hyperparams['dt'])/2.5, make_double_well_potential(v=5.0), label=f"s: {new_s}(post spde)", i=i)

        plot_path(path, (time/hyperparams['dt'])/2.5, make_double_well_potential(v=5.0), label=f"s: {new_s}", i=i)
        # plt.savefig('potential_new.png')
        plt.legend()

    # return new_b, A - ds*expectation_of_J
    return new_b, A - ds*expectation_of_J
