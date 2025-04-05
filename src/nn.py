import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Define the MLP using Equinox.
class MLP(eqx.Module):
    layers: list

    def __init__(self, dims, key):
        # dims: list of layer sizes, e.g. [1, 128, 64, 1]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x, t):

        x = jnp.concatenate([x, jnp.ones_like(x) * t], axis=-1)
        
        # Apply each layer; use ReLU activations for hidden layers.
        for layer in self.layers[:-1]:
            x = (layer)(x)
            x = jax.nn.relu(x)
        # Final layer (no activation).
        x = (self.layers[-1])(x)
        return x

# Define the mean squared error loss function.
def mse_loss(model, x, t, y):
    preds = model(x, t)
    return jnp.mean((preds - y) ** 2)

# Define a single training step.
def make_train_step(optimizer, loss):

    @jax.jit
    def train_step(model, opt_state, x, t, y):
        # Compute the loss and gradients.
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, t, y)
        # Use the optimizer to compute parameter updates.
        updates, opt_state = optimizer.update(grads, opt_state)
        # Apply updates to the model.
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    return train_step

def train(loss,model, optimizer, num_training_steps, x, t, y):

    train_step = make_train_step(optimizer, loss)
    opt_state = optimizer.init(model)

    for _ in range(num_training_steps):

        model, opt_state, loss_value = train_step(model, opt_state, x, t, y)
    
    jax.debug.print("loss_value {x}",x=loss_value)
    return model



def main():
    # Set up PRNG keys.
    key = jax.random.key(43)
    model_key, data_key = jax.random.split(key)

    # Define network architecture: input dim=1, two hidden layers, output dim=1.
    dims = [2, 128, 64, 1]
    model = MLP(dims, key=model_key)

    # Set up the optimizer (Adam in this case).
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    

    # Training parameters.
    num_steps = 1000

    # Generate training data.
    xs = jax.random.uniform(data_key, (10, 1), minval=-1.0, maxval=1.0)
    ys = xs ** 2


    model = train(mse_loss, model, optimizer, num_steps, x=xs, t=1.0, y=ys)

    # Test the trained model on some inputs.
    test_x = jnp.linspace(-1, 1, 10).reshape(-1, 1)
    preds = model(test_x, 1.0)
    print("Test inputs:", test_x.flatten())
    print("Model predictions:", preds.flatten())
    print("Loss", mse_loss(model, test_x, 1.0, test_x ** 2))

if __name__ == "__main__":
    main()
