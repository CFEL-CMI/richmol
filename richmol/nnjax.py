import jax
from jax import numpy as jnp
from jax import random, jit


def init_params(layer_sizes):
    no_layers = len(layer_sizes)
    rand_keys = random.split(random.PRNGKey(0), no_layers)
    params = []
    for nin, nout, key in zip(layer_sizes[:-1], layer_sizes[1:], rand_keys):
        weight_key, bias_key = random.split(key)
        weights = random.normal(weight_key, (nout, nin))
        biases = random.normal(bias_key, (nout,))
        params.append([weights, biases])
    return params


def nn(params, inp, activfunc = lambda x: (jnp.exp(x)-jnp.exp(-x))/(jnp.exp(x)+jnp.exp(-x))):
    weights, _ = params[0]
    assert (len(inp) == weights.shape[1]), f"number of input activations '{len(inp)}' " + \
        f"is not equal to the number of input layers '{weights.shape[1]}'"
    activations = inp
    for (weights, biases) in params[:-1]:
        out = jnp.dot(weights, activations) + biases
        activations = activfunc(out)
    weights, biases = params[-1]
    out = jnp.dot(weights, activations) + biases
    return out

batch_nn = jax.vmap(nn, in_axes=(None, 1))


def loss_rms(params, x, y):
    out = batch_nn(params, inp)
    ndata = out.shape[0]
    return jnp.sum( jnp.sqrt(jnp.sum((y - out)**2, axis=0)) / ndata )


@jit
def update_descent(params, x, y):
    grads = grad(loss)(params, x, y)


if __name__ == "__main__":
    par = init_params([1,20,30,30,4])
    inp = random.normal(random.PRNGKey(0), (1, 100))
    out = batch_nn(par, inp)
    print(out.shape)
