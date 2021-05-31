import numpy as np
from numpy.polynomial.hermite import hermgauss, hermval, hermder
from numpy.polynomial.legendre import leggauss, legval, legder
from potentials import h2s_tyuterev
from scipy import constants
import functools
import opt_einsum
import jax
from jax import numpy as jnp
from jax import jacfwd, jacrev, random, grad
from jax.config import config
import math
import sys
config.update("jax_enable_x64", True)


############################ KEO and potential ############################ 


G_to_invcm = constants.value('Planck constant') * constants.value('Avogadro constant') \
           * 1e16 / (4.0 * jnp.pi**2 * constants.value('speed of light in vacuum')) * 1e5


eps = jnp.array([[[ int((i - j) * (j - k) * (k - i) * 0.5)
    for k in range(3) ] for j in range(3) ] for i in range(3) ], dtype=jnp.float64)


def poten(q):
    return h2s_tyuterev(*q.T)


def com(method):
    @functools.wraps(method)
    def wrapper_cm(*args, **kwargs):
        xyz = method(*args, **kwargs)
        com = jnp.dot(masses, xyz) / jnp.sum(masses)
        return xyz - com[jnp.newaxis, :]
    return wrapper_cm


@com
def cartesian(q):
    r1, r2, a = q
    xyz = jnp.array([[0.0, 0.0, 0.0],
                    [ r1 * jnp.sin(a/2), 0.0, r1 * jnp.cos(a/2)],
                    [-r2 * jnp.sin(a/2), 0.0, r2 * jnp.cos(a/2)]],
                    dtype=jnp.float64)
    return xyz


@jax.jit
def gmat(q):
    xyz_g = jacfwd(cartesian)(q)
    # xyz_g = jacrev(cartesian)(q)
    tvib = xyz_g
    xyz = cartesian(q)
    natoms = xyz.shape[0]
    trot = jnp.transpose(jnp.dot(eps, xyz.T), (2,0,1))
    ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for iatom in range(natoms)])
    tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
    masses_sq = np.array([np.sqrt(masses[i]) for i in range(natoms)])
    tvec = tvec * masses_sq[:, jnp.newaxis, jnp.newaxis]
    tvec = jnp.reshape(tvec, (natoms*3, len(q)+6))
    return jnp.dot(tvec.T, tvec)


@jax.jit
def Gmat(q):
    return jnp.linalg.inv(gmat(q)) * G_to_invcm

batch_Gmat = jax.jit(jax.vmap(Gmat, in_axes=0))


@jax.jit
def dGmat(q):
    # G = Gmat(q)
    # dg = jacfwd(gmat)(q)
    # # dg = jacrev(gmat)(q)
    # return -jnp.dot(jnp.transpose(jnp.dot(G, dg), (2,0,1)), G)
    return jacrev(Gmat)(q)


@jax.jit
def DetGmat(q):
    return jnp.linalg.det(Gmat(q))


@jax.jit
def dDetGmat(q):
    return grad(DetGmat)(q)


@jax.jit
def hDetGmat(q):
    return jacfwd(jacrev(DetGmat))(q)


@jax.jit
def pseudo(q):
    nq = len(q)
    G = Gmat(q)[:nq, :nq]
    dG = dGmat(q)[:, :nq, :nq]
    det = DetGmat(q)
    det2 = det * det
    ddet = dDetGmat(q)
    hdet = hDetGmat(q)
    pseudo1 = (jnp.dot(ddet, jnp.dot(G, ddet))) / det2
    pseudo2 = (jnp.sum(jnp.diag(jnp.dot(dG, ddet))) + jnp.sum(G * hdet)) / det
    return (pseudo1 * 5 - pseudo2 * 4) / 32.0

batch_pseudo = jax.jit(jax.vmap(pseudo, in_axes=0))


############################ Quadratures ############################ 


def quad_her1d(npt, ind, ref, h=0.001):
    """One-dimensional Gauss-Hermite quadrature"""
    x, weights = hermgauss(npt)

    G = Gmat(ref)

    # 7-point finite-difference rule for 2-order derivative of potential
    fdn_step = np.array([3*h, 2*h, h, 0.0, -h, -2*h, -3*h], dtype=np.float64)
    fdn_coef = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], dtype=np.float64)
    fdn_denom = 180.0

    # compute scaling
    coords = np.array(np.broadcast_to(ref, (len(fdn_step), len(ref))))
    coords[:, ind] = [ref[ind] + step for step in fdn_step]
    freq = np.dot(poten(coords), fdn_coef) / (fdn_denom * h**2)
    scaling = np.sqrt(np.sqrt( 2.0 * np.abs(freq) / np.abs(G[ind, ind]) ))

    points = x / scaling + ref[ind]
    return points, weights, scaling


def quad_prod(quads, pthr=None, wthr=None):
    """Direct product of quadratures"""
    points = (elem[0] for elem in quads)
    weights = (elem[1] for elem in quads)
    scaling = [elem[2] for elem in quads]
    points = np.array(np.meshgrid(*points)).T.reshape(-1, len(quads))
    weights = np.array(np.meshgrid(*weights)).T.reshape(-1, len(quads))

    # remove points with large potential
    if pthr is not None:
        pot = poten(points)
        pmin = np.min(pot)
        ind = np.where(pot - pmin < pthr)
        points = points[ind]
        weights = weights[ind]

    # remove points with small weight
    if wthr is not None:
        ind = np.where(np.prod(weights, axis=1) > wthr)
        points = points[ind]
        weights = weights[ind]

    return points, weights, scaling


############################ Basis functions ############################ 


def her1d(nmax, r, r0, scaling):
    """Normalized Hermite functions and first derivatives

    f(r) = 1/(pi^(1/4) 2^(n/2) sqrt(n!)) exp(-1/2 x^2) Hn(x)
    df(r)/dr = 1/(pi^(1/4) 2^(n/2) sqrt(n!)) exp(-1/2 x^2) (-Hn(x) x + dHn(x)/dx) * scaling
    where x = (r - r0) * scaling

    NOTE: returns f(r) and df(r) without weight factor exp(-1/2 x^2)
    """
    x = (r - r0) * scaling
    sqsqpi = np.sqrt(np.sqrt(np.pi))
    c = np.diag([1.0 / np.sqrt(2.0**n * math.factorial(n)) / sqsqpi for n in range(nmax+1)])
    f = hermval(x, c)
    df = (hermval(x, hermder(c, m=1)) - f * x) * scaling
    return f, df


############################ classical solutions ############################ 


def sol1d(icoo, npt, quad, vmax, bas, ref):
    """Solves 1D Schrödinger equation for selected coordinate at fixed values of other coordinates"""

    # 1d quadrature
    n = [npt if i == icoo else 1 for i in range(len(ref))]
    quads = [quad(n[i], i, ref) for i in range(len(ref))]
    points, weights, scale = quad_prod(quads)
    weights = weights[:, icoo]

    # operators on quadrature grid
    v_grid = poten(points)
    g_grid = jnp.array([Gmat(p) for p in points])
    u_grid = jnp.array([pseudo(p) for p in points])

    # primitive basis functions
    psi, dpsi = bas(vmax, points[:, icoo], np.array(ref[icoo]), scale[icoo])

    # matrix elements of operators
    v = opt_einsum.contract("kg,lg,g,g->kl", np.conj(psi), psi, v_grid, weights)
    u = opt_einsum.contract("kg,lg,g,g->kl", np.conj(psi), psi, u_grid, weights)
    g = opt_einsum.contract("kg,lg,g,g->kl", np.conj(dpsi), dpsi, g_grid[:, icoo, icoo], weights)

    # Hamiltonian eigenvalues and eigenvectors
    h = v + 0.5*g + u
    e, vec = np.linalg.eigh(h)

    print(f"\n1D solutions for coordinate {icoo}")
    print("zero-energy:", e[0])
    print(e-e[0])

    psi = np.dot(vec.T, psi) * np.sqrt(weights)
    dpsi = np.dot(vec.T, dpsi) * np.sqrt(weights)

    return psi.T, dpsi.T, points


############################ NN shit ############################ 


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


def model(params, inp, activfunc = lambda x: jax.nn.sigmoid(x)):
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

batch_model = jax.vmap(model, in_axes=(None, 0))


def loss(params, x, y):
    out = batch_model(params, x)
    ndata = out.shape[0]
    return jnp.sum( jnp.sqrt(jnp.sum((y - out)**2, axis=0)) / ndata )

mom_vec = None

@jax.jit
def update(par, x, y, eta=0.01, gamma=0.0):
    grads = grad(loss)(par, x, y)
    # mom_vec = [(gamma * vw + eta * dw, gamma * vb + eta * db) for (vw, vb), (dw, db) in zip(mom_vec, grads)]
    return [(w - eta * dw, b - eta * db)
          for (w, b), (dw, db) in zip(par, grads)]
    # return [(w - vw, b - vb) for (w, b), (vw, vb) in zip(par, mom_vec)], mom_vec

@jax.jit
def update_adam(par, x, y, mt, vt, nit, beta1=0.9, beta2=0.999, epsilon=1e-8, eta=0.001):
    grads = grad(loss)(par, x, y)
    mt = [(beta1*w + (1-beta1)*dw, beta1*b + (1-beta1)*db) for (w, b), (dw, db) in zip(mt, grads)]
    vt = [(beta2*w + (1-beta2)*dw**2, beta2*b + (1-beta2)*db**2) for (w, b), (dw, db) in zip(vt, grads)]
    m = [(w/(1-beta1**nit), b/(1-beta1**nit)) for (w, b) in mt]
    v = [(w/(1-beta2**nit), b/(1-beta2**nit)) for (w, b) in vt]
    return [(w - eta*mw/(jnp.sqrt(vw)+epsilon), b - eta*mb/(jnp.sqrt(vb)+epsilon)) for (w, b), (mw, mb), (vw, vb) in zip(par, m, v)], mt, vt, nit+1


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    global masses
    masses = np.array([31.97207070, 1.00782505, 1.00782505]) # atomic masses of S and H

    # equilibrium (reference) values of internal coordinates
    ref = jnp.array([1.3359007, 1.3359007, 92.265883/180.0*jnp.pi])

    # Pretrain on 1D solutions

    for icoo in range(3):

        npt = 200
        quad = quad_her1d
        vmax = 30
        bas = her1d
        psi, dpsi, points = sol1d(icoo, npt, quad, vmax, bas, ref)

        vmax = 5
        par = init_params([1,30,30,30,vmax])
        data = jnp.hstack((psi[:,:vmax], points[:,icoo:icoo+1]-ref[icoo]))

        mt = [(0*w, 0*b) for (w, b) in par]
        vt = [(0*w, 0*b) for (w, b) in par]
        nit = 1

        for i in range(500):
            dat = random.shuffle(random.PRNGKey(0), data, axis=0)
            for batch in jnp.split(dat, 1, axis=0):
                y = batch[:,:vmax]
                x = batch[:,vmax:vmax+1]
                par, mt, vt, _ = update_adam(par, x, y, mt, vt, nit)
            l = loss(par, data[:,vmax:vmax+1], data[:,:vmax])
            nit +=1
            print(i, l)

        # plot functions
        res = batch_model(par, data[:,vmax:vmax+1])
        for i in range(vmax):
            plt.scatter(data[:,vmax+1], data[:,i])
            plt.plot(data[:,vmax+1], res[:,i])
        plt.show()

        sys.exit()
