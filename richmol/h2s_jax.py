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
import flax
from flax import linen as nn
from flax import optim
from typing import Sequence
config.update("jax_enable_x64", True)
import scipy


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
    """1D Gauss-Hermite quadrature"""
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
    """Direct product of 1D quadratures"""
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

    psi *= np.sqrt(weights)
    dpsi *= np.sqrt(weights)
    psi = np.dot(psi.T, vec)
    dpsi = np.dot(dpsi.T, vec)

    return e, psi, dpsi, points, weights, scale[icoo]


############################ NN shit ############################ 


def manzhos1d(icoo, npt, quad, vmax, bas, ref, eref):

    # Obtain reference solutions

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
    # v = opt_einsum.contract("kg,lg,g,g->kl", np.conj(psi), psi, v_grid, weights)
    # u = opt_einsum.contract("kg,lg,g,g->kl", np.conj(psi), psi, u_grid, weights)
    # g = opt_einsum.contract("kg,lg,g,g->kl", np.conj(dpsi), dpsi, g_grid[:, icoo, icoo], weights)

    # # Hamiltonian eigenvalues and eigenvectors
    # h = v + 0.5*g + u
    # e, vec = np.linalg.eigh(h)

    # print(f"\n1D solutions for coordinate {icoo}")
    # print("zero-energy:", e[0])
    # print(e-e[0])

    # psi = np.dot(vec.T, psi) * np.sqrt(weights)
    # dpsi = np.dot(vec.T, dpsi) * np.sqrt(weights)

    psi *= np.sqrt(weights)
    dpsi *= np.sqrt(weights)
    psi = psi.T
    dpsi = dpsi.T

    peaks = []
    for i in range(vmax):
        peaks_, _ = scipy.signal.find_peaks(abs(psi[:,i]))
        peaks += [p for p in peaks_]
        # plt.plot(points[:,icoo], psi[:,i])
        for peak in peaks:
            x0 = points[peak,icoo]
            amp = psi[peak,i]
            plt.plot(points[:,icoo], [amp*np.exp(-(r-x0)**2*scale[icoo]**2) for r in points[:,icoo]])
    plt.show()
    # sys.exit()
    print(points[peaks,icoo])
    # print(psi[peaks, i])
    # sys.exit()

    # Train NN

    class ManzhosSingleLayer(nn.Module):
        nhid: int
        nout : int
        @nn.compact
        def __call__(self, inp):
            x = inp
            w = self.param("w", lambda key, shape: jnp.array([points[peaks, icoo]]), (x.shape[-1], self.nhid))
            b = self.param("b", lambda key, shape: jnp.array([scale[icoo] for i in range(self.nhid)]), (self.nhid,))
            y = jnp.exp(-b**2 * (x - w)**2)
            y = nn.Dense(self.nout, name="c", use_bias=False)(y)
            if inp.ndim == 1:
                return y[0, :]
            else:
                return y

    vmax_train = vmax
    model = ManzhosSingleLayer(nhid=len(peaks), nout=vmax_train)
    x = points[:, icoo : icoo + 1]# - ref[icoo]
    y = psi[:, :vmax_train]
    params = model.init(jax.random.PRNGKey(0), x)

    def batch_loss(params, x_batch, y_batch):
        def mean(params):
            def sqerr(x, y):
                pred = model.apply(params, x)
                return jnp.inner(y - pred, y - pred) / 2.0
            return jnp.mean(jax.vmap(sqerr, in_axes=(0, 0))(x_batch, y_batch), axis=0)
        return jax.jit(mean)

    loss = batch_loss(params, x, y)

    optimizer_def = optim.Adam(learning_rate=0.01)
    optimizer = optimizer_def.create(params)
    loss_grad_fn = jax.value_and_grad(loss)
    for i in range(0):
        loss_val, grad = loss_grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        print('Loss step {}: '.format(i), loss_val)

    params = optimizer.target
    model_psi = model.apply(params, x)

    # # plotting
    # for i in range(vmax_train):
    #     plt.scatter(x, y[:,i])
    #     plt.plot(x, model_psi[:,i])
    # plt.show()

    def jac_model(params, x_batch):
        def jac_(params):
            def jac(x):
                return jax.jacrev(model.apply, 1)(params, x)
            return jax.vmap(jac, in_axes=0)(x_batch)
        return jax.jit(jac_)(params)

    def vme(params, x):
        res = model.apply(params, x)
        return opt_einsum.contract("gk,gl,g->kl", jnp.conj(res), res, v_grid)

    def ume(params, x):
        res = model.apply(params, x)
        return opt_einsum.contract("gk,gl,g->kl", jnp.conj(res), res, u_grid)

    def gme(params, x):
        res = jac_model(params, x)
        return opt_einsum.contract("gki,gli,g->kl", jnp.conj(res), res, g_grid[:, icoo, icoo])

    def sme(params, x):
        res = model.apply(params, x)
        return opt_einsum.contract("gk,gl->kl", jnp.conj(res), res)

    def hme(params, x):
        return vme(params, x) + 0.5*gme(params, x) + ume(params, x)


    def enr(params, x):
        def enr_(params):
            h = hme(params, x)
            s = sme(params, x)
            return jnp.sum(jnp.diag(jnp.dot(np.conj(v.T), jnp.dot(sinv, jnp.dot((h - jnp.dot(s, jnp.diag(e))), v)))))
        return jax.jit(enr_)(params)


    def enr2(params, x):
        def enr_(params):
            s = sme(params, x)

            d, v = jnp.linalg.eigh(s)
            d = jnp.diag(1.0 / jnp.sqrt(d))
            sqrt_inv = jnp.dot(v, jnp.dot(d, v.T))

            h = hme(params, x)
            h = jnp.dot(sqrt_inv, jnp.dot(h, sqrt_inv.T))
            e, v = jnp.linalg.eigh(h)
            return jnp.sum(e)
        return jax.jit(enr_)

    conv = []

    optimizer_def = optim.Adam(learning_rate=0.1)
    optimizer = optimizer_def.create(params)
    loss = enr2(params, x)
    loss_grad_fn = jax.value_and_grad(loss)
    for i in range(1000):
        h = hme(params, x)
        s = sme(params, x)
        e, v = scipy.linalg.eigh(h, s)
        # sinv = np.linalg.inv(s)

        loss_val, grad = loss_grad_fn(optimizer.target)
        optimizer = optimizer.apply_gradient(grad)
        print(e[0], e-eref[:vmax_train], np.sum(e))
        params = optimizer.target
        conv.append(e-eref[:vmax_train])

    conv = np.array(conv)
    plt.yscale("log")
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    for i in range(conv.shape[1]):
        plt.plot(np.arange(conv.shape[0]), np.abs(conv[:,i]), label=f"{round(eref[i],2)}")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy import interpolate
    import scipy

    global masses
    masses = np.array([31.97207070, 1.00782505, 1.00782505]) # atomic masses of S and H

    # equilibrium (reference) values of internal coordinates
    ref = jnp.array([1.3359007, 1.3359007, 92.265883/180.0*jnp.pi])

    icoo = 0
    npt = 200
    quad = quad_her1d
    bas = her1d

    vmax = 60
    e, psi, dpsi, points, weights, scale = sol1d(icoo, npt, quad, vmax, bas, ref)

    vmax = 10
    manzhos1d(icoo, npt, quad, vmax, bas, ref, e)

