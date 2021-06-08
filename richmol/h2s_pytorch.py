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
import flax
from flax import linen as nn
from flax import optim
import torch
config.update("jax_enable_x64", True)
import scipy
import random as py_random


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
    v_grid = torch.from_numpy(poten(points))
    g_grid = torch.from_numpy(np.array([Gmat(p) for p in points]))
    u_grid = torch.from_numpy(np.array([pseudo(p) for p in points]))

    # primitive basis functions
    psi, dpsi = bas(vmax, points[:, icoo], np.array(ref[icoo]), scale[icoo])
    psi, dpsi = torch.from_numpy(psi), torch.from_numpy(dpsi)

    # matrix elements of operators
    v = opt_einsum.contract("kg,lg,g,g->kl", torch.conj(psi), psi, v_grid, torch.from_numpy(weights))
    u = opt_einsum.contract("kg,lg,g,g->kl", torch.conj(psi), psi, u_grid, torch.from_numpy(weights))
    g = opt_einsum.contract("kg,lg,g,g->kl", torch.conj(dpsi), dpsi, g_grid[:, icoo, icoo], torch.from_numpy(weights))

    # Hamiltonian eigenvalues and eigenvectors
    h = v + 0.5*g + u
    e, vec = torch.linalg.eigh(h)

    print(f"\n1D solutions for coordinate {icoo}")
    print("zero-energy:", e[0])
    print(e-e[0])

    psi *= np.sqrt(weights)
    dpsi *= np.sqrt(weights)
    psi = torch.matmul(psi.T, vec)
    dpsi = torch.matmul(dpsi.T, vec)

    return e, psi, dpsi, points, weights, scale[icoo]


def manzhos1d(icoo, npt, quad, vmax, neigvals, bas, ref, eref):
    # 1d quadrature
    n = [npt if i == icoo else 1 for i in range(len(ref))]
    quads = [quad(n[i], i, ref) for i in range(len(ref))]
    points, weights, scale = quad_prod(quads)
    weights = weights[:, icoo]

    # operators on quadrature grid
    v_grid = torch.from_numpy(poten(points))
    g_grid = torch.from_numpy(np.array([Gmat(p) for p in points]))
    u_grid = torch.from_numpy(np.array([pseudo(p) for p in points]))

    # primitive basis functions (without weights!)
    psi, dpsi = bas(vmax, points[:, icoo], np.array(ref[icoo]), scale[icoo])
    psi, dpsi = torch.from_numpy(psi), torch.from_numpy(dpsi)

    # multiply functions by weights
    psi *= np.sqrt(weights)
    # dpsi *= np.sqrt(weights)
    psi = psi.T
    # dpsi = dpsi.T

    # initial params for NN: Gaussian centers
    peaks = [p for i in range(vmax) for p in scipy.signal.find_peaks(abs(psi[:, i]))[0]]
    print("Number of gaussians:", len(peaks))

    # Pytorch exponential NN
    class ExpNN(torch.nn.Module):
        def __init__(self, peaks, in_sz=1, out_sz=5, ):
            super().__init__()
            self.in_sz = in_sz
            self.ngaussians = len(peaks)
            self.out_sz = out_sz
            # self.peaks = peaks

            self.centers = torch.nn.Parameter(torch.Tensor(points[peaks, icoo]), requires_grad=True)
            self.betas = torch.nn.Parameter(torch.ones(self.ngaussians)*scale[icoo], requires_grad=True)
            self.Lin = torch.nn.Linear(self.ngaussians, self.out_sz, bias=False)

        def forward(self, x):
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            x = torch.from_numpy(x) if type(x) is np.ndarray else x
            x = x.type(torch.FloatTensor)
            y = torch.exp(-self.betas**2 * (x - self.centers) ** 2)
            y = self.Lin(y)
            return y

        def derivative(self, x):
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            x = torch.from_numpy(x) if type(x) is np.ndarray else x
            x = x.type(torch.FloatTensor)
            dh = 0.001
            f = self.forward(x)
            df = (self.forward(x + dh) - f) / dh
            df = df[:, :, None]
            return df

    def sum_of_enr(NN, x):
        res = NN(x)
        dres = NN.derivative(x)
        res, dres = res.type(torch.DoubleTensor), dres.type(torch.DoubleTensor)

        s = opt_einsum.contract("gk,gl->kl", torch.conj(res), res)
        d, v = torch.linalg.eigh(s)
        d = torch.diag(1.0 / torch.sqrt(d))
        sqrt_inv = torch.matmul(v, torch.matmul(d, v.T))

        v = opt_einsum.contract("gk,gl,g->kl", torch.conj(res), res, v_grid)
        u = opt_einsum.contract("gk,gl,g->kl", torch.conj(res), res, u_grid)
        g = opt_einsum.contract("gki,gli,g->kl", torch.conj(dres), dres, g_grid[:, icoo, icoo])

        h = v + 0.5*g + u

        h = torch.matmul(sqrt_inv, torch.matmul(h, sqrt_inv.T))
        e, v = torch.linalg.eigh(h)
        return e[:neigvals].sum(), e, v

    # initialization
    epochs = 1000

    NN = ExpNN(peaks=peaks, out_sz=vmax)
    x = points[:, icoo : icoo + 1]
    x = torch.from_numpy(x.reshape(-1, 1))

    plot_init = False
    if plot_init:
        plt.ion()
        pred = NN(x)
        pred = pred.detach().numpy()
        plt.plot(x, pred)
        plt.draw()
        plt.pause(0.01)

    optimizer = torch.optim.Adam(NN.parameters(), lr=0.1)
    err = torch.zeros((epochs,  neigvals))

    # training loop
    for i in range(epochs):
        optimizer.zero_grad()
        loss, eigvals, eigvecs = sum_of_enr(NN, x)
        loss.backward()
        optimizer.step()

        print("Epoch:", i, "Error:", loss - sum(eref[:neigvals])) # for sum_of_enr loss function
        # print(loss_val, loss_val - -sum(jnp.exp(-eref[:vmax_train]/temp))) # for trace_of_exp loss function

        err[i, :] = eigvals[:neigvals] - eref[:neigvals]

    err = err.detach().numpy()
    eref = eref.detach().numpy()
    plt.ioff()
    plt.clf()
    plt.yscale("log")
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    lower_ind = 0
    for i in range(lower_ind, err.shape[1]):
        plt.plot(np.arange(err.shape[0]), np.abs(err[:,i]), label=f"{round(eref[i],2)}")
        # plt.plot(np.arange(err.shape[0]), err[:,i], label=f"{round(eref[i],2)}")
    plt.legend(loc="upper right")
    plt.title(f'Error w.r.t true Energies, basis size = {vmax}, eigvals considered = {neigvals} ')
    plt.show()

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import scipy

    global masses
    masses = np.array([31.97207070, 1.00782505, 1.00782505]) # atomic masses of S and H

    # equilibrium (reference) values of internal coordinates
    ref = jnp.array([1.3359007, 1.3359007, 92.265883/180.0*jnp.pi])

    icoo = 0
    npt = 200
    quad = quad_her1d
    bas = her1d

    # reference energy values
    vmax = 105
    e, psi, dpsi, points, weights, scale = sol1d(icoo, npt, quad, vmax, bas, ref)

    plot_sol1d = False
    if plot_sol1d:
        psi, dpsi = psi.detach().numpy(), dpsi.detach().numpy()
        plt.plot(points[:, icoo], psi[:, :3], color='red')
        #plt.plot(points, dpsi[:, :3], color='green')
        plt.show()
        plt.title('First Solutions (red) and derivatives (green)')

    # variational NN

    vmax = 10
    neigvals = 5
    manzhos1d(icoo, npt, quad, vmax, neigvals, bas, ref, e)
