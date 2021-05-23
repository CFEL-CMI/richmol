import jax.numpy as np
from jax import grad, jacfwd, jacrev, jit
import jax
from jax.config import config
import functools
from scipy import constants
config.update("jax_enable_x64", True)


# converts G-matrix into units of cm^-1 providing masses in atomic units and distances in Angstrom
G_to_invcm = constants.value('Planck constant') * constants.value('Avogadro constant') \
           * 1e+16 / (4.0 * np.pi**2 * constants.value('speed of light in vacuum')) * 1e5

# Levi-Civita symbol
eps = np.array([[[ int((i - j) * (j - k) * (k - i) * 0.5)
                   for k in range(3) ] for j in range(3) ] for i in range(3) ], dtype=np.float64)


def init(mass, cart):
    global masses, natoms, totmass, masses_sq, cartesian
    masses = np.array(mass)
    natoms = len(masses)
    totmass = np.sum(masses)
    cartesian = cart
    masses_sq = np.array([np.sqrt(masses[i]) for i in range(natoms)])


def com(method):
    """Decorator for `cartesian` function, shifts frame origin to the centre of mass"""
    @functools.wraps(method)
    def wrapper_cm(*args, **kwargs):
        xyz = method(*args, **kwargs)
        com = np.dot(masses, xyz) / totmass
        return xyz - com[np.newaxis, :]
    return wrapper_cm


@jit
def gmat(q):
    xyz_g = np.array(jacfwd(cartesian)(q))
    tvib = np.transpose(xyz_g, (1, 2, 0))
    xyz = cartesian(q)
    trot = np.transpose(np.dot(eps, xyz.T), (2,0,1))
    ttra = np.array([np.eye(3, dtype=np.float64) for iatom in range(natoms)])
    tvec = np.concatenate((tvib, trot, ttra), axis=2)
    tvec = tvec * masses_sq[:, np.newaxis, np.newaxis]
    tvec = np.reshape(tvec, (natoms*3, len(q)+6))
    return np.dot(tvec.T, tvec)


@jit
def Gmat(q):
    return np.linalg.inv(gmat(q)) * G_to_invcm


@jit
def dGmat(q):
    G = Gmat(q)
    dg = np.array(jacfwd(gmat)(q))
    return -np.dot(np.dot(dg, G), G.T)


@jit
def pseudo(q):
    def _det(q):
        return np.linalg.det(Gmat(q))
    def _ddet(q):
        return np.array(jacfwd(_det)(q))
    def _dddet(q):
        return np.array(jacfwd(_ddet)(q))
    nq = len(q)
    G = Gmat(q)[:nq, :nq]
    dG = dGmat(q)[:, :nq, :nq]
    det = _det(q)
    det2 = det * det
    ddet = _ddet(q)
    dddet = _dddet(q)
    pseudo1 = (np.dot(ddet, np.dot(G, ddet))) / det2
    pseudo2 = (np.sum(np.diag(np.dot(dG, ddet))) + np.sum(G * dddet)) / det
    return (pseudo1 * 5 - pseudo2 * 4) / 32.0

