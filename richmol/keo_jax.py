import jax.numpy as np
from jax import grad, jacfwd, jacrev, jit
import const
import functools
from jax.config import config
config.update("jax_enable_x64", True)


levi_chivita = np.array([[[ int((i - j) * (j - k) * (k - i) * 0.5) \
                            for k in range(3) ] for j in range(3) ] for i in range(3) ], \
                            dtype=np.float64)

G_to_invcm = const.planck * const.avogno * 1.0e+16 / (4.0 * np.pi * np.pi * const.vellgt)
masses=np.array([31.97207070, 1.00782505, 1.00782505] ) 

def init(*args, **kwargs):
    """ Initializes global molecular parameters """
    global masses
    global natoms
    global totmass
    global internal_to_cartesian
    global poten
    if 'masses' in kwargs:
        masses = np.array(kwargs['masses'], dtype=np.float64)
        natoms = len(masses)
        totmass = np.sum(masses)
    if 'internal_to_cartesian' in kwargs:
        internal_to_cartesian = kwargs['internal_to_cartesian']


def com(method):
    """ Shifts frame origin to the centre of mass """
    @functools.wraps(method)
    def wrapper_cm(*args, **kwargs):
        xyz = method(*args, **kwargs)
        com = np.dot(masses, xyz) / totmass
        return xyz - com[np.newaxis, :]
    return wrapper_cm




def bisector(axes='zyx'):
    """ Bisector frame for triatomic molecule """
    axes_ind = [("x", "y", "z").index(s) for s in list(axes.lower())]
    def inner_function(method):
        @functools.wraps(method)
        def wrapper_bisect(*args, **kwargs):
            xyz = method(*args, **kwargs)
            r1 = np.linalg.norm(xyz[1, :] - xyz[0, :])
            r2 = np.linalg.norm(xyz[2, :] - xyz[0, :])
            e1 = (xyz[1, :] - xyz[0, :]) / r1
            e2 = (xyz[2, :] - xyz[0, :]) / r2
            n1 = (e1 + e2)
            l1 = np.linalg.norm(n1)
            n1 = n1 / l1
            n2 = np.cross(e1, n1)
            l2 = np.linalg.norm(n2)
            n2 = n2 / l2
            n3 = np.cross(n1, n2)
            l3 = np.linalg.norm(n3)
            n3 = n3 / l3
            tmat = np.stack((n1, n2, n3))
            return np.dot(xyz, tmat[axes_ind, :].T)
        return wrapper_bisect
    return inner_function



@jit
def gmat(coords):
    """ Computes g-small kinetic energy matrix """
    xyz_grad = np.array(jacfwd(internal_to_cartesian)(coords))
    tvib = np.transpose(xyz_grad, (1, 2, 0)) # shape = (iatom, ialpha, icoord_vib)
    xyz = internal_to_cartesian(coords)
    trot = np.transpose(np.dot(levi_chivita, xyz.T), (2,0,1)) # shape = (iatom, ialpha, icoord_rot)
    ttra = np.array([np.eye(3, dtype=np.float64) for iatom in range(natoms)]) # shape = (iatom, ialpha, icoord_tran)

    tvec = np.concatenate((tvib, trot, ttra), axis=2)
    sqm = np.array([np.sqrt(masses[iatom]) for iatom in range(natoms)])
    tvec = tvec * sqm[:, np.newaxis, np.newaxis]

    tvec = np.reshape(tvec, (natoms*3, natoms*3))
    return np.dot(tvec.T, tvec)


@jit
def Gmat(coords):
    """ Computes G kinetic energy matrix using """
    return np.linalg.inv(gmat(coords)) * G_to_invcm


@jit
def dGmat(coords):
    """ Computes jacobian of G kinetic energy matrix """
    return np.array(jacfwd(Gmat)(coords))


@jit
def pseudo(coords):
    """ Computes pseudopotential """
    def _det(coords):
        return np.linalg.det(Gmat(coords))
    def _ddet(coords):
        return np.array(jacfwd(_det)(coords))
    def _dddet(coords):
        return np.array(jacfwd(_ddet)(coords))
    ncoords = len(coords)
    G = Gmat(coords)[:ncoords, :ncoords]
    dG = dGmat(coords)[:,:ncoords,:ncoords]
    det = _det(coords)
    det2 = det * det
    ddet = _ddet(coords)
    dddet = _dddet(coords)
    pseudo1 = (np.dot(ddet, np.dot(G, ddet))) / det2
    pseudo2 = (np.sum(np.diag(np.dot(dG, ddet))) + np.sum(G * dddet)) / det
    return (pseudo1 * 5 - pseudo2 * 4) / 32.0
