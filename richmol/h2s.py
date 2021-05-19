import moljax
import jax.numpy as jnp
from potentials import h2s_tyuterev
from quadratures import gausshermite, gausshermite1d
import numpy as np
import basis
from linetimer import CodeTimer
import sys


@moljax.com
def cartesian(q):
    """Internal to Cartesian coordinate transformation for XY2-type molecule

    Args:
        q : list
            Internal coordinates in the order: X-Y1, X-Y2 bond distances, and Y1-X-Y2 bond angle
    Returns:
        xyz : array (no_atoms, 3)
            Cartesian coordinates of atoms, first index is the atom number (0, 1, 2 are X, Y1, Y2)
            and second index is Cartesian component (0, 1, 2 are x, y, z)
    """
    r1, r2, a = q
    xyz = jnp.array([[0.0, 0.0, 0.0],
                    [ r1 * jnp.sin(a/2), 0.0, r1 * jnp.cos(a/2)],
                    [-r2 * jnp.sin(a/2), 0.0, r2 * jnp.cos(a/2)]],
                    dtype=jnp.float64)
    return xyz


if __name__ == "__main__":

    # init molecule
    mass = [31.97207070, 1.00782505, 1.00782505] # atomic masses of S and H
    moljax.init(mass, cartesian)

    # equilibrium (reference) values of internal coordinates
    qref = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    # points_, weights, scale = gausshermite(50, [1], qref, moljax.Gmat, h2s_tyuterev)
    # points = np.array(np.broadcast_to(qref, (len(points_), len(qref))))
    # points[:,1] = points_[:,0]
    # poten = h2s_tyuterev(*points.T)
    # gmat_ = [moljax.Gmat(list(q)) for q in points]
    # gmat = np.array([g[0] for g in gmat_])
    # dgmat = np.array([g[1] for g in gmat_])
    # pseudo = [moljax.pseudo(list(q)) for q in points]
    # nmax = 20
    # psi, dpsi = basis.hermite(nmax, points[:,1], qref[1], scale[0])
    # v = basis.potme(psi, psi, poten, weights, nmax=nmax, w=[1])
    # u = basis.potme(psi, psi, pseudo, weights, nmax=nmax, w=[1])
    # g = basis.vibme(psi, psi, dpsi, dpsi, gmat[:,1:2,1:2], weights, nmax=nmax, w=[1])
    # o = basis.ovlp(psi, psi, weights, nmax=nmax, w=[1])
    # print(np.max(np.abs(o-np.eye(nmax+1))))

    # h = v + 0.5*g + u
    # e, _ = np.linalg.eigh(h.T)
    # print(e[0])
    # print(e-e[0])
    # # # nprim = [20, 20, 20]
    # # # with CodeTimer("primitive product basis"):
    # # #     psi = [b[0] for b in bas]
    # # #     dpsi = [b[1] for b in bas]
    # sys.exit()
    #=================================== full 3D ===================================
    # grid points
    with CodeTimer("quadrature"):
        points, weights, scale = gausshermite(30, [0,1,2], qref, moljax.Gmat, h2s_tyuterev)
        print("number of points:", len(points))

    # potential on grid
    with CodeTimer("potential"):
        poten = h2s_tyuterev(*points.T)

    # G-matrix on grid
    with CodeTimer("G-matrix"):
        gmat_ = [moljax.Gmat(list(q)) for q in points]
        gmat = np.array([g[0] for g in gmat_])
        dgmat = np.array([g[1] for g in gmat_])

    # pseudo-potential on grid
    with CodeTimer("potential"):
        pseudo = np.array([moljax.pseudo(list(q)) for q in points])

    # primitive 1D basis sets for each internal coordinate
    nprim = [20, 20, 20]
    with CodeTimer("primitive product basis"):
        bas = [basis.hermite(nmax, q, a, b) for nmax, q, a, b in zip(nprim, points.T, qref, scale)]
        psi = [b[0] for b in bas]
        dpsi = [b[1] for b in bas]

    nmax = 10

    with CodeTimer("potential matrix elements"):
        v = basis.potme(psi, psi, poten, weights, nmax=nmax, w=[2,2,1])

    with CodeTimer("pseudo-potential matrix elements"):
        u = basis.potme(psi, psi, pseudo, weights, nmax=nmax, w=[2,2,1])

    with CodeTimer("kinetic matrix elements"):
        g = basis.vibme(psi, psi, dpsi, dpsi, gmat[:,:3,:3], weights, nmax=nmax, w=[2,2,1])

    h = v + 0.5*g + u
    e, _ = np.linalg.eigh(h)
    print(e-e[0])
