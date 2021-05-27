import moljax
import jax.numpy as jnp
from potentials import h2s_tyuterev
from quadratures import herm1d, prodgrid, legendre1d
import numpy as np
import basis
import sys
import torch

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
    ref = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    ncoo = len(ref)

    # 1D solutions for each internal coordinate

    vec1D = []

    for icoo in range(ncoo):
        a = 0
        b = 3
        # 1D quadrature
        quads = [legendre1d(100, i, a, b, ref) if i == icoo else
                 legendre1d(1, i, a, b, ref) for i in range(ncoo)]
        points, weights, scale = prodgrid(quads, ind=[0, 1, 2], ref=ref, poten=h2s_tyuterev)
        weights = torch.from_numpy(weights[:, icoo])

        # operators on quadrature grid
        poten = torch.from_numpy(h2s_tyuterev(*points.T))
        gmat = torch.from_numpy(np.array([moljax.Gmat(list(q)) for q in points]).copy())
        pseudo = torch.from_numpy(np.array([moljax.pseudo(list(q)) for q in points]).copy())

        # primitive basis functions on quadrature grid
        nmax = 30
        psi, dpsi = basis.legendre(nmax, points[:, icoo], a, b, ref[icoo])
        # matrix elements of operators
        v = basis.potme(psi, psi, poten, weights)
        u = basis.potme(psi, psi, pseudo, weights)
        g = basis.vibme(psi, psi, dpsi, dpsi, gmat[:, icoo:icoo+1, icoo:icoo+1], weights)

        # Hamiltonian eigenvalues and eigenvectors
        h = v + 0.5*g + u
        #sys.exit()
        e, vec = torch.symeig(h)
        vec1D.append(vec.T)

        print(f"\n1D solutions for coordinate {icoo}")
        print("zero-energy:", e[0])
        print(e-e[0])
        sys.exit()

    # 3D solutions

    # quadrature
    quads = [herm1d(100, i, ref, moljax.Gmat, h2s_tyuterev) for i in range(len(ref))]
    points, weights, scale = prodgrid(quads, ind=[0,1,2], ref=ref, poten=h2s_tyuterev, wthr=1e-30)
    weights = np.prod(weights[:,0:3], axis=1)
    print(points.shape)

    # operators on quadrature grid
    poten = h2s_tyuterev(*points.T)
    gmat = np.array([moljax.Gmat(list(q)) for q in points])
    pseudo = [moljax.pseudo(list(q)) for q in points]

    # basis set
    nmax = 30
    psi = []
    dpsi = []
    for icoord in [0,1,2]:
        f, df = basis.legendre(nmax, points[:, icoord], a, b, ref[icoord])
        psi.append(torch.dot(vec1D[icoord], f))
        dpsi.append(torch.dot(vec1D[icoord], df))

    nmax = 10
    # matrix elements of operators
    v = basis.potme(psi, psi, poten, weights, nmax=nmax, w=[2,2,1])
    print(v.shape)
    u = basis.potme(psi, psi, pseudo, weights, nmax=nmax, w=[2,2,1])
    g = basis.vibme(psi, psi, dpsi, dpsi, gmat[:,0:3,0:3], weights, nmax=nmax, w=[2,2,1])

    # Hamiltonian eigenvalues and eigenvectors
    h = v + 0.5*g + u
    e, vec = torch.linalg.eigh(h)

    print(f"\n3D solutions")
    print("zero-energy:", e[0])
    print(e-e[0])
