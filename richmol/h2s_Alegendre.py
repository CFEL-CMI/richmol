import moljax
import jax.numpy as jnp
from potentials import h2s_tyuterev
from quadratures import herm1d, prodgrid, legendre1d
import numpy as np
import basis
import sys
import torch
import time

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
        e, vec = torch.symeig(h, eigenvectors=True)
        vec1D.append(vec.T)
        print(f"\n1D solutions for coordinate {icoo}")
        print("zero-energy:", e[0])
        print(e-e[0])


    # 3D solutions

    # quadrature
    NF = basis.InvResnet()
    quads = [legendre1d(40, i, None, b, ref) for i in range(len(ref))]
    points, weights, scale = prodgrid(quads, ind=[0,1,2], ref=ref, poten=h2s_tyuterev, wthr=1e-30)
    print(points.shape)
    weights = np.prod(weights[:,0:3], axis=1)
    weights = torch.from_numpy(weights)
    # Apply transformation on the points
    y = NF(torch.from_numpy(points).float())
    scale = (b-a)/2
    points_t = y/scale + torch.Tensor(ref) # points transformed
    # operators on quadrature grid
    poten = torch.from_numpy(h2s_tyuterev(*(points_t.detach().numpy()).T))
    gmat = torch.from_numpy(np.array([moljax.Gmat(list(q)) for q in points_t.detach().numpy()]).copy())
    pseudo = torch.from_numpy(np.array([moljax.pseudo(list(q)) for q in points_t.detach().numpy()]).copy())
    print("computed all operators on points")
    # Compute the Jacobian of the Inverse
    start = time.time()
    Jac = NF.Inv_Jacobian(points_t)
    end = time.time()
    print(Jac)
    print(Jac.shape)
    print(f"time needed to compute the Jacobian: {end-start}")
    sys.exit()
    # basis set
    nmax = 30
    psi = []
    dpsi = []
    for icoord in [0,1,2]:

        f, df = basis.legendre(nmax, points[:, icoord], a, b, ref[icoord])
        psi.append(torch.matmul(vec1D[icoord], f))
        dpsi.append(torch.matmul(vec1D[icoord], df))

    nmax = 10
    # matrix elements of operators
    v = basis.potme(psi, psi, poten, weights, nmax=nmax, w=[2,2,1])
    print("computed v")
    u = basis.potme(psi, psi, pseudo, weights, nmax=nmax, w=[2,2,1])
    print("computed u")
    g = basis.vibme(psi, psi, dpsi, dpsi, gmat[:,0:3,0:3], weights, nmax=nmax, w=[2,2,1])
    print("computed g")

    # Hamiltonian eigenvalues and eigenvectors
    h = v + 0.5*g + u
    print(h.shape)
    e, vec = torch.symeig(h, eigenvectors=True)

    print(f"\n3D solutions")
    print("zero-energy:", e[0])
    print(e-e[0])
