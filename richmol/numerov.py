import sympy as sp
import numpy as np


class Numerov1D():
    """ Numerov-Cooley 1D basis """

    def __init__(self, molec, ref_coord, icoord, npoints, vmax, ranges, \
                 verbose=False, npoints_stencil=5):
        """
        """
        # potential and keo on quadrature grid

        self.r, step = np.linspace(*ranges, npoints, endpoint=True, retstep=True)
        coords = np.array(np.broadcast_to(ref_coords, (len(self.r), len(ref_coords))))
        coords[:, icoord] = self.r[:]
        gmat = molec.G(coords)[:, icoord, icoord]
        poten = molec.V(coords)

        # print values of operators of grid
        if verbose==True:
            print("Operators on 1D grid\n" + "%18s"%"r" + "%18s"%"keo" + "%18s"%"poten")
            for i in range(len(self.r)):
                print(" %4i"%i + "  %16.8f"%self.r[i] + "  %16.8f"%gmat[i] + "  %16.8f"%poten[i])


        stencil, fdf = self.fdf_stencil(npoints_stencil)
        keo1 = 0
        keo2 = 0
        for i,n in enumerate(stencil):
            keo1 += fdf(step)[1,i] * np.diag([1 for i in range(npoints-abs(n))], k=n)
            keo2 += fdf(step)[2,i] * np.diag([1 for i in range(npoints-abs(n))], k=n)

        hmat = np.diag(poten) - 0.5 * keo2 * gmat
        eigval, eigvec = np.linalg.eigh(hmat)
        print(eigval[:vmax+1]-eigval[0])



    def fdf_stencil(self, npoints):
        """Generates 'npoint'-stencil finite-difference formulas for derivatives
        up to order = 'npoints'-1.

        Args:
            npoints (int): Number of stencil points.

        Returns:
            points (list): Stencil points, len(points) = npoints.
            fdf (sympy.Matirix): Finite-different formulas,
                fdf[iorder,ipoint] gives the coefficient of the ipoint stencil
                point for the iorder order derivative.
        """
        order = npoints - 1
        points = np.arange(-(npoints-1)/2, (npoints-1)/2 + 1).astype(int)
        coefs = sp.ZeroMatrix(npoints, npoints).as_mutable()

        x, h = sp.symbols('s h')
        f = sp.Function('f')
        taylor = lambda point, order: sum(point**i/sp.factorial(i) * f(x).diff(x, i) for i in range(order+1))

        for ipoint, ih in zip(range(npoints), points):
            expansion = taylor(ih * h, order)
            for iorder in range(order+1):
                term =  f(x).diff(x, iorder)
                coefs[iorder, ipoint] = expansion.coeff(term)

        mat = sp.eye(npoints)
        fdf = (coefs.inv() @ mat).T
        return points, sp.lambdify(h, fdf)


if __name__ == "__main__":

    from mol_xy2 import XY2_ralpha
    import poten_h2s_Tyuterev
    import sys

    # H2S, using valence-bond coordinates and Tyuterev potential
    h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    num = Numerov1D(h2s, ref_coords, 0, 2000, 30, [0.86, 2.6], verbose=True)