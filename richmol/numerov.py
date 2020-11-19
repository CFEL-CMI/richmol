import sympy as sp
import numpy as np


class Numerov1D():
    """ Numerov-Cooley 1D basis

    Args:

        molec : Molecule
            Information about the molecule, KEO, PES, and pseudopotential.

        ref_coords : array (no_coords)
            Reference values for all internal coordinates.

        icoord : int
            Index of internal coordinate for which the basis is generated.

        no_points : int
            Number of points in the Numerov equidistant grid.

        vmax : int
            Maximal vibrational quantum number in the basis.

        ranges : list [min, max]
            Variation ranges for internal coordinate 'icoord'.

        verbose : bool
            Set to 'True' if you want to print out some intermediate data.

        npoints_stencil : float
            Number of stencil points for finite differences.
    """

    def __init__(self, molec, ref_coord, icoord, npoints, vmax, ranges, \
                 verbose=False, npoints_stencil=5):
        """ Generates basis of one-dimensional Numerov-Cooley functions """
        # coordinate values on grid

        self.r, step = np.linspace(*ranges, npoints, endpoint=True, retstep=True)

        # potential, g-matrix, and pseudopotential on grid

        coords = np.array(np.broadcast_to(ref_coords, (len(self.r), len(ref_coords))))
        coords[:, icoord] = self.r[:]
        gmat = molec.G(coords)[:, icoord, icoord]
        #dgmat = molec.dG(coords)[]
        dgmat = np.zeros(len(self.r))
        poten = molec.V(coords)
        #pseudo = molec.PP(coords)
        pseudo = np.zeros(len(self.r))

        # print values of operators on grid

        if verbose==True:
            print("Operators on 1D grid\n" + "%23s"%"r" + "%18s"%"G"  + "%18s"%"dG" \
                  + "%18s"%"V" + "%18s"%"U")
            for i in range(len(self.r)):
                print(" %4i"%i + "  %16.8f"%self.r[i] + "  %16.8f"%gmat[i] \
                      + "  %16.8f"%dgmat[i] + "  %16.8f"%poten[i] + "  %16.8f"%pseudo[i])

        # finite-difference formulas

        stencil, fdf = self.fdf_stencil(npoints_stencil)

        # d/dr and d^2/dr^2 operators

        dr = 0
        ddr = 0
        for i,n in enumerate(stencil):
            # first derivative
            dr += fdf(step)[1,i] * np.diag([1 for i in range(npoints-abs(n))], k=n)
            # second derivative
            ddr += fdf(step)[2,i] * np.diag([1 for i in range(npoints-abs(n))], k=n)

        # Hamiltonian

        hmat = -0.5 * dgmat * dr  - 0.5 * gmat * ddr + np.diag(poten + pseudo)

        eigval, eigvec = np.linalg.eigh(hmat)

        self.enr = eigval[:vmax+1] - eigval[0]
        self.psi = eigvec[:,vmax+1]


    def fdf_stencil(self, npoints):
        """Generates npoint-stencil finite-difference formulas for derivatives
        up to order of npoints-1.

        Args:

            npoints : int
                Number of stencil points.

        Returns:

            points : list
                Stencil points, len(points) = npoints.

            fdf : lambda function
                Finite-difference coefficients, fdf(h)[iorder, ipoint] returns
                the coefficient of the ipoint stencil point for the iorder
                order derivative, with the step-size given by h.
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

    #strBas = Numerov1D(h2s, ref_coords, 0, 2000, 30, [0.86, 3.0], verbose=False)
    angBas = Numerov1D(h2s, ref_coords, 2, 2000, 30, [50*np.pi/180.0, 150.0*np.pi/180.0], verbose=True)

    # reference Numerov bending energies for H2S from TROVE
    trove_bend_enr = [0.00000000, 1209.51837915, 2413.11694104, 3610.38836754, 4800.89613073, \
                      5984.17417895, 7159.74903030, 8327.19623484, 9486.23461816,10636.84908026, \
                      11779.41405000,12914.77206889,14044.22229672,15169.39699217,16292.01473383, \
                      17413.45867039,18534.10106278,19652.46977498,20765.06867992,21869.61533575, \
                      22975.24449419]

    # reference Numerov stretching energies for H2S from TROVE
    trove_str_enr = [0.00000000, 2631.91316250, 5168.60694305, 7610.21707943, 9956.62949349, \
                     12207.53640553,14362.48285216,16420.90544929,18382.16446603,20245.56943697, \
                     22010.39748652,23675.90513652,25241.42657687,26708.10237789,28092.75163810, \
                     29462.97307758,30918.23065498,32507.23053835,34229.22373259,36072.32685441, \
                     38026.38406259]

    print("richmol-TROVE for bending")
    print(" ".join("  %12.4f"%e1 + "  %12.4f"%e2 + "  %12.4f"%(e1-e2) + "\n" for e1,e2 in zip(angBas.enr, trove_bend_enr)))

    #print("richmol-TROVE for stretching")
    #print(" ".join("  %12.4f"%e1 + "  %12.4f"%e2 + "  %12.4f"%(e1-e2) + "\n" for e1,e2 in zip(strBas.enr, trove_str_enr)))
