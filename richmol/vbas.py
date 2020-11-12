import numpy as np
from mol_xy2 import XY2_ralpha
import poten_h2s_Tyuterev
from numpy.polynomial.legendre import leggauss, legval, legder


class PrimBas:
    def __init__(self):
        pass

    def check_outliers(self, x, w, r, ranges, zero_weight_thresh):
        """Removes quadrature points with a low weight (w < zero_weight_thresh)
        that fall outside of the coordinate ranges
        """
        good_points_out = (r<ranges[0]) & (r>ranges[1]) & (w>zero_weight_thresh)
        print(r, ranges)
        if len(w[good_points_out])>0:
            raise ValueError(f"Some quadrature points with significant weight fall " \
                +f"outside of the coordinate ranges = {ranges} \n (r,w) = " \
                +f"{[(rr,ww) for rr,ww in zip(r[good_points_out], w[good_points_out])]}") from None

        good_points_in = (r>=ranges[0]) & (r<=ranges[1])
        return x[good_points_in], w[good_points_in], r[good_points_in]


class LegCos(PrimBas):
    """ One-dimensional basis of Legendre(cos) orthogonal polynomials """

    def __init__(self, molec, ref_coords, icoord, no_points, vmax, ranges, verbose=False, zero_weight_thresh=1e-16):
        """Generates basis of one-dimensional Legendre(cos) orthogonal polynomials

        Args:
            molec (Molecule): Information about the molecule, KEO and PES.
            ref_coords (array (no_coords)): Reference values of all internal coordinates.
            icoord (int): Index of internal coordinate for which the basis is generated.
            no_points (int): Desired number of quadrature points.
            vmax (int): Maximal vibrational quantum number in the basis.
            ranges (list [min, max]): Variation ranges for internal coordinate 'icoord',
                quadrature points that fall outside this ranges, and have corresponding
                weight smaller than a threshold zero_weight_thresh, will be neglected.
            verbose (bool): Set to 'True' if you want to print out some intermediate data.
            zero_weight_thresh (tol): Threshold for the quadrature weight below which
                the corresponding quadrature point can be neglected.
        """
        # quadratures
        x, w = leggauss(no_points)

        # quadrature abscissas -> coordinate values
        r = np.arccos(x)

        # delete points that fall outside the coordinate ranges
        x, w, r = self.check_outliers(x, w, r, ranges, zero_weight_thresh)
    
        # potential and keo on quadrature grid

        coords = np.array(np.broadcast_to(ref_coords, (len(r),len(ref_coords))))
        coords[:,icoord] = r[:]
        gmat = molec.G(coords)[:,icoord,icoord]
        poten = molec.V(coords)

        # print values on quadrature grid
        if verbose==True:
            print("Gauss-Legendre(cos) quadrature\n" + "%23s"%"x" + "%18s"%"w" \
                + "%18s"%"r" + "%18s"%"keo" + "%18s"%"poten")
            for i in range(len(r)):
                print(" %4i"%i + "  %16.8f"%x[i] + "  %16.8e"%w[i] + "  %16.8f"%r[i] \
                    + "  %16.8f"%gmat[i] + "  %16.8f"%poten[i])

        # basis functions and derivatives

        psi = np.zeros((len(r),vmax+1), dtype=np.float64) # basis functions
        dpsi = np.zeros((len(r),vmax+1), dtype=np.float64) # derivatives of basis functions
        jacob = np.ones(len(r), dtype=np.float64) # Jacobian due to changing a variable from x to r

        coefs = np.zeros(vmax+1, dtype=np.float64)
        sqsqpi = np.sqrt(np.sqrt(np.pi))

        for v in range(vmax+1):
            coefs[:] = 0
            coefs[v] = 1.0
            pol = legval(x, coefs)
            dpol = legval(x, legder(coefs, m=1))
            scale_fac = np.sqrt((2.0*v+1)*0.5)
            psi[:,v] = pol * scale_fac
            dpsi[:,v] = -np.sin(r) * dpol * scale_fac

        # Jacobian because we change variable from r=[0,pi] to x=[-1,1],
        #   which gives dr -> dx * 1/sin(r)
        jacob = 1.0/np.sin(r)


if __name__=="__main__":

    # H2S, using valence-bond coordinates and Tuyterev potential
    h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    # test KEO and potential
    G = h2s.G(np.array([ref_coords]))
    V = h2s.V(np.array([ref_coords]))

    angBas = LegCos(h2s, ref_coords, 2, 100, 20, [0, np.pi/2], verbose=False)