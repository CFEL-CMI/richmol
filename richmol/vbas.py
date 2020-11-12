import numpy as np
from mol_xy2 import XY2_ralpha
import poten_h2s_Tyuterev
import poten_h2o_Polyansky
from numpy.polynomial.legendre import leggauss, legval, legder
from numpy.polynomial.hermite import hermgauss, hermval, hermder


singular_tol = 1e-12 # tolerance for considering matrix singular
symmetric_tol = 1e-12 # tolerance for considering matrix symmetric


class PrimBas:
    def __init__(self):
        pass

    def check_outliers(self, x, w, r, ranges, zero_weight_thresh):
        """Removes quadrature points with a low weight (w < zero_weight_thresh)
        that fall outside of the coordinate ranges
        """
        points_out1 = (r<ranges[0]) & (w>zero_weight_thresh)
        points_out2 = (r>ranges[1]) & (w>zero_weight_thresh)
        if len(w[points_out1])>0 or len(w[points_out2])>0:
            maxw = max([np.max(w[points_out1]), np.max(w[points_out2])])
            raise ValueError(f"Some quadrature points with significant weight (max = {maxw}) " \
                +f"fall outside of the coordinate ranges = {ranges}") from None

        points_in = (r>=ranges[0]) & (r<=ranges[1])
        return x[points_in], w[points_in], r[points_in]


class LegCos(PrimBas):
    """ One-dimensional basis of Legendre(cos) orthogonal polynomials """

    def __init__(self, molec, ref_coords, icoord, no_points, vmax, ranges, \
                 verbose=False, zero_weight_thresh=1e-20):
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

        Returns:
            don't know yet
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
                print(" %4i"%i + "  %16.8f"%x[i] + "  %16.8e"%w[i] \
                    + "  %16.8f"%r[i] + "  %16.8f"%gmat[i] + "  %16.8f"%poten[i])

        # basis functions and derivatives

        psi = np.zeros((len(r),vmax+1), dtype=np.float64) # basis functions
        dpsi = np.zeros((len(r),vmax+1), dtype=np.float64) # derivatives of basis functions
        jacob = np.ones(len(r), dtype=np.float64) # Jacobian due to changing a variable from x to r

        coefs = np.zeros(vmax+1, dtype=np.float64)

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

        # overlap matrix
        ovlp = np.zeros((vmax+1, vmax+1), dtype=np.float64)
        for v1 in range(vmax+1):
            for v2 in range(vmax+1):
                ovlp[v1,v2] = np.sum(np.conjugate(psi[:,v1]) * psi[:,v2] * w[:] * jacob[:])

        # check if overlap matrix is symmetric
        if np.allclose(ovlp, ovlp.T, atol=symmetric_tol) == False:
            raise RuntimeError(f"Overlap matrix is not symmetric (tol = {symmetric_tol})")

        # inverse square root of overlap matrix
        val, vec = np.linalg.eigh(ovlp)
        if np.any(np.abs(val) < singular_tol):
            raise RuntimeError(f"Overlap matrix is singular (tol = {singular_tol}") from None
        d = np.diag(1.0/np.sqrt(val))
        sqrt_inv = np.dot(vec, np.dot(d, vec.T))

        # orthonormalize basis functions
        psi = np.dot(psi, sqrt_inv.T)
        dpsi = np.dot(dpsi, sqrt_inv.T)

        # Hamiltonian matrix
        hmat = np.zeros((vmax+1, vmax+1), dtype=np.float64)
        for v1 in range(vmax+1):
            for v2 in range(vmax+1):
                fint = 0.5 * gmat * np.conjugate(dpsi[:,v1]) * dpsi[:,v2] \
                     + poten * np.conjugate(psi[:,v1]) * psi[:,v2]
                hmat[v1,v2] = np.sum(fint * w * jacob)

        # check if Hamiltonian is hermitian
        if np.allclose(hmat, np.conjugate(hmat.T), atol=symmetric_tol) == False:
            raise RuntimeError(f"Hamiltonian matrix is not hermitian (tol = {symmetric_tol})")

        # diagonalize Hamiltonian
        eigval, eigvec = np.linalg.eigh(hmat)

        # transform basis
        psi = np.dot(psi, eigvec.T)
        dpsi = np.dot(dpsi, eigvec.T)
        print(eigval-eigval[0])


class Hermite(PrimBas):
    """ One-dimensional basis of Hermite orthogonal polynomials """

    def __init__(self, molec, ref_coords, icoord, no_points, vmax, ranges, \
                 verbose=False, zero_weight_thresh=1e-20, fdf_h=0.001):
        """Generates basis of one-dimensional Hermite orthogonal polynomials

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
            fdf_h (float): Step size for finite-difference calculation of second-order
                derivative of potential function.

        Returns:
            don't know yet
        """
        # quadratures
        x, w = hermgauss(no_points)

        # quadrature abscissas -> coordinate values

        gmat = molec.G(np.array([ref_coords]))[0,icoord,icoord]
        # use finite-differences (7-point) to compute frequency
        fdf_steps = np.array([3*fdf_h, 2*fdf_h, fdf_h, 0.0, -fdf_h, -2*fdf_h, -3*fdf_h], dtype=np.float64)
        fdf_coefs = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], dtype=np.float64)
        fdf_denom = 180.0
        coords = np.array(np.broadcast_to(ref_coords, (len(fdf_steps),len(ref_coords))))
        coords[:,icoord] = [ref_coords[icoord]+st for st in fdf_steps]
        poten = molec.V(coords)
        freq = np.dot(poten, fdf_coefs)/(fdf_denom*fdf_h*fdf_h)
        # mapping between r and x
        xmap = np.sqrt(np.sqrt( 2.0*np.abs(freq)/np.abs(gmat) ))
        r = x / xmap + ref_coords[icoord]
        if verbose==True:
            print(f"Mapping x <--> r calculated for Gauss-Hermite quadrature: {xmap}, (mu={gmat}, omega={freq})")

        # delete points that fall outside the coordinate ranges
        x, w, r = self.check_outliers(x, w, r, ranges, zero_weight_thresh)

        # potential and keo on quadrature grid

        coords = np.array(np.broadcast_to(ref_coords, (len(r),len(ref_coords))))
        coords[:,icoord] = r[:]
        gmat = molec.G(coords)[:,icoord,icoord]
        poten = molec.V(coords)

        # print values on quadrature grid
        if verbose==True:
            print("Gauss-Hermite quadrature\n" + "%23s"%"x" + "%18s"%"w" \
                + "%18s"%"r" + "%18s"%"keo" + "%18s"%"poten")
            for i in range(len(r)):
                print(" %4i"%i + "  %16.8f"%x[i] + "  %16.8e"%w[i] \
                    + "  %16.8f"%r[i] + "  %16.8f"%gmat[i] + "  %16.8f"%poten[i])

        # basis functions and derivatives

        psi = np.zeros((len(r),vmax+1), dtype=np.float64) # basis functions
        dpsi = np.zeros((len(r),vmax+1), dtype=np.float64) # derivatives of basis functions
        jacob = np.ones(len(r), dtype=np.float64) # Jacobian due to changing a variable from x to r

        coefs = np.zeros(vmax+1, dtype=np.float64)
        sqsqpi = np.sqrt(np.sqrt(np.pi))

        for v in range(vmax+1):
            coefs[:] = 0
            coefs[v] = 1.0
            pol = hermval(x, coefs)
            dpol = hermval(x, hermder(coefs, m=1))
            scale_fac = 1.0/np.sqrt(2.0**v*np.math.factorial(v))/sqsqpi
            psi[:,v] = pol * scale_fac
            dpsi[:,v] = (dpol - pol * x) * scale_fac * xmap

        # overlap matrix
        ovlp = np.zeros((vmax+1, vmax+1), dtype=np.float64)
        for v1 in range(vmax+1):
            for v2 in range(vmax+1):
                ovlp[v1,v2] = np.sum(np.conjugate(psi[:,v1]) * psi[:,v2] * w[:] * jacob[:])

        # check if overlap matrix is symmetric
        if np.allclose(ovlp, ovlp.T, atol=symmetric_tol) == False:
            raise RuntimeError(f"Overlap matrix is not symmetric (tol = {symmetric_tol})")

        # inverse square root of overlap matrix
        val, vec = np.linalg.eigh(ovlp)
        if np.any(np.abs(val) < singular_tol):
            raise RuntimeError(f"Overlap matrix is singular (tol = {singular_tol}") from None
        d = np.diag(1.0/np.sqrt(val))
        sqrt_inv = np.dot(vec, np.dot(d, vec.T))

        # orthonormalize basis functions
        psi = np.dot(psi, sqrt_inv.T)
        dpsi = np.dot(dpsi, sqrt_inv.T)

        # Hamiltonian matrix
        hmat = np.zeros((vmax+1, vmax+1), dtype=np.float64)
        for v1 in range(vmax+1):
            for v2 in range(vmax+1):
                fint = 0.5 * gmat * np.conjugate(dpsi[:,v1]) * dpsi[:,v2] \
                     + poten * np.conjugate(psi[:,v1]) * psi[:,v2]
                hmat[v1,v2] = np.sum(fint * w * jacob)

        # check if Hamiltonian is hermitian
        if np.allclose(hmat, np.conjugate(hmat.T), atol=symmetric_tol) == False:
            raise RuntimeError(f"Hamiltonian matrix is not hermitian (tol = {symmetric_tol})")

        # diagonalize Hamiltonian
        eigval, eigvec = np.linalg.eigh(hmat)

        # transform basis
        psi = np.dot(psi, eigvec.T)
        dpsi = np.dot(dpsi, eigvec.T)
        print(eigval-eigval[0])


if __name__=="__main__":

    # H2S, using valence-bond coordinates and Tyuterev potential
    h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    # test KEO and potential
    G = h2s.G(np.array([ref_coords]))
    V = h2s.V(np.array([ref_coords]))

    angBas = LegCos(h2s, ref_coords, 2, 100, 30, [0, np.pi], verbose=True)
    strBas = Hermite(h2s, ref_coords, 0, 100, 30, [0, 30], verbose=True)
