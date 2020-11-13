import numpy as np
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
            raise ValueError(f"Some quadrature points with significant weight " \
                +f"fall outside of the coordinate ranges = {ranges}") from None

        points_in = (r>=ranges[0]) & (r<=ranges[1])
        return x[points_in], w[points_in], r[points_in]


    def PObas(self, molec, ref_coords, icoord, verbose=True):
        """Generates potential-optimized basis

        Args:
            molec (Molecule): Information about molecule, KEO and PES.
            ref_coords (array (no_coords)): Reference values of all internal coordinates.
            icoord (int): Index of internal coordinate for which the basis is generated.
            verbose (bool): Set to 'True' if you want to print out some intermediate data.
        """
        # potential and keo on quadrature grid

        coords = np.array(np.broadcast_to(ref_coords, (len(self.r),len(ref_coords))))
        coords[:,icoord] = self.r[:]
        gmat = molec.G(coords)[:,icoord,icoord]
        poten = molec.V(coords)

        # print values on quadrature grid
        if verbose==True:
            print("Operators on 1D quadrature grid\n" + "%23s"%"x" + "%18s"%"w" \
                + "%18s"%"r" + "%18s"%"keo" + "%18s"%"poten")
            for i in range(len(self.r)):
                print(" %4i"%i + "  %16.8f"%self.x[i] + "  %16.8e"%self.w[i] \
                    + "  %16.8f"%self.r[i] + "  %16.8f"%gmat[i] + "  %16.8f"%poten[i])

        # overlap matrix
        ovlp = np.zeros((self.vmax, self.vmax), dtype=np.float64)
        for v1 in range(self.vmax):
            for v2 in range(self.vmax):
                ovlp[v1,v2] = np.sum( np.conjugate(self.psi[:,v1]) * self.psi[:,v2] \
                                     * self.w[:] * self.jacob[:] )

        # check if overlap matrix is symmetric
        if np.allclose(ovlp, ovlp.T, atol=symmetric_tol) == False:
            raise RuntimeError(f"Overlap matrix is not symmetric (tol = {symmetric_tol})")

        # inverse square root of overlap matrix
        val, vec = np.linalg.eigh(ovlp)
        if np.any(np.abs(val) < singular_tol):
            raise RuntimeError(f"Overlap matrix is singular (tol = {singular_tol})") from None
        d = np.diag(1.0/np.sqrt(val))
        sqrt_inv = np.dot(vec, np.dot(d, vec.T))

        # orthonormalize basis functions
        self.psi = np.dot(self.psi, sqrt_inv.T)
        self.dpsi = np.dot(self.dpsi, sqrt_inv.T)

        # Hamiltonian matrix
        hmat = np.zeros((self.vmax, self.vmax), dtype=np.float64)
        for v1 in range(self.vmax):
            for v2 in range(self.vmax):
                fint = 0.5 * gmat * np.conjugate(self.dpsi[:,v1]) * self.dpsi[:,v2] \
                     + poten * np.conjugate(self.psi[:,v1]) * self.psi[:,v2]
                hmat[v1,v2] = np.sum(fint * self.w * self.jacob) + self.uv[v1,v2]

        # check if Hamiltonian is hermitian
        if np.allclose(hmat, np.conjugate(hmat.T), atol=symmetric_tol) == False:
            raise RuntimeError(f"Hamiltonian matrix is not hermitian (tol = {symmetric_tol})")

        # diagonalize Hamiltonian
        eigval, eigvec = np.linalg.eigh(hmat)

        # transform basis
        self.psi = np.dot(self.psi, eigvec.T)
        self.dpsi = np.dot(self.dpsi, eigvec.T)
        self.enr = eigval-eigval[0]


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
        """
        # quadratures
        x, w = leggauss(no_points)

        # quadrature abscissas -> coordinate values
        r = np.arccos(x)

        # delete points that fall outside the coordinate ranges
        self.x, self.w, self.r = self.check_outliers(x, w, r, ranges, zero_weight_thresh)

        # basis functions and derivatives

        self.vmax = vmax + 1
        self.psi = np.zeros((len(self.r),self.vmax), dtype=np.float64)  # basis functions
        self.dpsi = np.zeros((len(self.r),self.vmax), dtype=np.float64) # derivatives of basis functions
        self.jacob = np.ones(len(self.r), dtype=np.float64)             # Jacobian due to changing a variable from x to r
        self.uv = np.zeros((self.vmax,self.vmax), dtype=np.float64)     # some extra terms arising in some cases from the integration by parts

        coefs = np.zeros(self.vmax, dtype=np.float64)

        for v in range(self.vmax):
            coefs[:] = 0
            coefs[v] = 1.0
            pol = legval(self.x, coefs)
            dpol = legval(self.x, legder(coefs, m=1))
            scale_fac = np.sqrt((2.0*v+1)*0.5)
            self.psi[:,v] = pol * scale_fac
            self.dpsi[:,v] = -np.sin(self.r) * dpol * scale_fac

        # Jacobian because we change variable from r=[0,pi] to x=[-1,1],
        #   which gives dr -> dx * 1/sin(r)
        self.jacob = 1.0/np.sin(self.r)

        # generate potential-optimized basis
        self.PObas(molec, ref_coords, icoord, verbose)


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
        self.x, self.w, self.r = self.check_outliers(x, w, r, ranges, zero_weight_thresh)

        # basis functions and derivatives

        self.vmax = vmax + 1
        self.psi = np.zeros((len(self.r),self.vmax), dtype=np.float64)  # basis functions
        self.dpsi = np.zeros((len(self.r),self.vmax), dtype=np.float64) # derivatives of basis functions
        self.jacob = np.ones(len(self.r), dtype=np.float64)             # Jacobian due to changing a variable from x to r
        self.uv = np.zeros((self.vmax,self.vmax), dtype=np.float64)     # some extra terms arising in some cases from the integration by parts

        coefs = np.zeros(self.vmax, dtype=np.float64)
        sqsqpi = np.sqrt(np.sqrt(np.pi))

        for v in range(self.vmax):
            coefs[:] = 0
            coefs[v] = 1.0
            pol = hermval(self.x, coefs)
            dpol = hermval(self.x, hermder(coefs, m=1))
            scale_fac = 1.0/np.sqrt(2.0**v*np.math.factorial(v))/sqsqpi
            self.psi[:,v] = pol * scale_fac
            self.dpsi[:,v] = (dpol - pol * self.x) * scale_fac * xmap

        # generate potential-optimized basis
        self.PObas(molec, ref_coords, icoord, verbose)



if __name__=="__main__":

    from mol_xy2 import XY2_ralpha
    import poten_h2s_Tyuterev
    import poten_h2o_Polyansky

    # H2S, using valence-bond coordinates and Tyuterev potential
    h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    # test KEO and potential
    G = h2s.G(np.array([ref_coords]))
    V = h2s.V(np.array([ref_coords]))

    angBas = LegCos(h2s, ref_coords, 2, 100, 60, [0, np.pi], verbose=True)
    strBas = Hermite(h2s, ref_coords, 0, 100, 30, [0.6, 10], verbose=True)

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

    print("richmol-TROVE for stretching")
    print(" ".join("  %12.4f"%e1 + "  %12.4f"%e2 + "  %12.4f"%(e1-e2) + "\n" for e1,e2 in zip(strBas.enr, trove_str_enr)))