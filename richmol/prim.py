import sympy as sp
import numpy as np
import math
from numpy.polynomial.legendre import leggauss, legval, legder
from numpy.polynomial.hermite import hermgauss, hermval, hermder
from numpy.polynomial.laguerre import laggauss, lagval, lagder


singular_tol = 1e-10 # tolerance for considering matrix singular
symmetric_tol = 1e-10 # tolerance for considering matrix symmetric


def legcos(icoord, ref_coords, npoints, vmax, ranges, poten, gmat, \
           pseudo=None, dgmat=None, verbose=False, zero_weight_thresh=1e-20):
    """Generates one-dimensional basis of potential-optimized Legendre(cos)
    orthogonal polynomials

    Args:

        icoord : int
            Coordinate number for which the 1D basis is generated

        ref_coords : array (no_coords)
            Reference/equilibrium values for all internal coordinates

        npoints : int
            Number of Gauss-Legendre quadrature points

        vmax : int
            Maximal vibrational quantum number spanned by the basis

        ranges : list [min, max]
            Variation ranges for internal coordinate icoord,
            quadrature points that fall outside of this ranges and have
            the corresponding weight smaller than a threshold zero_weight_thresh
            will be neglected

        poten : function
            Potential energy function: poten(coords) = scalar, where coords
            is a numpy ndarray of shape (no_points, no_coords) containing values
            of internal coordinates on grid of no_points points

        gmat : function
            Kinetic energy G matrix: gmat(coords) = array(no_atoms*3, no_atoms*3),
            where coords is a list of length (no_coords) containing values
            of internal cooridnates

        pseudo : function
            Kinetic energy pseudpotential: pseudo(coords) = scalar,
            where coords is a list of length (no_coords) containing values
            of internal cooridnates

        dgmat : function
            Derivative of kinetic energy G matrix wrt internal cooridnates:
            dgmat(coords) = array(no_coords, no_atoms*3, no_atoms*3),
            where coords is a list of length (no_coords) containing values
            of internal cooridnates

        verbose : bool
            Set to 'True' if you want to print out some intermediate data

        zero_weight_thresh : float
            Threshold for the quadrature weight below which the corresponding
            quadrature point can be neglected

    Returns:

        r : array (npoints)
            Values of icoord coordinate on Gauss-Legendre quadrature grid

        enr : array (vmax + 1)
            Basis state energies

        psi : array (npoints, vmax + 1)
            Basis state wave functions on quadrature grid

        dpsi : array (npoints, vmax + 1)
            Derivatives of basis state wave functions on quadrature grid
    """
    assert (vmax < npoints), f"Number of quadrature npoints = {npoints} < vmax = {vmax}"

    # quadratures
    x, w = leggauss(npoints)

    # quadrature abscissas -> coordinate values
    r = np.arccos(x)

    # delete points that fall outside the coordinate ranges
    x, w, r = check_outliers(x, w, r, ranges, zero_weight_thresh)

    # basis functions and derivatives

    psi = np.zeros((len(r), vmax + 1), dtype=np.float64)  # basis functions
    dpsi = np.zeros((len(r), vmax + 1), dtype=np.float64) # derivatives of basis functions
    jacob = np.ones(len(r), dtype=np.float64)             # Jacobian due to changing a variable from x to r
    coefs = np.zeros(vmax + 1, dtype=np.float64)

    for v in range(vmax + 1):
        coefs[:] = 0
        coefs[v] = 1.0
        pol = legval(x, coefs)
        dpol = legval(x, legder(coefs, m=1))
        scale_fac = np.sqrt((2.0 * v + 1) * 0.5)
        psi[:,v] = pol * scale_fac
        dpsi[:,v] = -np.sin(r) * dpol * scale_fac

    # Jacobian because we change variable from r=[0,pi] to x=[-1,1],
    #   which gives dr -> dx * 1/sin(r)
    jacob = 1.0/np.sin(r)

    # coordinate values on grid
    coords = np.array(np.broadcast_to(ref_coords, (len(r),len(ref_coords))))
    coords[:,icoord] = r[:]

    # potential, g-matrix, and pseudopotential on grid
    G = np.array([gmat(list(coord))[icoord, icoord] for coord in coords], np.float64)
    V = poten(coords)
    try:
        dG = np.array([dgmat(list(coord))[icoord, icoord, icoord] for coord in coords], np.float64)
    except TypeError:
        dG = np.zeros(len(coords))
    try:
        U = np.array([pseudo(list(coord)) for coord in coords], np.float64)
    except TypeError:
        U = np.zeros(len(coords))

    # print values of operators on grid
    if verbose==True:
        print("Operators on 1D grid for coordinate no. %3i"%icoord \
              + "  (ref values = " + "".join("%12.6f"%x for x in ref_coords) + ") \n" \
              + "%23s"%"r" + "%18s"%"G"  + "%18s"%"dG" + "%18s"%"V" + "%18s"%"U")
        for i in range(len(r)):
            print(" %4i"%i + "  %16.8f"%r[i] + "  %16.8f"%G[i] + "  %16.8f"%dG[i] \
                  + "  %16.8f"%V[i] + "  %16.8f"%U[i])

    # overlap matrix
    ovlp = np.zeros((vmax + 1, vmax + 1), dtype=np.float64)
    for v1 in range(vmax + 1):
        for v2 in range(vmax + 1):
            ovlp[v1,v2] = np.sum(np.conjugate(psi[:,v1]) * psi[:,v2] * w[:] * jacob[:])

    # check if overlap matrix is symmetric
    if np.allclose(ovlp, ovlp.T, atol=symmetric_tol) == False:
        raise RuntimeError(f"Overlap matrix is not symmetric (tol = {symmetric_tol})")

    # inverse square root of overlap matrix
    val, vec = np.linalg.eigh(ovlp)
    if np.any(np.abs(val) < singular_tol):
        raise RuntimeError(f"Overlap matrix is singular (tol = {singular_tol})") from None
    d = np.diag(1.0 / np.sqrt(val))
    sqrt_inv = np.dot(vec, np.dot(d, vec.T))

    # orthonormalize basis functions
    psi = np.dot(psi, sqrt_inv.T)
    dpsi = np.dot(dpsi, sqrt_inv.T)

    # Hamiltonian matrix
    hmat = np.zeros((vmax + 1, vmax + 1), dtype=np.float64)
    for v1 in range(vmax + 1):
        for v2 in range(vmax + 1):
            fint = 0.5 * G * np.conjugate(dpsi[:,v1]) * dpsi[:,v2] \
                 + (V + U) * np.conjugate(psi[:,v1]) * psi[:,v2]
            hmat[v1,v2] = np.sum(fint * w * jacob)

    # check if Hamiltonian is hermitian
    if np.allclose(hmat, np.conjugate(hmat.T), atol=symmetric_tol) == False:
        raise RuntimeError(f"Hamiltonian matrix is not hermitian (tol = {symmetric_tol})")

    # diagonalize Hamiltonian
    eigval, eigvec = np.linalg.eigh(hmat)

    # transform basis
    psi = np.dot(psi, eigvec.T)
    dpsi = np.dot(dpsi, eigvec.T)

    return r, eigval - eigval[0], psi, dpsi


def herm(icoord, ref_coords, npoints, vmax, ranges, poten, gmat, \
         pseudo=None, dgmat=None, verbose=False, zero_weight_thresh=1e-20, \
         fdf_h=0.001):
    """Generates one-dimensional basis of potential-optimized Hermite orthogonal
    polynomials

    Args:

        icoord : int
            Coordinate number for which the 1D basis is generated

        ref_coords : array (no_coords)
            Reference/equilibrium values for all internal coordinates

        npoints : int
            Number of Gauss-Hermite quadrature points

        vmax : int
            Maximal vibrational quantum number spanned by the basis

        ranges : list [min, max]
            Variation ranges for internal coordinate icoord,
            quadrature points that fall outside of this ranges and have
            the corresponding weight smaller than a threshold zero_weight_thresh
            will be neglected

        poten : function
            Potential energy function: poten(coords) = scalar, where coords
            is a numpy ndarray of shape (no_points, no_coords) containing values
            of internal coordinates on grid of no_points points

        gmat : function
            Kinetic energy G matrix: gmat(coords) = array(no_atoms*3, no_atoms*3),
            where coords is a list of length (no_coords) containing values
            of internal cooridnates

        pseudo : function
            Kinetic energy pseudpotential: pseudo(coords) = scalar,
            where coords is a list of length (no_coords) containing values
            of internal cooridnates

        dgmat : function
            Derivative of kinetic energy G matrix wrt internal cooridnates:
            dgmat(coords) = array(no_coords, no_atoms*3, no_atoms*3),
            where coords is a list of length (no_coords) containing values
            of internal cooridnates

        verbose : bool
            Set to 'True' if you want to print out some intermediate data

        zero_weight_thresh : float
            Threshold for the quadrature weight below which the corresponding
            quadrature point can be neglected

        fdf_h : float
            Step size for finite-difference calculation of the second-order
            derivative of potential function

    Returns:

        r : array (npoints)
            Values of icoord coordinate on Gauss-Hermite quadrature grid

        enr : array (vmax + 1)
            Basis state energies

        psi : array (npoints, vmax + 1)
            Basis state wave functions on quadrature grid

        dpsi : array (npoints, vmax + 1)
            Derivatives of basis state wave functions on quadrature grid
    """
    assert (vmax < npoints), f"Number of quadrature npoints = {npoints} < vmax = {vmax}"

    # quadratures
    x, w = hermgauss(npoints)

    # quadrature abscissas -> coordinate values

    G = gmat(ref_coords)[icoord, icoord]
    # use finite-differences (7-point) to compute frequency
    fdf_steps = np.array([3*fdf_h, 2*fdf_h, fdf_h, 0.0, -fdf_h, -2*fdf_h, -3*fdf_h], dtype=np.float64)
    fdf_coefs = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], dtype=np.float64)
    fdf_denom = 180.0
    coords = np.array(np.broadcast_to(ref_coords, (len(fdf_steps),len(ref_coords))))
    coords[:,icoord] = [ref_coords[icoord] + st for st in fdf_steps]
    V = poten(coords)
    freq = np.dot(V, fdf_coefs) / (fdf_denom * fdf_h * fdf_h)
    # mapping between r and x
    xmap = np.sqrt(np.sqrt( 2.0 * np.abs(freq) / np.abs(G) ))
    r = x / xmap + ref_coords[icoord]
    if verbose == True:
         print(f"Mapping x <--> r calculated for Gauss-Hermite quadrature: {xmap}, (mu = {gmat}, omega = {freq})")

    # delete points that fall outside the coordinate ranges
    x, w, r = check_outliers(x, w, r, ranges, zero_weight_thresh)

    # basis functions and derivatives

    psi = np.zeros((len(r), vmax + 1), dtype=np.float64)  # basis functions
    dpsi = np.zeros((len(r), vmax + 1), dtype=np.float64) # derivatives of basis functions
    coefs = np.zeros(vmax + 1, dtype=np.float64)
    sqsqpi = np.sqrt(np.sqrt(np.pi))

    for v in range(vmax + 1):
        coefs[:] = 0
        coefs[v] = 1.0
        pol = hermval(x, coefs)
        dpol = hermval(x, hermder(coefs, m=1))
        scale_fac = 1.0 / np.sqrt(2.0**v * math.factorial(v)) / sqsqpi
        psi[:, v] = pol * scale_fac
        dpsi[:, v] = (dpol - pol * x) * scale_fac * xmap

    # coordinate values on grid
    coords = np.array(np.broadcast_to(ref_coords, (len(r),len(ref_coords))))
    coords[:,icoord] = r[:]

    # potential, g-matrix, and pseudopotential on grid
    G = np.array([gmat(list(coord))[icoord, icoord] for coord in coords], np.float64)
    V = poten(coords)
    try:
        dG = np.array([dgmat(list(coord))[icoord, icoord, icoord] for coord in coords], np.float64)
    except TypeError:
        dG = np.zeros(len(coords))
    try:
        U = np.array([pseudo(list(coord)) for coord in coords], np.float64)
    except TypeError:
        U = np.zeros(len(coords))

    # print values of operators on grid
    if verbose==True:
        print("Operators on 1D grid for coordinate no. %3i"%icoord \
              + "  (ref values = " + "".join("%12.6f"%x for x in ref_coords) + ") \n" \
              + "%23s"%"r" + "%18s"%"G"  + "%18s"%"dG" + "%18s"%"V" + "%18s"%"U")
        for i in range(len(r)):
            print(" %4i"%i + "  %16.8f"%r[i] + "  %16.8f"%G[i] + "  %16.8f"%dG[i] \
                  + "  %16.8f"%V[i] + "  %16.8f"%U[i])

    # Hamiltonian matrix
    hmat = np.zeros((vmax + 1, vmax + 1), dtype=np.float64)
    for v1 in range(vmax + 1):
        for v2 in range(vmax + 1):
            fint = 0.5 * G * np.conjugate(dpsi[:,v1]) * dpsi[:,v2] \
                 + (V + U) * np.conjugate(psi[:,v1]) * psi[:,v2]
            hmat[v1,v2] = np.sum(fint * w)

    # check if Hamiltonian is hermitian
    #if np.allclose(hmat, np.conjugate(hmat.T), atol=symmetric_tol) == False:
    #    raise RuntimeError(f"Hamiltonian matrix is not hermitian (tol = {symmetric_tol})")

    # diagonalize Hamiltonian
    eigval, eigvec = np.linalg.eigh(hmat)

    # transform basis
    psi = np.dot(psi, eigvec.T)
    dpsi = np.dot(dpsi, eigvec.T)

    return r, eigval - eigval[0], psi, dpsi


def numerov(icoord, ref_coords, npoints, vmax, ranges, poten, gmat, \
            pseudo=None, dgmat=None, verbose=False, npoints_stencil=7):
    """Generates basis of one-dimensional Numerov-Cooley functions

    Args:

        icoord : int
            Coordinate number for which the 1D basis is generated

        ref_coords : array (no_coords)
            Reference/equilibrium values for all internal coordinates

        npoints : int
            Number of points in the Numerov equidistant grid

        vmax : int
            Maximal vibrational quantum number spanned by the basis

        ranges : list [min, max]
            Variation ranges for internal coordinate icoord

        poten : function
            Potential energy function: poten(coords) = scalar, where coords
            is a numpy ndarray of shape (no_points, no_coords) containing values
            of internal coordinates on grid of no_points points

        gmat : function
            Kinetic energy G matrix: gmat(coords) = array(no_atoms*3, no_atoms*3),
            where coords is a list of length (no_coords) containing values
            of internal cooridnates

        pseudo : function
            Kinetic energy pseudpotential: pseudo(coords) = scalar,
            where coords is a list of length (no_coords) containing values
            of internal cooridnates

        dgmat : function
            Derivative of kinetic energy G matrix wrt internal cooridnates:
            dgmat(coords) = array(no_coords, no_atoms*3, no_atoms*3),
            where coords is a list of length (no_coords) containing values
            of internal cooridnates

        verbose : bool
            Set to 'True' if you want to print out some intermediate data

        npoints_stencil : float
            Number of stencil points for finite differences

    Returns:

        r : array (npoints)
            Values of icoord coordinate on Numerov grid

        enr : array (vmax + 1)
            Basis state energies

        psi : array (npoints, vmax + 1)
            Basis state wave functions on Numerov grid
    """
    assert (vmax < npoints), f"Number of npoints = {npoints} < vmax = {vmax}"

    # coordinate values on grid
    r, step = np.linspace(*ranges, npoints, endpoint=True, retstep=True)
    coords = np.array(np.broadcast_to(ref_coords, (len(r),len(ref_coords))))
    coords[:,icoord] = r[:]

    # potential, g-matrix, and pseudopotential on grid
    G = np.array([gmat(list(coord))[icoord, icoord] for coord in coords], np.float64)
    V = poten(coords)
    try:
        dG = np.array([dgmat(list(coord))[icoord, icoord, icoord] for coord in coords], np.float64)
    except TypeError:
        dG = np.zeros(len(coords))
    try:
        U = np.array([pseudo(list(coord)) for coord in coords], np.float64)
    except TypeError:
        U = np.zeros(len(coords))

    # print values of operators on grid
    if verbose==True:
        print("Operators on 1D grid for coordinate no. %3i"%icoord \
              + "  (ref values = " + "".join("%12.6f"%x for x in ref_coords) + ") \n" \
              + "%23s"%"r" + "%18s"%"G"  + "%18s"%"dG" + "%18s"%"V" + "%18s"%"U")
        for i in range(len(r)):
            print(" %4i"%i + "  %16.8f"%r[i] + "  %16.8f"%G[i] + "  %16.8f"%dG[i] \
                  + "  %16.8f"%V[i] + "  %16.8f"%U[i])

    # finite-difference formulas
    stencil, fdf = fdf_stencil(npoints_stencil)

    # d/dr and d^2/dr^2 operators
    dr = 0
    ddr = 0
    for i,n in enumerate(stencil):
        # first derivative
        dr += fdf(step)[1,i] * np.diag([1 for i in range(npoints-abs(n))], k=n)
        # second derivative
        ddr += fdf(step)[2,i] * np.diag([1 for i in range(npoints-abs(n))], k=n)

    # Hamiltonian
    hmat = -0.5 * dG * dr  - 0.5 * G * ddr + np.diag(V + U)

    eigval, eigvec = np.linalg.eigh(hmat)

    return r, eigval[:vmax + 1] - eigval[0], eigvec[:, vmax + 1]


def fdf_stencil(npoints):
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


def check_outliers(x, w, r, ranges, zero_weight_thresh):
    """Removes quadrature points with a low weight (w < zero_weight_thresh)
    that fall outside of the coordinate ranges
    """
    points_out1 = (r < ranges[0]) & (w > zero_weight_thresh)
    points_out2 = (r > ranges[1]) & (w > zero_weight_thresh)
    if len(w[points_out1]) > 0 or len(w[points_out2]) > 0:
        raise ValueError(f"Some quadrature points with significant weight " \
            +f"fall outside of the coordinate ranges = {ranges}") from None
    points_in = (r >= ranges[0]) & (r <= ranges[1])
    return x[points_in], w[points_in], r[points_in]
