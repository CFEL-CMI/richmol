"""Tools for computing functions on Euler angular grids, such as symmetric-top
and Wigner D-functions, rotational density functions, vibrationally averaged
tensor matrix elements, etc.
"""
import numpy as np
from ctypes import c_double, c_int
import sys
import os
from richmol_data import read_coefs, read_vibme
import itertools


# load Fortran library symtoplib
symtoplib_path = os.path.join(os.path.dirname(__file__), '../symtoplib')
fsymtop = np.ctypeslib.load_library('symtoplib', symtoplib_path)


def DJmk(Jmin, Jmax, grid, symtop=False):
    """Computes Wigner D-functions D_{m,k}^{(J)} on a grid of Euler angles
    for J=Jmin..Jmax and m,k=-J..J.

    Args:
        Jmin, Jmax : int
            Min and max values of J.
        grid : numpy.ndarray (3,no_grid_points)
            3D grid of values of Euler angles, grid[:3,ipoint] = (phi, theta, chi).
        symtop : bool
            If set to True, instead of Wigner D-functions, the symmetric-top
            functions will be returned.

    Returns:
        D : numpy.ndarray (no_grid_points,2*Jmax+1,2*Jmax+1,Jmax-Jmin+1)
           D_{m,k}^{(J)} = D[ipoint,m+J,k+J,J-Jmin].
    """
    assert (Jmax>=Jmin), f"max J = {Jmax} is lower than min J = {Jmin}"
    npoints = grid.shape[1]
    grid3d = np.asfortranarray(grid)
    jkm_real = np.asfortranarray(np.zeros((npoints,2*Jmax+1,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jkm_imag = np.asfortranarray(np.zeros((npoints,2*Jmax+1,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jmin_c = c_int(Jmin)
    jmax_c = c_int(Jmax)
    npoints_c = c_int(npoints)

    fsymtop.symtop_3d_grid.argtypes = [ \
        c_int, \
        c_int, \
        c_int, \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=4, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=4, flags='F') ]

    fsymtop.symtop_3d_grid.restype = None
    fsymtop.symtop_3d_grid(npoints_c, jmin_c, jmax_c, grid3d, jkm_real, jkm_imag)

    jkm = jkm_real.reshape((npoints,2*Jmax+1,2*Jmax+1,Jmax-Jmin+1)) \
        + jkm_imag.reshape((npoints,2*Jmax+1,2*Jmax+1,Jmax-Jmin+1))*1j

    if symtop==True:
        D = jkm
    else:
        # Wigner D-functions [D_{m,k}^{(j)}]^* from symmetric-top functions |j,k,m>
        D = np.zeros(jkm.shape, jkm.dtype)
        for j in range(Jmin, Jmax+1):
            D[:,:,:,j-Jmin] = np.conjugate(jkm[:,:,:,j-Jmin]) / np.sqrt((2*j+1)/(8.0*np.pi**2))
    return D


def DJk_m_3D(Jmin, Jmax, m, grid, symtop=False):
    """Computes Wigner D-functions D_{m,k}^{(J)} on a 3D grid of Euler angles
    for J=Jmin..Jmax, k=-J..J, and selected value of m.

    Args:
        Jmin, Jmax : int
            Min and max values of J.
        m : int
            Value of m.
        grid : numpy.ndarray (3,no_grid_points)
            3D grid of values of Euler angles, grid[:3,ipoint] = (phi, theta, chi).
        symtop : bool
            If set to True, instead of Wigner D-functions, the symmetric-top
            functions will be returned.

    Returns:
        D : numpy.ndarray (no_grid_points,2*Jmax+1,Jmax-Jmin+1)
           D_{m,k}^{(J)} = D[ipoint,k+J,J-Jmin].
    """
    assert (Jmax>=Jmin), f"max J = {Jmax} is lower than min J = {Jmin}"
    assert (Jmin>=m and Jmax>=m), f"max J = {Jmax} or min J = {Jmin} is smaller than m = {m}"
    npoints = grid.shape[1]
    grid3d = np.asfortranarray(grid)
    jkm_real = np.asfortranarray(np.zeros((npoints,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jkm_imag = np.asfortranarray(np.zeros((npoints,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jmin_c = c_int(Jmin)
    jmax_c = c_int(Jmax)
    m_c = c_int(m)
    npoints_c = c_int(npoints)

    fsymtop.symtop_3d_grid_m.argtypes = [ \
        c_int, \
        c_int, \
        c_int, \
        c_int, \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=3, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=3, flags='F') ]

    fsymtop.symtop_3d_grid.restype = None
    fsymtop.symtop_3d_grid(npoints_c, jmin_c, jmax_c, m, grid3d, jkm_real, jkm_imag)

    jkm = jkm_real.reshape((npoints,2*Jmax+1,Jmax-Jmin+1)) \
        + jkm_imag.reshape((npoints,2*Jmax+1,Jmax-Jmin+1))*1j

    if symtop==True:
        D = jkm
    else:
        # Wigner D-functions [D_{m,k}^{(j)}]^* from symmetric-top functions |j,k,m>
        D = np.zeros(jkm.shape, jkm.dtype)
        for j in range(Jmin, Jmax+1):
            D[:,:,j-Jmin] = np.conjugate(jkm[:,:,j-Jmin]) / np.sqrt((2*j+1)/(8.0*np.pi**2))
    return D


def DJk_m_2D(Jmin, Jmax, m, grid, symtop=False):
    """ Computes Wigner D-functions D_{m,k}^{(J)} on a 2D grid of Euler angles
    for J=Jmin..Jmax, k=-J..J, and selected value of m.

    Args:
        Jmin, Jmax : int
            Min and max values of J.
        m : int
            Value of m.
        grid : numpy.ndarray (2,no_grid_points)
            2D grid of values of Euler angles, grid[:2,ipoint] = (theta, chi).
        symtop : bool
            If set to True, instead of Wigner D-functions, the symmetric-top
            functions will be returned.

    Returns:
        D : numpy.ndarray (no_grid_points,2*Jmax+1,Jmax-Jmin+1)
           D_{m,k}^{(J)} = D[ipoint,k+J,J-Jmin].
    """
    assert (Jmax>=Jmin), f"max J = {Jmax} is lower than min J = {Jmin}"
    assert (Jmin>=m and Jmax>=m), f"max J = {Jmax} or min J = {Jmin} is smaller than m = {m}"
    npoints = grid.shape[1]
    grid2d = np.asfortranarray(grid)
    jkm_real = np.asfortranarray(np.zeros((npoints,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jkm_imag = np.asfortranarray(np.zeros((npoints,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jmin_c = c_int(Jmin)
    jmax_c = c_int(Jmax)
    m_c = c_int(m)
    npoints_c = c_int(npoints)

    fsymtop.symtop_2d_grid_m.argtypes = [ \
        c_int, \
        c_int, \
        c_int, \
        c_int, \
        np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=3, flags='F'), \
        np.ctypeslib.ndpointer(np.float64, ndim=3, flags='F') ]

    fsymtop.symtop_2d_grid_m.restype = None
    fsymtop.symtop_2d_grid_m(npoints_c, jmin_c, jmax_c, m, grid2d, jkm_real, jkm_imag)

    jkm = jkm_real.reshape((npoints,2*Jmax+1,Jmax-Jmin+1)) \
        + jkm_imag.reshape((npoints,2*Jmax+1,Jmax-Jmin+1))*1j

    if symtop==True:
        D = jkm
    else:
        # Wigner D-functions [D_{m,k}^{(j)}]^* from symmetric-top functions |j,k,m>
        D = np.zeros(jkm.shape, jkm.dtype)
        for j in range(Jmin, Jmax+1):
            D[:,:,j-Jmin] = np.conjugate(jkm[:,:,j-Jmin]) / np.sqrt((2*j+1)/(8.0*np.pi**2))
    return D


def tensor_2D(filename_vibme, filename_coefs, npoints, J_id_m_bra, J_id_m_ket, \
              coef_thresh=1e-12):
    """Computes matrix element of tensor operator between two stationary
    rovibrational states, where the integration takes place only over vibrational
    cooridnates and rotational phi-angle (associated with the 'm' quantum number),
    while the two other rotational angles, theta and chi (associated with the 'k'
    quantum number) are sampled on a 2D grid.

    Args:
        filename_vibme : str
            Name of Richmol vibrational matrix elements file.
        filename_coefs : str
            Name of Richmol wavefunctions coefficients file.
        npoints : int
            Number of points in the 2D grid sampling theta and chi angles.
        J_id_m_bra and J_id_m_ket : list of three elements
            Quantum numbers [J, id, m] of bra and ket stationary states.
        coef_thresh : float
            Threshold for neglecting wavefunction coefficients redd from Richmol
            coefficients file.

    Returns:
        res : numpy.ndarray((dimen,npoints_), dtype=complex128)
            Vibrationally-averaged matrix element on 2D grid, where dimen is the
            number of tensor elements (in vibrational matrix elements file filename_vibme),
            and npoints_=int(npoints**(1/2))**2.
        grid_2d : numpy.ndarray((2,npoints_))
            2D grid of Euler angles grid_2d[:2,ipoint] = (theta, chi).
    """
    # generate 2D grid of Euler angles theta and chi
 
    npt = int(npoints**(1/2)) # number of points in 1D
    theta  = list(np.linspace(0, np.pi, num=npt, endpoint=True))
    chi = list(np.linspace(0, 2*np.pi, num=npt, endpoint=True))
    grid = [theta, chi]
    grid_2d = np.array(list(itertools.product(*grid))).T

    # read Richmol coefficients file
    coefs = read_coefs(filename_coefs, coef_thresh=coef_thresh)

    # read vibraitonal matrix elements file
    vibme = read_vibme(filename_vibme)

    try:
        ibra = [[j,id] for j,id in zip(coefs['J'],coefs['id'])].index(J_id_m_bra[:2])
    except ValueError:
        raise ValueError(f"could not find state with [J, id] = {J_id_m_bra} in file {filename_coefs}") from None

    try:
        iket = [[j,id] for j,id in zip(coefs['J'],coefs['id'])].index(J_id_m_ket[:2])
    except ValueError:
        raise ValueError(f"could not find state with [J, id] = {J_id_m_ket} in file {filename_coefs}") from None

    # precompute symmetric-top functions on grid
    jkm_bra = DJk_m_2D(J_id_m_bra[0], J_id_m_bra[0], J_id_m_bra[2], grid_2d, symtop=True)
    jkm_ket = DJk_m_2D(J_id_m_ket[0], J_id_m_ket[0], J_id_m_ket[2], grid_2d, symtop=True)

    # bra-state
    #   primitive functions on grid
    J = J_id_m_bra[0]
    fbra = np.array( [ jkm_bra[:,k+J,0]*c for c,k in zip(coefs['c'][ibra,:], coefs['k'][ibra,:]) ] )
    vbra = list(set(coefs['v'][ibra,:]))
    #   contract functions with same v quanta
    ind = [np.where(coefs['v'][ibra,:]==v) for v in vbra]
    fbra = np.array([np.sum(fbra[i], axis=0) for i in ind])

    # ket-state
    #   primitive functions on grid
    J = J_id_m_ket[0]
    fket = np.array( [ jkm_ket[:,k+J,0]*c for c,k in zip(coefs['c'][iket,:], coefs['k'][iket,:]) ] )
    vket = list(set(coefs['v'][iket,:]))
    #   contract functions with same v quanta
    ind = [np.where(coefs['v'][iket,:]==v) for v in vket]
    fket = np.array([np.sum(fket[i], axis=0) for i in ind])

    # extract vibrational matrix elements of tensor that enter fbra and fket
    vibme_reduced = vibme[np.ix_([i for i in range(vibme.shape[0])],vbra,vket)]

    # contract bra and ket functions with vibrational tensor matrix elements
    #   and multiply with spherical volume element
    sintheta_twopi = np.sin(grid_2d[0,:]) * 2.0 * np.pi # 2Pi comes from integration over phi
    tmat = np.dot(vibme_reduced, fket)
    res = np.einsum('ij,kij->kj', np.conjugate(fbra), tmat) * sintheta_twopi
    return res, grid_2d


def test_tensor_2D_polariz_d2s():
    # test tensor_2D for matrix elements of polarizability tensor of D2S molecule

    filename_vibme = "../../etc/utils_test/d2s/extfield_vibme.txt.gz"
    filename_coefs = "../../etc/utils_test/d2s/coefficients_j0_j60.rchm.gz"
    filename_out = "../../etc/utils_test/d2s/d2s_alpha_vibme_2D.txt"
    me, grid = tensor_2D(filename_vibme, filename_coefs, 100000, [0,1,0], [2,1,-2])

    with open(filename_out, "w") as fl:
        for i in range(me.shape[1]):
            fl.write("%16.8e"%grid[0,i] + "%16.8e"%grid[1,i] + \
                     "    ".join(" %16.8e %16.8e"%(elem.real, elem.imag) for elem in me[:,i]) + "\n" )


if __name__ == "__main__":
    test_tensor_2D_polariz_d2s()
