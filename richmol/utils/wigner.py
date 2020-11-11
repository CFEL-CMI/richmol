""" Tools for computing Wigner functions """
import numpy as np
from ctypes import c_double, c_int
import sys
import os

# load Fortran library symtoplib
symtoplib_path = os.path.join(os.path.dirname(__file__), '../symtoplib')
fsymtop = np.ctypeslib.load_library('symtoplib', symtoplib_path)


def DJmk(Jmin, Jmax, grid):
    """ Computes Wigner D-functions D_{mk}^{(J)} on a grid of Euler angles
    for J=Jmin..Jmax and m,k=-J..J

    Args:
        Jmin, Jmax : int
            Min and max values of J
        grid : numpy.ndarray (3,no_grid_points)
            3D grid of values of Euler angles, grid[:3,ipoint] = (phi, theta, chi)
    Returns:
        D : numpy.ndarray (no_grid_points,2*Jmax+1,2*Jmax+1,Jmin-Jmax+1)
           D_{m,k}^{(J)} = D[ipoint,m+J,k+J,J-Jmin]
    """
    assert (Jmax>=Jmin), f"max J = {Jmax} is lower than min J = {Jmin}"
    npoints3d = grid.shape[1]
    grid3d = np.asfortranarray(grid)
    jkm_real = np.asfortranarray(np.zeros((npoints,2*Jmax+1,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jkm_imag = np.asfortranarray(np.zeros((npoints,2*Jmax+1,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jmin_c = c_int(Jmin)
    jmax_c = c_int(Jmax)
    npoints_c = c_int(npoints3d)

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

    # Wigner D-functions [D_{m,k}^{(j)}]^* from symmetric-top functions |j,k,m>
    for j in range(Jmin, Jmax+1):
        D[:,:,:,j-Jmin] = jkm[:,:,:,j-Jmin] / np.sqrt((2*j+1)/(8.0*np.pi**2))
    return D


def DJk_m(Jmin, Jmax, m, grid):
    """ Computes Wigner D-functions D_{mk}^{(J)} on a 3D grid of Euler angles
    for J=Jmin..Jmax, k=-J..J, and selected value of m

    Args:
        Jmin, Jmax : int
            Min and max values of J
        m : int
            Value of m
        grid : numpy.ndarray (3,no_grid_points)
            3D grid of values of Euler angles, grid[:3,ipoint] = (phi, theta, chi)
    Returns:
        D : numpy.ndarray (no_grid_points,2*Jmax+1,Jmin-Jmax+1)
           D_{m,k}^{(J)} = D[ipoint,k+J,J-Jmin]
    """
    assert (Jmax>=Jmin), f"max J = {Jmax} is lower than min J = {Jmin}"
    assert (Jmin>=m and Jmax>=m), f"max J = {Jmax} or min J = {Jmin} is smaller than m = {m}"
    npoints3d = grid.shape[1]
    grid3d = np.asfortranarray(grid)
    jkm_real = np.asfortranarray(np.zeros((npoints,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jkm_imag = np.asfortranarray(np.zeros((npoints,2*Jmax+1,Jmax-Jmin+1), dtype=np.float64))
    jmin_c = c_int(Jmin)
    jmax_c = c_int(Jmax)
    m_c = c_int(m)
    npoints_c = c_int(npoints3d)

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

    # Wigner D-functions [D_{m,k}^{(j)}]^* from symmetric-top functions |j,k,m>
    for j in range(Jmin, Jmax+1):
        D[:,:,j-Jmin] = jkm[:,:,j-Jmin] / np.sqrt((2*j+1)/(8.0*np.pi**2))
    return D

if __name__ == "__main__":
    print(symtoplib_path)
