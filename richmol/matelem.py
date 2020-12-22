from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from mapping import indexmap,gridmap
from matplotlib import pyplot as plt
from test import eval_k
import sympy as sp
import numpy as np
import math
from numpy.polynomial.legendre import leggauss, legval, legder
from numpy.polynomial.hermite import hermgauss, hermval, hermder
import scipy.special as ss
from keo_jax import com, bisector
import keo_jax
import jax.numpy as jnp
import poten_h2s_Tyuterev
from prim import numerov, legcos, herm, laguerre
import sys
import jax
from jax import jit
import time
from jax.config import config
config.update("jax_enable_x64", True)
from numba import jit
#import scipy.special.eval_genlaguerre
#import scipy.special.roots_genlaguerre as laggauss
singular_tol = 1e-10 # tolerance for considering matrix singular
symmetric_tol = 1e-10 # tolerance for considering matrix symmetric
from prim import herm, legcos
import matplotlib.pyplot as plt


def init(*args):
    global w_tol
    global Nbas
    global Nquad1D_herm  #number of Gauss-Hermite quadrature points in 1D
    global Nquad1D_leg  #number of Gauss-Legendre quadrature points in 1D
    global ref_coords #reference coordinates
    global masses #nuclear masses

@com
@bisector('zxy')
def internal_to_cartesian(coords):
    r1, r2, alpha = coords
    xyz = jnp.array([[0.0, 0.0, 0.0], \
                    [ r1 * jnp.sin(alpha/2), 0.0, r1 * jnp.cos(alpha/2)], \
                    [-r2 * jnp.sin(alpha/2), 0.0, r2 * jnp.cos(alpha/2)]], \
                    dtype=jnp.float64)
    return xyz #this function can be called from XY2 or molecule module. It does not belong here.

@jit
def matelem_keo(psi_i, dpsi_i, psi_j, dpsi_j, qgrid_ind, G):
    """
    This routine calculates the matrix element of the kinetic energy operator.

    Input:
    psi_i: array (Nquad, 3) of values of the three basis functions (phi_r_i1,phi_r_i2,phi_theta_i3), whose indices correspond to multiindex ivec = (i1,i2,i3).
            The functions are evaluated on respective quadrature grids.
    dpsi_i: array (Nquad, 3) of values of the three basis functions derivatives (dphi_r_i1,dphi_r_i2,dphi_theta_i3), whose indices correspond to multiindex ivec = (i1,i2,i3).
            The derivatives are evaluated on respective quadrature grids.
    G: array (Ngrid,9,9)  values of G-matrix elements on the quadrature grid

    Returns:
    keo_elem: matrix element of the KEO
    """
    k1 = qgrid_ind[:,0]
    k2 = qgrid_ind[:,1]
    k3 = qgrid_ind[:,2]
    #print(np.shape( dpsi_j ))

    keo_elem = np.zeros((1,))
    start = time.time()
    """ lame way of calculating KEO """
    Ti1 = dpsi_i[k1[:], 0 ] * psi_i[k2[:], 1 ] * psi_i[k3[:], 2 ]
    Ti2 = psi_i[k1[:], 0 ]*dpsi_i[k2[:], 1]*psi_i[k3[:], 2 ] 
    Ti3 = psi_i[k1[:], 0 ]*psi_i[k2[:], 1]*dpsi_i[k3[:], 2 ]
    Tj1 = dpsi_j[k1[:], 0 ] * psi_j[k2[:], 1] * psi_j[k3[:], 2]
    Tj2 = psi_j[k1[:], 0 ]*dpsi_j[k2[:], 1]*psi_j[k3[:], 2 ] 
    Tj3 = psi_j[k1[:], 0 ]*psi_j[k2[:], 1]*dpsi_j[k3[:], 2 ]
    #print(np.shape(Ti1))
    #print(np.shape(G[:,0,0] ))
    #exit()
    keo_int = Ti1[:] * G[:,0,0]  *  Tj1[:] \
            + Ti1[:]  * G[:,0,1]  * Tj1[:] \
            + Ti1[:]  * G[:,0,2] * Tj1[:] \
            + Ti2[:]  * G[:,1,0] * Tj2[:]\
            + Ti2[:]  * G[:,1,1] * Tj2[:] \
            + Ti2[:]  * G[:,1,2]  * Tj2[:]\
            + Ti3[:]  * G[:,2,0] * Tj3[:]\
            + Ti3[:]  * G[:,2,1]  * Tj3[:] \
            + Ti3[:]  * G[:,2,2]  * Tj3[:]

    keo_elem = np.sum(keo_int)
    """for ipoint in range(Ngrid):
        keo_elem += dpsi_i[qgrid_ind[ipoint,0], 0 ] * psi_i[qgrid_ind[ipoint,1], 1 ] * psi_i[qgrid_ind[ipoint,2], 2 ] * G[ipoint][0][0]  * dpsi_j[qgrid_ind[ipoint,0], 0 ] * psi_j[qgrid_ind[ipoint,1], 1] * psi_j[qgrid_ind[ipoint,2], 2] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[ipoint][0][1]  * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[ipoint][0][2] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[ipoint][1][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[ipoint][1][1] * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[ipoint][1][2]  * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ] * G[ipoint][2][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ] * G[ipoint][2][1]  * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ] * G[ipoint][2][2]  * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ]
    """
    end = time.time()
    #print("time for KEO =  ", str(end-start))

    return keo_elem

@jit
def matelem_pes(psi_i, psi_j, x1,  x3, qgrid_ind):

    Ngrid = np.size(qgrid_ind,axis=0)
    qcoords = np.zeros((Ngrid,3))

    pot_int = np.zeros(Ngrid)
    pot_elem = np.zeros((1,))


    #print("Harmonic frequency")
    #print(freq)

    qcoords[:,0] = x1[qgrid_ind[:,0]]
    qcoords[:,1] = x1[qgrid_ind[:,1]]
    qcoords[:,2] = x3[qgrid_ind[:,2]]

    pot_int[:] = psi_i[qgrid_ind[:,0],0] * psi_i[qgrid_ind[:,1],1] * psi_i[qgrid_ind[:,2],2] \
            * psi_j[qgrid_ind[:,0],0] * psi_j[qgrid_ind[:,1],1] * psi_j[qgrid_ind[:,2],2] \
            * poten_h2s_Tyuterev.poten(qcoords[:,0:4]) 

    pot_elem = np.sum(pot_int)
    return pot_elem

@jit
def hmat(bas_ind,qgrid_ind):
    """calculate full Hamiltonian Matrix"""
    Ngrid = np.size(qgrid_ind,axis=0)
    print("number of grid points = " + str(Ngrid))
    
    Nbas = np.size(bas_ind,axis=0)
    print("number of basis functions = " + str(Nbas))

    """construct the quadrature grid. For now it is direct product Gaussian grid"""
    x1, w1 = hermgauss(Nquad1D_herm)
    x3, w3 = leggauss(Nquad1D_leg)
    #print(type(w1))
    """print(x3,w3)
    plt.stem(x3,w3)
    plt.show()"""

    """ For later improved version: grid_dp = np.array(np.meshgrid(x3, x2, x1, indexing = 'ij')).T.reshape(-1,3)
                                    weights_dp = np.array(np.meshgrid(w3, w2, w1, indexing = 'ij')).T.reshape(-1,3)"""


    #initialize the KEO
    keo_jax.init(masses=masses, internal_to_cartesian=internal_to_cartesian)

    # compute strechting basis on the quadrature grid
    _, _, psi_r, dpsi_r = herm(0, ref_coords, Nquad1D_herm, Nquad1D_herm-1, [0.5, 10.0],
                                    poten_h2s_Tyuterev.poten, keo_jax.Gmat,
                                    verbose=False)
    # compute bending basis on the quadrature grid
    _, _, psi_theta, dpsi_theta= legcos(2, ref_coords, Nquad1D_leg, Nquad1D_herm-1, [0, np.pi],
                                poten_h2s_Tyuterev.poten, keo_jax.Gmat,
                                verbose=False)


    hmat = np.zeros((Nbas, Nbas), dtype = float)
    phi_i = np.zeros((Nquad1D_herm,3),dtype = float)
    phi_j = np.zeros((Nquad1D_herm,3),dtype = float)
    dphi_i = np.zeros((Nquad1D_herm,3),dtype = float)
    dphi_j = np.zeros((Nquad1D_herm,3),dtype = float)

    #generate G-matrix 
    G = np.zeros((Ngrid,9,9))
    start = time.time()
    for ipoint in range(np.size(qgrid_ind,axis=0)):
        qcoords = [x1[qgrid_ind[ipoint][0]],x1[qgrid_ind[ipoint][1]],x3[qgrid_ind[ipoint][2]]]
        g = keo_jax.Gmat(qcoords)
        g = np.asarray(g)
        G[ipoint,:,:] = g
        #G = np.concatenate((G,g),axis=0)
    end = time.time()
    #print(G)
    #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in G]))
    #print('\n')
    print("time for generation of G-matrix on full quadrature grid =  ", str(end-start))

    # use finite-differences (7-point) to compute frequency
    fdf_h=0.001
    fdf_steps = np.array([3*fdf_h, 2*fdf_h, fdf_h, 0.0, -fdf_h, -2*fdf_h, -3*fdf_h], dtype=np.float64)
    fdf_coefs = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], dtype=np.float64)
    fdf_denom = 180.0
    coords = np.array(np.broadcast_to(ref_coords, (len(fdf_steps),len(ref_coords))))

    G_ref = keo_jax.Gmat(ref_coords)[0, 0]
    coords[:,0] = [ref_coords[0] + st for st in fdf_steps]
    test_coords = np.asarray(ref_coords)
    V = poten_h2s_Tyuterev.poten(coords)
    freq = np.dot(V, fdf_coefs) / (fdf_denom * fdf_h * fdf_h)
    # scaling of x
    xmap = np.sqrt(np.sqrt( 2.0 * np.abs(freq) / np.abs(G_ref) ))
    r = x1 / xmap + ref_coords[0]


    """calculate the <psi_i | H | psi_j> integral """


    for i in range(Nbas):
        #start = time.time()
        ivec = [bas_ind[i][0],bas_ind[i][1],bas_ind[i][2]]

        phi_i[:,0] = np.sqrt(w1[:]) * psi_r[:,ivec[0]]
        phi_i[:,1] = np.sqrt(w1[:]) * psi_r[:,ivec[1]]
        phi_i[:,2] = np.sqrt(w3[:]) * psi_theta[:,ivec[2]]
        
        dphi_i[:,0] = np.sqrt(w1[:]) * dpsi_r[:,ivec[0]]
        dphi_i[:,1] = np.sqrt(w1[:]) * dpsi_r[:,ivec[1]]
        dphi_i[:,2] = np.sqrt(w3[:]) * dpsi_theta[:,ivec[2]]

        for j in range(i,Nbas):
            print("Element: ", str(i), str(j))
            jvec = [bas_ind[j][0],bas_ind[j][1],bas_ind[j][2]]

            phi_j[:,0] = np.sqrt(w1[:]) * psi_r[:,jvec[0]]
            phi_j[:,1] = np.sqrt(w1[:]) * psi_r[:,jvec[1]]
            phi_j[:,2] = np.sqrt(w3[:]) * psi_theta[:,jvec[2]]
            
            dphi_j[:,0] = np.sqrt(w1[:]) * dpsi_r[:,jvec[0]]
            dphi_j[:,1] = np.sqrt(w1[:]) * dpsi_r[:,jvec[1]]
            dphi_j[:,2] = np.sqrt(w3[:]) * dpsi_theta[:,jvec[2]]

            keo = matelem_keo(phi_i, dphi_i, phi_j, dphi_j, qgrid_ind, G)
            pot = matelem_pes(phi_i, phi_j, r ,x3, qgrid_ind)
            hmat[i,j] = keo + pot
            #end = time.time()
            #print("time per matrix element =  ", str(end-start))


   
    end = time.time()
    print("time for calculation of matrix elements =  ", str(end-start))
    #print('\n'.join([' '.join(["  %12.3f"%item for item in row]) for row in hmat]))
    eval, eigvec = np.linalg.eigh(hmat,UPLO='U')
    #print(eval-eval[0])
    return eval-eval[0]


if __name__=="__main__":

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]
    masses=[31.97207070, 1.00782505, 1.00782505]

    """generate mapping function"""
    b = 10 #basis set pruning parameters

    simpleMap = indexmap(b,'simple',3)
    #bas_ind = np.asarray(simpleMap.gen_map())
    bas_ind = simpleMap.gen_map()
    print(bas_ind)
    Nbas = np.size(bas_ind , axis =0)
    Nquad1D_herm = 20
    Nquad1D_leg = 20
    grid_type = 'dp'


    w_tol =  1e-15 #threshold value for keeping 3D quadrature product-weights

    """Tyuterev H2S energy levels"""
    energies_ref = [0.000000,
    1182.569532,
    2353.907164,
    2614.394631,
    2628.463126,
    3513.704879,
    3779.189344,
    3789.269895,
    4661.605594,
    4932.689271,
    4939.130231,
    5145.032233,
    5147.167000,
    5243.158607,
    5797.207386,
    6074.566919,
    6077.627575,
    6288.137389,
    6289.130997,
    6385.321103,
    6920.081237,
    7204.310047,
    7204.436792,
    7419.853766,
    7420.083864,
    7516.826950,
    7576.418099,
    7576.601014,
    7752.343405,
    7779.352384,
    8029.814410,
    8318.687462,
    8321.866003,
    8539.576768,
    8539.826215,
    8637.163913,
    8697.151138,
    8697.196560,
    8878.593110,
    8897.380662,
    9126.085650,
    9420.240371,
    9426.397111,
    9647.116010,
    9647.624407,
    9745.801753,
    9806.747682,
    9806.782892,
    9911.125222,
    9911.135434,
    9993.686588,
    10004.984409,
    10188.361597]

    """generate 3D quadrature grid (for now direct product, only indices)"""
    qgrid = gridmap(Nquad1D_herm, 'dp', 3, w_tol, Nquad1D_herm, Nquad1D_herm, Nquad1D_leg)
    qgrid_ind = np.asarray(qgrid.gen_map())
    Ngrid = np.size(qgrid_ind , axis =0)
    #print(qgrid_ind)

    init(Nbas, Ngrid, Nquad1D_herm, Nquad1D_leg, ref_coords, masses) 
    energies_calc = hmat(bas_ind,qgrid_ind)

    for i in range(len(energies_ref)):
        print(" %4i"%i + "  %16.8f"%energies_ref[i] + "  %16.8f"%energies_calc[i] + "  %16.8f"%(energies_ref[i]-energies_calc[i]))

        """  A1         1      0.000000   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   0   0 )      1.00 (   0   0   0   0 ) (    1    1 )
    A1        2   1182.569532   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   0   1 )      1.00 (   0   0   1   0 ) (    1    2 )
    A1        3   2353.907164   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   0   2 )      0.99 (   0   0   2   0 ) (    1    3 )
    A1        4   2614.394631   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   1   0 )      1.00 (   1   0   0   0 ) (    2    1 )
    A1        5   3513.704879   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   0   3 )      0.98 (   0   0   3   0 ) (    1    4 )
    A1        6   3779.189344   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   1   1 )      0.98 (   1   0   1   0 ) (    2    2 )
    A1        7   4661.605594   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   0   4 )      0.97 (   0   0   4   0 ) (    1    5 )
    A1        8   4932.689271   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   1   2 )      0.96 (   1   0   2   0 ) (    2    3 )
    A1        9   5145.032233   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   2   0 )      0.99 (   2   0   0   0 ) (    4    1 )
    A1       10   5243.158607   ( A1 ;  0  0  0 ) ( A1  A1 ;   1   1   0 )      1.00 (   2   0   0   0 ) (    6    1 )
    A1       11   5797.207386   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   0   5 )      0.95 (   0   0   5   0 ) (    1    6 )
    A1       12   6074.566919   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   1   3 )      0.93 (   1   0   3   0 ) (    2    4 )
    A1       13   6288.137389   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   2   1 )      0.97 (   2   0   1   0 ) (    4    2 )
    A1       14   6385.321103   ( A1 ;  0  0  0 ) ( A1  A1 ;   1   1   1 )      0.98 (   2   0   1   0 ) (    6    2 )
    A1       15   6920.081237   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   0   6 )      0.93 (   0   0   6   0 ) (    1    7 )
    A1       16   7204.436792   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   1   4 )      0.89 (   1   0   4   0 ) (    2    5 )
    A1       17   7419.853766   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   2   2 )      0.94 (   2   0   2   0 ) (    4    3 )
    A1       18   7516.826950   ( A1 ;  0  0  0 ) ( A1  A1 ;   1   1   2 )      0.95 (   2   0   2   0 ) (    6    3 )
    A1       19   7576.418099   ( A1 ;  0  0  0 ) ( A1  A1 ;   3   0   0 )      0.99 (   3   0   0   0 ) (    7    1 )
    A1       20   7752.343405   ( A1 ;  0  0  0 ) ( A1  A1 ;   2   1   0 )      0.99 (   3   0   0   0 ) (    9    1 )
    A1       21   8029.814410   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   0   7 )      0.91 (   0   0   7   0 ) (    1    8 )
    A1       22   8321.866003   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   1   5 )      0.84 (   1   0   5   0 ) (    2    6 )
    A1       23   8539.826215   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   2   3 )      0.89 (   2   0   3   0 ) (    4    4 )
    A1       24   8637.163913   ( A1 ;  0  0  0 ) ( A1  A1 ;   1   1   3 )      0.91 (   2   0   3   0 ) (    6    4 )
    A1       25   8697.151138   ( A1 ;  0  0  0 ) ( A1  A1 ;   3   0   1 )      0.97 (   3   0   1   0 ) (    7    2 )
    A1       26   8878.593110   ( A1 ;  0  0  0 ) ( A1  A1 ;   2   1   1 )      0.97 (   3   0   1   0 ) (    9    2 )
    A1       27   9126.085650   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   0   8 )      0.88 (   0   0   8   0 ) (    1    9 )
    A1       28   9426.397111   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   1   6 )      0.79 (   1   0   6   0 ) (    2    7 )
    A1       29   9647.624407   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   2   4 )      0.83 (   2   0   4   0 ) (    4    5 )
    A1       30   9745.801753   ( A1 ;  0  0  0 ) ( A1  A1 ;   1   1   4 )      0.87 (   2   0   4   0 ) (    6    5 )
    A1       31   9806.782892   ( A1 ;  0  0  0 ) ( A1  A1 ;   3   0   2 )      0.92 (   3   0   2   0 ) (    7    3 )
    A1       32   9911.135434   ( A1 ;  0  0  0 ) ( A1  A1 ;   0   4   0 )      0.99 (   4   0   0   0 ) (   12    1 )
    A1       33   9993.686588   ( A1 ;  0  0  0 ) ( A1  A1 ;   2   1   2 )      0.92 (   3   0   2   0 ) (    9    3 )
    A1       34  10188.361597   ( A1 ;  0  0  0 ) ( A1  A1 ;   1   3   0 )      0.99 (   4   0   0   0 ) (   13    1 )
    B2        1   2628.463126   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   0   0 )      1.00 (   1   0   0   0 ) (    3    1 )
    B2        2   3789.269895   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   0   1 )      1.00 (   1   0   1   0 ) (    3    2 )
    B2        3   4939.130231   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   0   2 )      0.99 (   1   0   2   0 ) (    3    3 )
    B2        4   5147.167000   ( A1 ;  0  0  0 ) ( B2  A1 ;   2   0   0 )      1.00 (   2   0   0   0 ) (    5    1 )
    B2        5   6077.627575   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   0   3 )      0.97 (   1   0   3   0 ) (    3    4 )
    B2        6   6289.130997   ( A1 ;  0  0  0 ) ( B2  A1 ;   2   0   1 )      0.98 (   2   0   1   0 ) (    5    2 )
    B2        7   7204.310047   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   0   4 )      0.96 (   1   0   4   0 ) (    3    5 )
    B2        8   7420.083864   ( A1 ;  0  0  0 ) ( B2  A1 ;   2   0   2 )      0.96 (   2   0   2   0 ) (    5    3 )
    B2        9   7576.601014   ( A1 ;  0  0  0 ) ( B2  A1 ;   0   3   0 )      0.99 (   3   0   0   0 ) (    8    1 )
    B2       10   7779.352384   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   2   0 )      1.00 (   3   0   0   0 ) (   10    1 )
    B2       11   8318.687462   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   0   5 )      0.94 (   1   0   5   0 ) (    3    6 )
    B2       12   8539.576768   ( A1 ;  0  0  0 ) ( B2  A1 ;   2   0   3 )      0.92 (   2   0   3   0 ) (    5    4 )
    B2       13   8697.196560   ( A1 ;  0  0  0 ) ( B2  A1 ;   0   3   1 )      0.97 (   3   0   1   0 ) (    8    2 )
    B2       14   8897.380662   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   2   1 )      0.98 (   3   0   1   0 ) (   10    2 )
    B2       15   9420.240371   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   0   6 )      0.91 (   1   0   6   0 ) (    3    7 )
    B2       16   9647.116010   ( A1 ;  0  0  0 ) ( B2  A1 ;   2   0   4 )      0.87 (   2   0   4   0 ) (    5    5 )
    B2       17   9806.747682   ( A1 ;  0  0  0 ) ( B2  A1 ;   0   3   2 )      0.93 (   3   0   2   0 ) (    8    3 )
    B2       18   9911.125222   ( A1 ;  0  0  0 ) ( B2  A1 ;   4   0   0 )      0.99 (   4   0   0   0 ) (   11    1 )
    B2       19  10004.984409   ( A1 ;  0  0  0 ) ( B2  A1 ;   1   2   2 )      0.96 (   3   0   2   0 ) (   10    3 )"""