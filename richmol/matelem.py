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
import time
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


def f_temp(ivec,jvec,coords):
    """temporary test function for 3D integration
    ivec: a vector of shape (3, ) containing i1,i2,i3
    jvec : a vector of shape (3, ) containing j1,j2,j3
    coords: (3, ): r1,r2,theta"""
    return coords[0]**2 * coords[1]**2 * coords[2]**2 * np.exp(-coords[0]**2) * np.exp(-coords[1]**2)

def matelem_keo( ivec, jvec, psi_i, dpsi_i, psi_j, dpsi_j, x1,x2,x3,  qgrid_ind):
    """
    This routine calculates the matrix element of the kinetic energy operator.

    Input:
    ivec: a vector of shape (3, ) containing i1,i2,i3 (left indices)
    jvec: a vector of shape (3, ) containing j1,j2,j3 (right indices)
    psi_ivec: array (Nquad, 3) of values of the three basis functions (phi_r_i1,phi_r_i2,phi_theta_i3), whose indices correspond to multiindex ivec = (i1,i2,i3).
            The functions are evaluated on respective quadrature grids.
    dpsi_ivec: array (Nquad, 3) of values of the three basis functions derivatives (dphi_r_i1,dphi_r_i2,dphi_theta_i3), whose indices correspond to multiindex ivec = (i1,i2,i3).
            The derivatives are evaluated on respective quadrature grids.
    weights: array (Nquad,) of product quadrature weights weigths[k] = w1[k1] * w2[k2] * w3[k3]
    G: array ((3,3,Nquad, Nquad?)) of values of G-matrix elements on the quadrature grid

    Returns:
    keo_elem: matrix element of the KEO
    """

    keo_elem = np.zeros((1,))
    f_int = np.zeros(np.size(qgrid_ind,axis=0))


    #print(np.shape(keo_jax.Gmat(icoords))) #need to add here full
    keo_jax.init(masses=masses, internal_to_cartesian=internal_to_cartesian)
    for ipoint in range(np.size(qgrid_ind,axis=0)):
        qcoords = [x1[qgrid_ind[ipoint][0]],x2[qgrid_ind[ipoint][1]],x3[qgrid_ind[ipoint][2]]]
        start = time.time()
        G = keo_jax.Gmat(qcoords)
        G = np.asarray(G)
        print("type of G")
        print(type(G))
        end = time.time()
        #print("time for keo_jax.Gmat(qcoords) =  ", str(end-start))
        #print(' '.join(["  %15.8f"%item for item in qcoords]))
        #print(dpsi_i[qgrid_ind[ipoint,0], 0 ] * psi_i[qgrid_ind[ipoint,1], 1 ] * psi_i[qgrid_ind[ipoint,2], 2 ] * G[0][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ] * psi_j[qgrid_ind[ipoint,1], 1] * psi_j[qgrid_ind[ipoint,2], 2])
        #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in G]))

        """start = time.time()
            jax.ops.index_add(f_int,ipoint, dpsi_i[qgrid_ind[ipoint,0], 0 ] * psi_i[qgrid_ind[ipoint,1], 1 ] * psi_i[qgrid_ind[ipoint,2], 2 ] * G[0][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ] * psi_j[qgrid_ind[ipoint,1], 1] * psi_j[qgrid_ind[ipoint,2], 2] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[0][1]* psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[0][2] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[1][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[1][1] * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ]* G[1][2] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ]* G[2][0]* dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ]* G[2][1] * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ]* G[2][2] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ])

        end = time.time()
        print("time for jax.ops.index_add( =  ", str(end-start))"""


        start = time.time()
        keo_elem += dpsi_i[qgrid_ind[ipoint,0], 0 ] * psi_i[qgrid_ind[ipoint,1], 1 ] * psi_i[qgrid_ind[ipoint,2], 2 ] * G[0][0]  * dpsi_j[qgrid_ind[ipoint,0], 0 ] * psi_j[qgrid_ind[ipoint,1], 1] * psi_j[qgrid_ind[ipoint,2], 2] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[0][1]  * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + dpsi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[0][2] * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[1][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[1][1] * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*dpsi_i[qgrid_ind[ipoint,1], 1]*psi_i[qgrid_ind[ipoint,2], 2 ] * G[1][2]  * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ] * G[2][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ] * G[2][1]  * psi_j[qgrid_ind[ipoint,0], 0 ]*dpsi_j[qgrid_ind[ipoint,1], 1]*psi_j[qgrid_ind[ipoint,2], 2 ] \
            + psi_i[qgrid_ind[ipoint,0], 0 ]*psi_i[qgrid_ind[ipoint,1], 1]*dpsi_i[qgrid_ind[ipoint,2], 2 ] * G[2][2]  * psi_j[qgrid_ind[ipoint,0], 0 ]*psi_j[qgrid_ind[ipoint,1], 1]*dpsi_j[qgrid_ind[ipoint,2], 2 ]
        end = time.time()
        print("type of keo_elem")
        print(type(keo_elem))
        print("time for jax.ops.index_add =  ", str(end-start))

    #jax.ops.index_update(f,ipoint,f_int)
    #print(type(f))
    #keo_elem = np.sum(f)

    print("we are returning value")

    return keo_elem

def matelem_pes(ivec, jvec, psi_i, psi_j, x1, x2, x3, qgrid_ind):
    # concatenate points
    x = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1),x3.reshape(-1,1)), axis=1) #is ordering of the grid the same as in psi_i_product ?
    #print(np.shape(x))
    #print(x)

    qcoords = np.concatenate(np.concatenate(x1[qgrid_ind[:][0]],x2[qgrid_ind[:][1]]),x3[qgrid_ind[:][2]])
    print(qcoords)
    PES = poten_h2s_Tyuterev.poten(qcoords)
    #print(PES)
    psi_i_product = np.multiply(np.multiply(psi_i[:,0], psi_i[:,1]), psi_i[:,2])
    psi_j_product = np.multiply(np.multiply(psi_j[:,0], psi_j[:,1]), psi_j[:,2])

    return np.sum(np.multiply(np.multiply(psi_i_product, psi_j_product), PES))

def hmat(bas_ind,qgrid_ind):
    """calculate full Hamiltonian Matrix"""
    Ngrid = np.size(qgrid_ind,axis=0)
    print("number of grid points = " + str(Ngrid))
    
    Nbas = np.size(bas_ind,axis=0)
    print("number of basis functions = " + str(Nbas))

    """construct the quadrature grid. For now it is direct product Gaussian grid"""
    x1, w1 = hermgauss(Nquad1D_herm)
    x2 = x1
    w2 = w1
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


    #grid_dp = np.array(np.meshgrid(x3, x2, x1, indexing = 'ij')).T.reshape(-1,3)
    #print(type(psi_r))

    hmat = np.zeros((Nbas, Nbas), dtype = float)
    phi_i = np.zeros((Ngrid,3),dtype = float)
    phi_j = np.zeros((Ngrid,3),dtype = float)
    dphi_i = np.zeros((Ngrid,3),dtype = float)
    dphi_j = np.zeros((Ngrid,3),dtype = float)

    #generate G-matrix 

    G = np.zeros((9,9))

    keo_jax.init(masses=masses, internal_to_cartesian=internal_to_cartesian)
    start = time.time()
    for ipoint in range(np.size(qgrid_ind,axis=0)):
        qcoords = [x1[qgrid_ind[ipoint][0]],x2[qgrid_ind[ipoint][1]],x3[qgrid_ind[ipoint][2]]]
        g = keo_jax.Gmat(qcoords)
        g = np.asarray(g) 
        G = np.concatenate((G,g),axis=0)
    
    end = time.time()
    #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in G]))
    #print('\n')
    print("time for generation of G-matrix on full quadrature grid =  ", str(end-start))
    #print(' '.join(["  %15.8f"%item for item in qcoords]))


    #print(dpsi_i[qgrid_ind[ipoint,0], 0 ] * psi_i[qgrid_ind[ipoint,1], 1 ] * psi_i[qgrid_ind[ipoint,2], 2 ] * G[0][0] * dpsi_j[qgrid_ind[ipoint,0], 0 ] * psi_j[qgrid_ind[ipoint,1], 1] * psi_j[qgrid_ind[ipoint,2], 2])
    #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in G]))
    exit()
    """calculate the <psi_i | H | psi_j> integral """
    for i in range(Nbas):

        ivec = [bas_ind[i][0],bas_ind[i][1],bas_ind[i][2]]
        #print(ivec)

        phi_i = psi_r[:,ivec]
        phi_i[:,0] *= np.sqrt(w1[:]) #compact it outside the loop
        phi_i[:,1] *= np.sqrt(w2[:])
        phi_i[:,2] *= np.sqrt(w3[:])
        dphi_i = dpsi_r[:,ivec]
        dphi_i[:,0] *= np.sqrt(w1[:])
        dphi_i[:,1] *= np.sqrt(w2[:])
        dphi_i[:,2] *= np.sqrt(w3[:])

        for j in range(Nbas):
            print("Element: ", str(i), str(j))
            jvec = [bas_ind[j][0],bas_ind[j][1],bas_ind[j][2]]
            phi_j = psi_r[:,jvec]
            phi_j[:,0] *= np.sqrt(w1[:])
            phi_j[:,1] *= np.sqrt(w2[:])
            phi_j[:,2] *= np.sqrt(w3[:])
            dphi_j = dpsi_r[:,jvec]
            dphi_j[:,0] *= np.sqrt(w1[:])
            dphi_j[:,1] *= np.sqrt(w2[:])
            dphi_j[:,2] *= np.sqrt(w3[:])
            f = matelem_keo(ivec, jvec, phi_i, dphi_i, phi_j, dphi_j, x1,x2,x3, qgrid_ind)
            #a = matelem_pes(ivec, jvec, phi_i, phi_j, x1,x2,x3, qgrid_ind)
            #f = matelem_keo(ivec, jvec, phi_i, dphi_i, phi_j, dphi_j, x1,x2,x3, qgrid_ind)[0]
            #print(a)
            #jax.ops.index_update(hmat,(i,j),a) #assignment issue
            #hmat[i,j] = a
            #print(hmat[i,j])
            #print(a)


    print(hmat)
    #eval, eigvec = np.linalg.eigh(hmat)
    #print(eval)
    return hmat


if __name__=="__main__":

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]
    masses=[31.97207070, 1.00782505, 1.00782505]

    """generate mapping function"""
    b = 2 #basis set pruning parameters

    simpleMap = indexmap(b,'simple',3)
    #bas_ind = np.asarray(simpleMap.gen_map())
    bas_ind = simpleMap.gen_map()
    print(type(simpleMap.gen_map()))

    Nbas = np.size(bas_ind , axis =0)
    #print(bas_ind)
    print(type(bas_ind))
    Nquad1D_herm = 8
    Nquad1D_leg = 8
    grid_type = 'dp'

    """Grid types:
                    'dp': direct product Nquad1D_herm x Nquad1D_herm x Nquad1D_leg
                    'ndp_weights': non-direct product with pruning based on product weights (w_tol)
    """

    w_tol =  1e-15 #threshold value for keeping 3D quadrature product-weights

    """generate 3D quadrature grid (for now direct product, only indices)"""
    qgrid = gridmap(Nquad1D_herm, 'dp', 3, w_tol, Nquad1D_herm, Nquad1D_herm, Nquad1D_leg)
    qgrid_ind = np.asarray(qgrid.gen_map())
    Ngrid = np.size(qgrid_ind , axis =0)
    #print(qgrid_ind)

    init(Nbas, Ngrid, Nquad1D_herm, Nquad1D_leg, ref_coords, masses) #hamiltonian class
    hmat(bas_ind,qgrid_ind)
