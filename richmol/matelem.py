import numpy as np
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
import jax.numpy as np
import poten_h2s_Tyuterev
from prim import numerov, legcos, herm, laguerre
import sys

#import scipy.special.eval_genlaguerre
#import scipy.special.roots_genlaguerre as laggauss
singular_tol = 1e-10 # tolerance for considering matrix singular
symmetric_tol = 1e-10 # tolerance for considering matrix symmetric
from prim import herm, legcos
import matplotlib.pyplot as plt



class matelem:
    def __init__(self, Nbas, Ngrid, Nquad1D_herm, Nquad1D_leg, ref_coords, masses):

        self.w_tol = 1e-30 #threshold value for keeping 3D quadrature product-weights
        self.Nbas = Nbas #basis set size
        self.Ngrid = Ngrid
        self.Nquad1D_herm =  Nquad1D_herm  #number of Gauss-Hermite quadrature points in 1D
        self.Nquad1D_leg =  Nquad1D_leg  #number of Gauss-Legendre quadrature points in 1D
        self.ref_coords = ref_coords #reference coordinates
        self.masses = masses #nuclear masses

    @com
    @bisector('zxy')
    def internal_to_cartesian(self,coords):
        r1, r2, alpha = coords
        xyz = np.array([[0.0, 0.0, 0.0], \
                        [ r1 * np.sin(alpha/2), 0.0, r1 * np.cos(alpha/2)], \
                        [-r2 * np.sin(alpha/2), 0.0, r2 * np.cos(alpha/2)]], \
                        dtype=np.float64)
        return xyz #this function can be called from XY2 or molecule module. It does not belong here. 


    def matelem_keo(self, ivec, jvec, phi_i, dphi_i, phi_j, dphi_j, G , qgrid_ind):
        """
        This routine calculates the matrix element of the kinetic energy operator.

        Input: 
        ivec: a vector of shape (3, ) containing i1,i2,i3 (left indices)
        jvec: a vector of shape (3, ) containing j1,j2,j3 (right indices)
        phi_i: array (Nquad, 3) of values of the three basis functions (phi_r_i1,phi_r_i2,phi_theta_i3), whose indices correspond to multiindex ivec = (i1,i2,i3). 
                The functions are evaluated on respective quadrature grids.
        dphi_i: array (Nquad, 3) of values of the three basis functions derivatives (dphi_r_i1,dphi_r_i2,dphi_theta_i3), whose indices correspond to multiindex ivec = (i1,i2,i3). 
                The derivatives are evaluated on respective quadrature grids.
        weights: array (Nquad,) of product quadrature weights weigths[k] = w1[k1] * w2[k2] * w3[k3]
        G: array ((3,3,Nquad, Nquad?)) of values of G-matrix elements on the quadrature grid

        Returns: 
        keo_elem: matrix element of the KEO
        """

        keo_elem = np.zeros((1,))
        f_int = np.zeros(np.size(qgrid_ind,axis=0))

        f_int[:] = dphi_i[qgrid_ind[:,0], 0 ] * phi_i[qgrid_ind[:,1], 1 ] * phi_i[qgrid_ind[:,2], 2 ] * G[0,0][qgrid_ind[:,0],qgrid_ind[:,1],qgrid_ind[:,2]] * dphi_j[qgrid_ind[:,0], 0 ] * phi_j[qgrid_ind[:,1], 1] * phi_j[qgrid_ind[:,2], 2] 


        """  + dpsi_r[k[0], j[0]]*psi_r[k[1], j[1]]*psi_theta[k[2], j[2]]*G[0,1]*psi_r[k[0], i[0]]*dpsi_r[k[1], i[1]]*psi_theta[k[2], i[2]] \
                + dpsi_r[k[0], j[0]]*psi_r[k[1], j[1]]*psi_theta[k[2], j[2]]*G[0,2]*psi_r[k[0], i[0]]*psi_r[k[1], i[1]]*dpsi_theta[k[2], i[2]] \
                + psi_r[k[0], j[0]]*dpsi_r[k[1], j[1]]*psi_theta[k[2], j[2]]*G[1,0]*dpsi_r[k[0], i[0]]*psi_r[k[1], i[1]]*psi_theta[k[2], i[2]] \
                + psi_r[k[0], j[0]]*dpsi_r[k[1], j[1]]*psi_theta[k[2], j[2]]*G[1,1]*psi_r[k[0], i[0]]*dpsi_r[k[1], i[1]]*psi_theta[k[2], i[2]] \
                + psi_r[k[0], j[0]]*dpsi_r[k[1], j[1]]*psi_theta[k[2], j[2]]*G[1,2]*psi_r[k[0], i[0]]*psi_r[k[1], i[1]]*dpsi_theta[k[2], i[2]] \
                + psi_r[k[0], j[0]]*psi_r[k[1], j[1]]*dpsi_theta[k[2], j[2]]*G[2,0]*dpsi_r[k[0], i[0]]*psi_r[k[1], i[1]]*psi_theta[k[2], i[2]] \
                + psi_r[k[0], j[0]]*psi_r[k[1], j[1]]*dpsi_theta[k[2], j[2]]*G[2,1]*psi_r[k[0], i[0]]*dpsi_r[k[1], i[1]]*psi_theta[k[2], i[2]] \
                + psi_r[k[0], j[0]]*psi_r[k[1], j[1]]*dpsi_theta[k[2], j[2]]*G[2,2]*psi_r[k[0], i[0]]*psi_r[k[1], i[1]]*dpsi_theta[k[2], i[2]] \ """

        
        keo_elem = np.sum(f_int)


        return keo_elem 

    def f_temp(self,ivec,jvec,coords):
        """temporary test function for 3D integration
        ivec: a vector of shape (3, ) containing i1,i2,i3
        jvec : a vector of shape (3, ) containing j1,j2,j3
        coords: (3, ): r1,r2,theta"""
        return coords[0]**2 * coords[1]**2 * coords[2]**2 * np.exp(-coords[0]**2) * np.exp(-coords[1]**2)

    def hmat(self,bas_ind,qgrid_ind):
        """calculate full Hamiltonian Matrix"""
        
        print("number of basis functions = " + str(self.Nbas))

        """construct the quadrature grid. For now it is direct product Gaussian grid"""
        x1, w1 = hermgauss(self.Nquad1D_herm)
        x2 = x1
        w2 = w1
        x3, w3 = leggauss(self.Nquad1D_leg)

        """print(x3,w3)
        plt.stem(x3,w3)
        plt.show()"""

        """ For later improved version: grid_dp = np.array(np.meshgrid(x3, x2, x1, indexing = 'ij')).T.reshape(-1,3) 
                                        weights_dp = np.array(np.meshgrid(w3, w2, w1, indexing = 'ij')).T.reshape(-1,3)"""


        #initialize the KEO
        keo_jax.init(masses=self.masses, internal_to_cartesian=self.internal_to_cartesian)
        for ipoint in range(Ngrid):
            icoords = [qgrid_ind[ipoint][0],qgrid_ind[ipoint][1],qgrid_ind[ipoint][2]]
            G[icoords,:,:] = keo_jax.Gmat(icoords) #need to add here full 
        print(np.shape(G))

        # compute strechting basis on the quadrature grid
        _, _, psi_r, dpsi_r = herm(0, ref_coords, self.Nquad1D_herm, self.Nquad1D_herm-1, [0.5, 10.0],
                                        poten_h2s_Tyuterev.poten, keo_jax.Gmat,
                                        verbose=False)
        # compute bending basis on the quadrature grid
        _, _, psi_theta, dpsi_theta= legcos(2, ref_coords, self.Nquad1D_leg, self.Nquad1D_herm-1, [0, np.pi],
                                    poten_h2s_Tyuterev.poten, keo_jax.Gmat,
                                    verbose=False)


        #grid_dp = np.array(np.meshgrid(x3, x2, x1, indexing = 'ij')).T.reshape(-1,3) 

        hmat = np.zeros((self.Nbas, self.Nbas), dtype = float)

        """calculate the <psi_i | H | psi_j> integral """
        for i in range(self.Nbas):

            ivec = [bas_ind[i][0],bas_ind[i][1],bas_ind[i][2]]
            phi_i = [np.sqrt(w1[:]) * psi_r[:,ivec[0]], np.sqrt(w2[:]) * psi_r[:,ivec[1]], np.sqrt(w3[:]) * psi_theta[:,ivec[2]]]
            dphi_i = [np.sqrt(w1[:]) * dpsi_r[:,ivec[0]],np.sqrt(w2[:]) * dpsi_r[:,ivec[1]],np.sqrt(w3[:]) * dpsi_theta[:,ivec[2]]]

            for j in range(self.Nbas):
                
                jvec = [bas_ind[j][0],bas_ind[j][1],bas_ind[j][2]]
                phi_j = [np.sqrt(w1[:]) * psi_r[:,jvec[0]],np.sqrt(w2[:]) * psi_r[:,jvec[1]],np.sqrt(w3[:]) * psi_theta[:,jvec[2]]]
                dphi_j = [np.sqrt(w1[:]) * dpsi_r[:,jvec[0]],np.sqrt(w2[:]) * dpsi_r[:,jvec[1]],np.sqrt(w3[:]) * dpsi_theta[:,jvec[2]]]

                hmat[i,j] = self.matelem_keo(ivec, jvec, phi_i, dphi_i, phi_j, dphi_j, G, qgrid_ind)

        print(hmat)
        eval, eigvec = np.linalg.eigh(hmat)
        print(eval)
        return hmat


if __name__=="__main__":

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]
    masses=[31.97207070, 1.00782505, 1.00782505]    

    """generate mapping function"""
    b = 2 #basis set pruning parameters

    simpleMap = indexmap(b,'simple',3)
    bas_ind = np.asarray(simpleMap.gen_map())
    Nbas = np.size(bas_ind , axis =0) 
    #print(bas_ind)

    Nquad1D_herm = 5
    Nquad1D_leg = 5
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


    ham = matelem( Nbas, Ngrid, Nquad1D_herm, Nquad1D_leg, ref_coords, masses) #hamiltonian class
    ham.hmat(bas_ind,qgrid_ind)