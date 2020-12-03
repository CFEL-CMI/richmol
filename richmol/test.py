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



def eval_k(i, j, ind):
    """
    i: a vector of shape (3, ) containing i1,i2,i3
    J: a vector of shape (3, ) containing j1,j2,j3
    coords: (3, )
    """
    keo_jax.init(masses=[31.97207070, 1.00782505, 1.00782505], \
                 internal_to_cartesian=internal_to_cartesian)

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]
    # compute strechting basis

    xherm, _, psi_r, dpsi_r = herm(0, ref_coords, 100, 30, [0.5, 10.0],
                                    poten_h2s_Tyuterev.poten, keo_jax.Gmat,
                                    verbose=False)
    xleg, _, psi_theta, dpsi_theta= legcos(2, ref_coords, 100, 30, [0, np.pi],
                                  poten_h2s_Tyuterev.poten, keo_jax.Gmat,
                                  verbose=False)
    #ind = 0 # index of the grid point; need to find it
    G = keo_jax.Gmat([xherm[ind],xherm[ind],xleg[ind]])
    evals = np.zeros((1,))
    evals = dpsi_r[ind, j[0]]*psi_r[ind, j[1]]*psi_theta[ind, j[2]]*G[0,0]*dpsi_r[ind, i[0]]*psi_r[ind, i[1]]*psi_theta[ind, i[2]] \
            + dpsi_r[ind, j[0]]*psi_r[ind, j[1]]*psi_theta[ind, j[2]]*G[0,1]*psi_r[ind, i[0]]*dpsi_r[ind, i[1]]*psi_theta[ind, i[2]] \
            + dpsi_r[ind, j[0]]*psi_r[ind, j[1]]*psi_theta[ind, j[2]]*G[0,2]*psi_r[ind, i[0]]*psi_r[ind, i[1]]*dpsi_theta[ind, i[2]] \
            + psi_r[ind, j[0]]*dpsi_r[ind, j[1]]*psi_theta[ind, j[2]]*G[1,0]*dpsi_r[ind, i[0]]*psi_r[ind, i[1]]*psi_theta[ind, i[2]] \
            + psi_r[ind, j[0]]*dpsi_r[ind, j[1]]*psi_theta[ind, j[2]]*G[1,1]*psi_r[ind, i[0]]*dpsi_r[ind, i[1]]*psi_theta[ind, i[2]] \
            + psi_r[ind, j[0]]*dpsi_r[ind, j[1]]*psi_theta[ind, j[2]]*G[1,2]*psi_r[ind, i[0]]*psi_r[ind, i[1]]*dpsi_theta[ind, i[2]] \
            + psi_r[ind, j[0]]*psi_r[ind, j[1]]*dpsi_theta[ind, j[2]]*G[2,0]*dpsi_r[ind, i[0]]*psi_r[ind, i[1]]*psi_theta[ind, i[2]] \
            + psi_r[ind, j[0]]*psi_r[ind, j[1]]*dpsi_theta[ind, j[2]]*G[2,1]*psi_r[ind, i[0]]*dpsi_r[ind, i[1]]*psi_theta[ind, i[2]] \
            + psi_r[ind, j[0]]*psi_r[ind, j[1]]*dpsi_theta[ind, j[2]]*G[2,2]*psi_r[ind, i[0]]*psi_r[ind, i[1]]*dpsi_theta[ind, i[2]] \
    

    
    return evals



@com
@bisector('zxy')
def internal_to_cartesian(coords):
    r1, r2, alpha = coords
    xyz = np.array([[0.0, 0.0, 0.0], \
                    [ r1 * np.sin(alpha/2), 0.0, r1 * np.cos(alpha/2)], \
                    [-r2 * np.sin(alpha/2), 0.0, r2 * np.cos(alpha/2)]], \
                    dtype=np.float64)
    return xyz
if __name__ == "__main__":
    """
    keo_jax.init(masses=[31.97207070, 1.00782505, 1.00782505], \
                 internal_to_cartesian=internal_to_cartesian)

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]
    r, her_enr_str, psi, dpsi = herm(0, ref_coords, 100, 30, [0.5, 10.0], \
                                    poten_h2s_Tyuterev.poten, keo_jax.Gmat, \
                                    verbose=True)
    """
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]
    i = [1,2,3]
    j = [1,2,3]

    eval = eval_k(i,j, 0)
    print(eval)