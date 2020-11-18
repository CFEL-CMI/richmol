from molecule import Molecule
import autograd.numpy as np
import poten_h2o_Polyansky
import poten_h2s_Tyuterev
import sys
import matplotlib.pyplot as plt
#import poten_from_tf
import scipy.misc as mc
from autograd import elementwise_grad, jacobian
from autograd import grad

class XY2_ralpha(Molecule):
    """ XY2-type molecule (e.g., H2O, H2S) / valence-bond internal cooridnates """

    def __init__(self, *args, **kwargs):
        Molecule.__init__(self, *args, **kwargs)


    @Molecule.autograd
    @Molecule.com
    @Molecule.bisector('zyx')
    def internal_to_cartesian(self, coords, **kwargs):
        xyz = np.array([[[ 0.0, 0.0, 0.0], \
                        [ r1 * np.sin(alpha/2), 0.0, r1 * np.cos(alpha/2)], \
                        [ -r2 * np.sin(alpha/2), 0.0, r2 * np.cos(alpha/2)]] for r1,r2,alpha in coords], \
                        dtype=np.float64)
        return xyz



if __name__=="__main__":

    #h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)
    h2s = XY2_ralpha(masses=[31.9721, 1.00783, 1.00783], poten=poten_h2s_Tyuterev.poten)
    coords = np.array([[1.3359,1.3359,1.61034]],dtype=np.float64)
    npt = 1
    #G = h2s.G(coords)
    # test pseudo-potential
    delta = h2s.PP(coords)
    print(delta)
    sys.exit()
    def A(x,y):
        return np.array((x**1)*(y**2))
    x = np.array([1,2,3,4])
    y = np.array([2])
    print(A(x,y))
    #grada = elementwise_grad(A,0)
    #dA_dX = grada(x,y)

    def B(x,y):
        grada = elementwise_grad(A,0)
        dA_dX = grada(x,y)
        return dA_dX
    print(B(x,y))
    #sys.exit()
    #grada2 = jacobian(B,1)
    grada2 = elementwise_grad(B,1)

    dA_dX_dY = grada2(x,y)
    print(dA_dX_dY)
