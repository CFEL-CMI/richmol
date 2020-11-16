from molecule import Molecule
import autograd.numpy as np
import poten_h2o_Polyansky
import poten_h2s_Tyuterev
import sys
import matplotlib.pyplot as plt
#import poten_from_tf
import scipy.misc as mc
from autograd import elementwise_grad
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

    h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)
    coords = np.array([[1,1,0.5], [1,0.9,0.5]],dtype=np.float64)
    npt = 1
    G = h2s.G(coords)
    # test pseudo-potential
    delta = h2s.PP(coords)
    print(delta)
