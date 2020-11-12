from molecule import Molecule
import autograd.numpy as np
import poten_h2o_Polyansky
import poten_h2s_Tyuterev
import numpy as np
import sys
import matplotlib.pyplot as plt
#import poten_from_tf
import scipy.misc as mc

class XY2_ralpha(Molecule):
    """ XY2-type molecule (e.g., H2O, H2S) / valence-bond internal cooridnates """

    def __init__(self, *args, **kwargs):
        Molecule.__init__(self, *args, **kwargs)


    @Molecule.autograd
    @Molecule.com
    #@Molecule.bisector('zyx')
    def internal_to_cartesian(self, coords, **kwargs):
        xyz = np.array([[[ 0.0, 0.0, 0.0], \
                        [ r1 * np.sin(alpha/2), 0.0, r1 * np.cos(alpha/2)], \
                        [ -r2 * np.sin(alpha/2), 0.0, r2 * np.cos(alpha/2)]] for r1,r2,alpha in coords], \
                        dtype=np.float64)
        return xyz



if __name__=="__main__":

    h2o = XY2_ralpha(masses=[12,1,1], poten=poten_h2o_Polyansky.poten)
    #h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)
    #coords1 = np.array([[1,1,0.5],[1,1,0.5],[1,1,0.5],[1,1,0.5],[1,1,0.5],[1,1,0.5]],dtype=np.float64)
    npt = 100
    r1 = np.linspace(0.6, 1.8, npt)
    r2 = np.repeat(0.9586, npt)
    theta = np.repeat(np.deg2rad(104.48), npt)
    grid = np.stack((r1,r2,theta), axis=1)
    coords1 = np.array([[0.9586,0.9586,np.deg2rad(104.48)]],dtype=np.float64)

    G = h2o.G(coords1) #resulting in an error
    V = h2o.V(grid)
    grad = mc.derivative(h2o.V, grid, dx=0.1, n=1, order=5)

    plt.plot(r1, V, 'r', label="func")
    plt.plot(r1, grad, 'g', label="gradients")
    plt.legend()
    plt.show()
    #V = poten_from_tf.Potential('potens_tensorflow/PyrroleWater_NN.h5', coords1)
