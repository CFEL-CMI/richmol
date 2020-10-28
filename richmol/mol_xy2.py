from molecule import Molecule
import autograd.numpy as np
import poten_h2o_Polyansky
import poten_from_tf
import matplotlib.pyplot as plt
import sys
import itertools
import h5py as hp

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
        # change the angles to degrees.
        return xyz



if __name__=="__main__":

    mol = XY2_ralpha(masses=[12,1,1], poten=poten_h2o_Polyansky.poten)
    #coords1 = np.array([[1,1,0.5],[1,1,0.5],[1,1,0.5],[1,1,0.5],[1,1,0.5],[1,1,0.5]],dtype=np.float64)
    coords1 = np.array([[0.9586,0.9586,104.48],[0.9586,0.9586,104.48]],dtype=np.float64)
    bohr = 1.8897259886
    r = bohr*np.linspace(0.6, 1.8, 50)
    theta = np.deg2rad(np.linspace(45, 180, 50))
    grid = np.array(list(itertools.product(r,r,theta)))
    grid_save = np.array(list(itertools.product(r/bohr,r/bohr,np.rad2deg(theta))))
    #G = mol.G(coords1)
    V = 219474.624*mol.V(grid)
    with hp.File('water_PES_data.h5', 'w') as ff:
        ff.create_dataset("x", data=grid_save)
        ff.create_dataset("E", data=V)
    #V = poten_from_tf.Potential('potens_tensorflow/PyrroleWater_NN.h5', coords1)
    #plt.scatter(np.rad2deg(theta),V)

    #plt.show()
