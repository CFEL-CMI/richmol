from molecule import Molecule
import autograd.numpy as np
import poten_h2o_Polyansky
import poten_h2s_Tyuterev
#import poten_from_tf

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

    mol = XY2_ralpha(masses=[12,1,1], poten=poten_h2s_Tyuterev.poten)
    #coords1 = np.array([[1,1,0.5],[1,1,0.5],[1,1,0.5],[1,1,0.5],[1,1,0.5],[1,1,0.5]],dtype=np.float64)
    coords1 = np.array([[0.9586,0.9586,104.48],[0.9586,0.9586,104.48]],dtype=np.float64)

    G = mol.G(coords1)
    V = mol.V(coords1)
    print(V)
    #V = poten_from_tf.Potential('potens_tensorflow/PyrroleWater_NN.h5', coords1)
