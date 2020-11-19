import numpy as np
from matplotlib import pyplot as plt
from scipy.special import binom

class indexmap:
    def __init__(self,b,pruntype):
        self.pruntype = pruntype #type of pruning
        self.b = b #pruning parameter
    
    def get_basis_size(self):
            return binom(self.b + 1, 3)

    def gen_map(self):
        maparray = np.zeros((self.get_basis_size(),4))

        for i1 in range(b):
            for i2 in range(b):
                for i3 in range(b):
                    maparray[i,i1] = i1
                    maparray[i,i2] = i2
                    maparray[i,i3] = i3

        return maparray 


if __name__=="__main__":

    from mol_xy2 import XY2_ralpha
    import poten_h2s_Tyuterev

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    # H2S, using valence-bond coordinates and Tyuterev potential
    h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)

  