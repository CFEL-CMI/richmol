import numpy as np
from matplotlib import pyplot as plt

class Grid:
    def __init__(self):
        pass
    
   


if __name__=="__main__":

    from mol_xy2 import XY2_ralpha
    import poten_h2s_Tyuterev

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    # H2S, using valence-bond coordinates and Tyuterev potential
    h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)

  