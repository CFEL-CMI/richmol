from cmirichmol.watie.molecule import Molecule, MHz
from cmirichmol.watie.solve import solve
import numpy as np
import sys

ocs = Molecule()



ocs.xyz = ["angstrom",
    "C",  0.0,  0.0,  -0.522939783141,
    "O",  0.0,  0.0,  -1.680839357,
    "S",  0.0,  0.0,  1.037160128]


#ocs.dipole = ["z", -0.31093]
ocs.polariz = ["xx", 25.5778097, "yy", 25.5778097, "zz", 52.4651140]

#def cos2theta2d(theta, phi):
#   tol = 1.0e-12
#   if abs(theta-np.pi/2.0)<=tol or abs(theta-3.0*np.pi/2.0)<=tol:
#       if abs(phi)<=tol or abs(phi-np.pi)<=tol or abs(phi-2.0*np.pi)<=tol:
#           return 1.0
#   return np.cos(theta)**2/(np.sin(theta)**2*np.sin(phi)**2 + np.cos(theta)**2)

#ocs.func = (lambda theta, phi: (np.cos(theta))**2, "cos2theta")
#ocs.func = (cos2theta2d, "cos2theta2d" )



ocs.frame = "kappa,pas" # PAS with x,y,z=c,a,b for near-prolate and x,y,z=a,b,c for near-oblate

solve(ocs, jmax=30)
