
import numpy as np
from richmol.rot import solve, Solution, LabTensor, Molecule, cos2theta, cos2theta2d
from richmol.field import CarTens


if __name__ == "__main__":

    ocs = Molecule()
    ocs .XYZ = (
        "C",  0.0,  0.0,  -0.522939783141,
        "O",  0.0,  0.0,  -1.680839357,
        "S",  0.0,  0.0,  1.037160128)

    ocs.dip = [0,0,-0.31093]
    ocs.pol = [[25.5778097,0,0],
               [0,25.5778097,0],
               [0,0,52.4651140]]

    ocs.frame = "ipas"
    ocs.sym = "C1"

    sol = solve(ocs, Jmin=0, Jmax=20)

    states = LabTensor(ocs, sol)
    dip = LabTensor(ocs.dip, sol)
    pol = LabTensor(ocs.pol, sol)

    cos2 = LabTensor(cos2theta, sol)
