from richmol.watie import RigidMolecule, SymtopBasis, JJ, settings, CartTensor
import numpy as np
import sys

##################################################################################
# OCS rotational energies and richmol matrix elements of dipole and polarizability
# using ab initio values computed at CCSD(T)/ACVQZ level of theory
##################################################################################

ocs = RigidMolecule()

ocs.XYZ = ("angstrom",
    "C",  0.0,  0.0,  -0.522939783141,
    "O",  0.0,  0.0,  -1.680839357,
    "S",  0.0,  0.0,  1.037160128)

ocs.tensor = ("dipole moment", [0,0,-0.31093])
ocs.tensor = ("polarizability", [[25.5778097,0,0], \
                                 [0,25.5778097,0], \
                                 [0,0,52.4651140]])

#ocs.frame = "pas"

Bx, By, Bz = ocs.B
print("rotational constants:", Bx, By, Bz)

# compute rotational energies for J = 0..30

Jmax = 10

wavefunc = {}
for J in range(Jmax+1):
    bas = SymtopBasis(J, linear=True)
    H = Bx * JJ(bas) # linear molecule Hamiltonian
    hmat = bas.overlap(H)
    enr, vec = np.linalg.eigh(hmat.real)
    wavefunc[J] = bas.rotate(krot=(vec.T, enr))

# store rotational energies in Richmol energy file
for J in range(Jmax+1):
    wavefunc[J].store_richmol("OCS_energies_j0_j"+str(Jmax)+".h5")

mu = CartTensor(ocs.tensor["dipole moment"], name='mu')
alpha = CartTensor(ocs.tensor["polarizability"], name='alpha')

Jmax = 10
for J1 in range(Jmax+1):
    for J2 in range(J1, Jmax + 1):
        if abs(J1-J2)>2: continue
        alpha.store_richmol(wavefunc[J1], wavefunc[J2], "OCS_energies_j0_j"+str(Jmax)+".h5", thresh=1e-10)
        #alpha.store_richmol_old(wavefunc[J1], wavefunc[J2], 'alpha', "OCS_energies", thresh=1e-10)

