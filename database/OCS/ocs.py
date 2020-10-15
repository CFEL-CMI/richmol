from richmol.watie import RigidMolecule, SymtopBasis, vellgt, Jxx, Jyy, Jzz, settings, CartTensor
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

ocs.frame = "pas"

Bx, By, Bz = ocs.B
print(Bx, By, Bz)

# compute rotational energies for J = 0..30

Jmax = 30

settings.assign_nprim = 2 # number of primitive functions used for state assignment
settings.assign_ndig_c2 = 6 # number of digits printed for the assignment coefficient

wavefunc = {}
for J in range(Jmax+1):
    bas = SymtopBasis(J, linear=True)
    Jy2 = Jyy(bas)
    Jz2 = Jzz(bas)
    H = By * Jy2 + Bz * Jz2
    hmat = bas.overlap(H)
    enr, vec = np.linalg.eigh(hmat.real)
    wavefunc[J] = bas.rotate(krot=(vec.T, enr))

# store rotational energies in Richmol energy file
for J in range(Jmax+1):
    wavefunc[J].store_richmol("OCS_energies_j0_j"+str(Jmax)+".rchm")

mu = CartTensor(ocs.tensor["dipole moment"])
alpha = CartTensor(ocs.tensor["polarizability"])

for J1 in range(Jmax+1):
    for J2 in range(J1,Jmax+1):
        if abs(J1-J2)>2: continue
        mu.store_richmol(wavefunc[J2], wavefunc[J1], thresh=1e-12, name="mu", fname="OCS_mu")
        alpha.store_richmol(wavefunc[J2], wavefunc[J1], thresh=1e-12, name="alpha", fname="OCS_alpha")

