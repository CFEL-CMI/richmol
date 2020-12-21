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

Jmax = 30

wavefunc = {}
for J in range(Jmax+1):
    bas = SymtopBasis(J, linear=True)
    H = Bx * JJ(bas) # linear molecule Hamiltonian
    hmat = bas.overlap(H)
    enr, vec = np.linalg.eigh(hmat.real)
    wavefunc[J] = bas.rotate(krot=(vec.T, enr))

# store rotational energies in Richmol energy file
for J in range(Jmax+1):
    wavefunc[J].store_richmol("OCS_j0_j"+str(Jmax)+".h5", descr="OCS Cartesian coords: C(0,0,-0.522939783141), O(0,0,-1.680839357), S(0,0,1.037160128)")
    wavefunc[J].store_richmol_old("OCS_energies_j0_j"+str(Jmax)+".rchm") # old formatted text file

mu = CartTensor(ocs.tensor["dipole moment"], name='mu', units='au', descr="ab initio CCSD(T)/ACVQZ")
alpha = CartTensor(ocs.tensor["polarizability"], name='alpha', units='au', descr="ab initio CCSD(T)/ACVQZ")

for J1 in range(Jmax+1):
    for J2 in range(J1,Jmax+1):
        if abs(J1-J2)>2: continue # selection rules for alpha
        mu.store_richmol(wavefunc[J1], wavefunc[J2], "OCS_j0_j"+str(Jmax)+".h5", thresh=1e-10)
        alpha.store_richmol(wavefunc[J1], wavefunc[J2], "OCS_j0_j"+str(Jmax)+".h5", thresh=1e-10)

        # old formatted text files
        mu.store_richmol_old(wavefunc[J1], wavefunc[J2], 'mu', "OCS_mu", thresh=1e-10)
        alpha.store_richmol_old(wavefunc[J1], wavefunc[J2], 'alpha', "OCS_alpha", thresh=1e-10)

