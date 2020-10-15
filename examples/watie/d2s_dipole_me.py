
from richmol.watie import RigidMolecule, SymtopBasis, Jxx, Jyy, Jzz, CartTensor, settings, WignerD
import numpy as np
import sys
settings.assign_nprim = 2
settings.assign_ndig_c2 = 1

d2s = RigidMolecule()

d2s.XYZ = ( "angstrom", \
            "S",   0.00000000,        0.00000000,        0.10358697, \
            "H2", -0.96311715,        0.00000000,       -0.82217544, \
            "H2",  0.96311715,        0.00000000,       -0.82217544 )

d2s.tensor = ("dipole moment", [0, 0, -9.70662418E-01])

d2s.frame = "pas"

Bx, By, Bz = d2s.B

Jmax = 10
enr_all = []
wavefunc = {}

for J in range(Jmax+1):
    bas = SymtopBasis(J)
    Jx2 = Jxx(bas)
    Jy2 = Jyy(bas)
    Jz2 = Jzz(bas)
    H = Bx * Jx2 + By * Jy2 + Bz * Jz2
    hmat = bas.overlap(H)
    enr, vec = np.linalg.eigh(hmat.real)
    enr_all += [(J,i,e) for i,e in enumerate(enr)]

    wavefunc[J] = bas.rotate(krot=(vec.T, enr))

    # store energies in Richmol energy file
    wavefunc[J].store_richmol("d2s_energies")
    continue

    # check if Hamiltonian is diagonal
    Jx2 = Jxx(wavefunc[J])
    Jy2 = Jyy(wavefunc[J])
    Jz2 = Jzz(wavefunc[J])
    H = Bx * Jx2 + By * Jy2 + Bz * Jz2
    hmat = wavefunc[J].overlap(H)
    hmat = hmat.real
    print( "J = ", J, "max H(off-diag) = ", np.max( abs( hmat - np.diag(np.diag(hmat)) )), \
           "H(diag):", np.diag(hmat) )

wig = WignerD(3,3,3)

mu = CartTensor(d2s.tensor["dipole moment"])

for J1 in range(Jmax+1):
    for J2 in range(Jmax+1):
        if abs(J1-J2)>1: continue
        #mu.store_richmol(wavefunc[J1], wavefunc[J2], thresh=1e-12)
        wig.store_richmol(wavefunc[J1], wavefunc[J2], thresh=1e-12)
