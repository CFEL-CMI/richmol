import numpy as np
from basis import SymtopBasis
from symmetry import symmetrize


def solve(mol, Jmin=0, Jmax=10):

    assert(Jmax >= Jmin), f"Jmax = {Jmax} < Jmin = {Jmin}"
    Jlist = [J for J in range(Jmin, Jmax+1)]

    for J in Jlist:
        symbas = symmetrize(SymtopBasis(J), sym=mol.sym)
        for sym,bas in symbas.items():
            print(sym, bas)
        #     H = hamiltonian(mol, bas)
        #     hmat = bas.overlap(H)
        #     enr, vec = np.linalg.eigh(hmat.real)
        #     bas = bas.rotate(krot=(vec.T, enr))


