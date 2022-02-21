import numpy as np
from rme import spinMe_IxI
from basis import nearEqualCoupling
from richmol.field import CarTens
import py3nj
import sys


def defaultSymmetryRules(spin, rovibSym):
    return True


def solve(f, spins, h0, ss, symmetryRules=defaultSymmetryRules):

    spinQuanta, jQuanta = nearEqualCoupling(f, spins)

    symmetries1 = h0.symlist1
    symmetries2 = h0.symlist2

    # list with unique symmetry labels
    symList = [sym for symJ in symmetries1.values() for sym in symJ] \
            + [sym for symJ in symmetries2.values() for sym in symJ]
    symList = list(set(symList))

    # check symmetry rules
    for spin in spinQuanta:
        if all(not symmetryRules(spin, sym) for sym in symList):
            raise ValueError(
                f"symmetryRules forbid coupling of spin '{spin}' state " + \
                f"with all rovibrational symmetries '{symList}', " + \
                f"check symmetryRules() function") from None

    if any(n < 0 or n > len(spins) for n12 in ss.keys() for n in n12):
        raise ValueError(
            f"Bad spin indices in keys of 'ss' argument = {list(ss.keys())}, " + \
            f"these must be positive integers not exceeding total number of " + \
            f"spins = {len(spins)}") from None

    if any(n[0] == n[1] for n in ss.keys()):
        raise ValueError(f"Same-center spin-spin coupling is not allowed") from None

    # nuclear spin reduced matrix elements
    rank = 2
    rme = spinMe_IxI(spinQuanta, spins, rank)

    # 6j-symbol prefactor

    coef = np.zeros((len(jQuanta), len(jQuanta)), dtype=np.float64)
    for i, (spin1, j1) in enumerate(zip(spinQuanta, jQuanta)):
        for j, (spin2, j2) in enumerate(zip(spinQuanta, jQuanta)):
            fac = spin2[-1] + j1 + j2 + f
            assert (float(fac).is_integer()), f"Non-integer power in (-1)**f: '(-1)**{fac}'"
            fac = int(fac)
            coef[i,j] = (-1)**fac * np.sqrt((2 * j1 + 1) * (2 * j2 + 1)) \
                      * py3nj.wigner6j(int(spin1[-1]*2), int(j1*2), int(f*2),
                                       int(j2*2), int(spin2[-1]*2), 2*2)

    # build Hamiltonian matrix

    for i, (spin1, j1) in enumerate(zip(spinQuanta, jQuanta)):
        for sym1 in symmetries[j1]:
            dim1 = h0.dim_k1[j1][sym1]
            for j, (spin2, j2) in enumerate(zip(spinQuanta, jQuanta)):
                for sym2 in symmetries[j2]:
                    dim2 = h0.dim_k2[j2][sym2]
                    rmeMat = rme[:, :, i, j] * coef[i, j]
                    mat = []
                    for (n1, n2) in ss.keys():
                        try:
                            kmat = ss[(n1, n2)].kmat[(j1, j2)][(sym1, sym2)][rank]
                            x = k.shape
                        except (AttributeError, KeyError):
                            continue
                        mat.append(kmat.toarray() * rmeMat[n1, n2])
                    if len(mat) > 1:
                        mat = sum(mat)
                    elif len(kMat) == 1:
                        mat = mat[0]
                    else:
                        mat = np.zeros((dim1, dim2))



if __name__ == '__main__':

    def c2vOrthoParaRules(spin, rovibSym):
        assert (rovibSym.lower() in ('a1', 'a2', 'b1', 'b2')), \
                f"unknown symmetry: '{rovibSym}'"
        return (spin[-1] == 0) * (rovibSym.lower() in ('a1', 'a2')) \
            + (spin[-1] == 1) * (rovibSym.lower() in ('b1', 'b2')) \

    f = 10.0
    spins = [1/2, 1/2]
    richmolFile = "/gpfs/cfel/group/cmi/data/Theory_H2O_hyperfine/H2O-16/basis_p48/richmol_database_rovib/h2o_p48_j40_rovib.h5"
    h0 = CarTens(richmolFile, name='h0')
    ss = CarTens(richmolFile, name='spin-spin H1-H2')

    solve(f, spins, h0, ss={(0, 1): ss}, symmetryRules=c2vOrthoParaRules)
