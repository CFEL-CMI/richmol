import numpy as np
import itertools
from richmol.convert_units import MHz_to_invcm
from richmol.field import CarTens, filter
from richmol.hyperfine import Hyperfine
from richmol.hyperfine import LabTensor as HyperLabTensor
from scipy import constants
from scipy.sparse import csr_matrix


kHz_to_invcm = MHz_to_invcm(1/1000)[0] # converts kHz to cm^-1


def c2vOrthoParaRules(spin, rovibSym):
    """Example of selection rules for water molecule
    where all ortho and all para states are included
    """
    assert (rovibSym.lower() in ('a1', 'a2', 'b1', 'b2')), \
            f"unknown symmetry: '{rovibSym}'"
    sym = { (0.0, 'a1') : 'b2',
            (0.0, 'a2') : 'b1',
            (1.0, 'b1') : 'b1',
            (1.0, 'b2') : 'b2'
          }
    I = round(float(spin[-1]), 1)
    key = (I, rovibSym.lower())
    try:
        return sym[key]
    except KeyError:
        return ""


def c2vOrthoParaB1Rules(spin, rovibSym):
    """Example of selection rules for water molecule
    where ortho and para states are included that
    belong to the total symmetry B1
    """
    assert (rovibSym.lower() in ('a1', 'a2', 'b1', 'b2')), \
            f"unknown symmetry: '{rovibSym}'"
    sym = {
            (0.0, 'a2') : 'b1',
            (1.0, 'b1') : 'b1',
          }
    I = round(float(spin[-1]), 1)
    key = (I, rovibSym.lower())
    try:
        return sym[key]
    except KeyError:
        return None


def c2vOrthoParaB2Rules(spin, rovibSym):
    """Example of selection rules for water molecule
    where ortho and para states are included that
    belong to the total symmetry B2
    """
    assert (rovibSym.lower() in ('a1', 'a2', 'b1', 'b2')), \
            f"unknown symmetry: '{rovibSym}'"
    sym = {
            (0.0, 'a1') : 'b2',
            (1.0, 'b2') : 'b2'
          }
    I = round(float(spin[-1]), 1)
    key = (I, rovibSym.lower())
    try:
        return sym[key]
    except KeyError:
        return ""


def storeHyperfine(fmin, fmax, spins, symmetryRules, richmolRovibFile, richmolHyperFile):
    # rovibrational Hamiltonian
    h0 = CarTens(richmolRovibFile, name='h0')
    # spin-spin matrix elements
    ss = CarTens(richmolRovibFile, name='spin-spin H1-H2')
    # spin-rotation matrix elements for H1 and H2
    sr1 = CarTens(richmolRovibFile, name='spin-rot H1')
    sr2 = CarTens(richmolRovibFile, name='spin-rot H2')
    # convert spin-rotation and spin-spin from kHz to cm^-1
    ss *= kHz_to_invcm
    sr1 *= -kHz_to_invcm
    sr2 *= -kHz_to_invcm
    # solve hyperfine problem and return the result in CarTens object
    hyper_h0 = Hyperfine(fmin, fmax, spins, h0, ss={(0, 1): ss},
                         sr={0: sr1, 1: sr2}, symmetryRules=symmetryRules)
    print(f"store hyperfine solutions into file '{richmolHyperFile}'")
    hyper_h0.store(richmolHyperFile, name='h0', replace=True)


def storeHyperfineTens(richmolFile, richmolHyperFile, tensorNames):
    print(f"read hyperfine solutions into file '{richmolHyperFile}'")
    h0 = CarTens(richmolHyperFile, name='h0')
    for name in tensorNames:
        print(f"compute hyperfine matrix elements of '{name}' operator")
        tens = CarTens(richmolFile, name=name)
        hyper_tens = HyperLabTensor(h0, tens)
        print(f"store matrix elements into file '{richmolHyperFile}'")
        hyper_tens.store(richmolHyperFile, name=name, replace=True)


def einsteinACoef(h0, dipole, ls_thresh=1e-20, filt=lambda **kw: True):

    # filter out some states

    h0_ = filter(h0, filt, filt)
    dipole_ = filter(dipole, filt, filt)

    # generate id-labeled states data

    states = {}
    idNum = {}
    id = 0
    assign, _ = h0_.assign()
    for j in h0_.Jlist1:
        for sym in h0_.symlist1[j]:
            enr = h0_.kmat[(j, j)][(sym, sym)][0].diagonal()
            quanta = assign[j][sym]['k']
            deg = int(2 * j + 1)
            id0 = id + 1
            for e, q in zip(enr, quanta):
                id += 1
                states[id] = (e, deg, j, sym, q)
            idNum[(j, sym)] = np.arange(id0, id+1, 1)

    # compute dipole transition data for selected states

    irrep = 1 # for dipole moment only (rank-1 tensor)
    acoef_fac = np.pi**4 / (3 * constants.value('Planck constant') * 1e7) * 64.0e-36
    trans = []

    for jPair in list(set(dipole_.mmat.keys()) & set(dipole_.kmat.keys())):

        j1, j2 = jPair
        if j1 > j2:
            continue

        mmatJ = dipole_.mmat[jPair]
        kmatJ = dipole_.kmat[jPair]

        symList = {}

        for symPair in list(set(mmatJ.keys()) & set(kmatJ.keys())):

            sym1, sym2 = symPair

            # this is to exclude repeated pairs of symmetries for j1 == j2
            if j1 == j2:
                try:
                    if (sym2, sym1) in symList[j1]:
                        continue
                    else:
                        symList[j1].append(symPair)
                except KeyError:
                    symList[j1] = [symPair]

            mmat = mmatJ[symPair]
            kmat = kmatJ[symPair]

            k2 = abs(kmat[irrep]).power(2)
            m2 = np.sum(sum(abs(mmat).power(2) for mmat in mmat[irrep].values()))
            # ls = k2 * m2 / (2*j1+1) / (2*j2+1)
            ls = k2 # the term `m2 / (2*j1+1) / (2*j2+1)` must be equal to 1
            ls = ls.toarray()
            ls[ls < ls_thresh] = 0
            ls = csr_matrix(ls)
            nnz_ind = np.array(ls.nonzero())
            id1 = idNum[(j1, sym1)][nnz_ind[0]]
            id2 = idNum[(j2, sym2)][nnz_ind[1]]

            nu = np.abs([states[i][0] - states[j][0] for i, j in zip(id1, id2)])
            acoef = ls.data * acoef_fac * nu**3
            trans += [(i, j, a, n) for i, j, a, n in zip(id1, id2, acoef, nu)]
    return states, trans


def stateFilter(**kwargs):
    passE = True
    passJ = True
    if 'enr' in kwargs:
        passE = float(kwargs['enr']) < 15000.0
    if 'J' in kwargs:
        passJ = round(float(kwargs['J']), 1) <= 10.0
    return passE * passJ


if __name__ == '__main__':

    spins = [1/2, 1/2]
    fmin = 0
    fmax = 10

    # richmolFile = "/gpfs/cfel/group/cmi/data/Theory_H2O_hyperfine/H2O-16/basis_p48/richmol_database_rovib/h2o_p48_j40_rovib.h5"
    # richmolHyperFile = "h2o_p48_j40_hyper.h5"
    # storeHyperfine(fmin, fmax, spins, c2vOrthoParaRules, richmolFile, richmolHyperFile)
    # storeHyperfineTens(richmolFile, richmolHyperFile, ["dipole"])

    # richmolHyperFile = "h2o_p48_j40_hyperfine.h5"
    richmolHyperFile = "h2o_p48_j40_hyper.h5"

    h0 = CarTens(richmolHyperFile, name='h0')
    mu = CarTens(richmolHyperFile, name='dipole')
    states, trans = einsteinACoef(h0, mu, filt=stateFilter)

    statesFile = "water.states_richmol"
    transFile = "water.trans_richmol"

    # with open(statesFile, 'w') as fl:
    #     for id, state in states.items():
    #         enr, deg, f, sym, quanta = state
    #         q_rv, e = quanta
    #         fl.write(" %6i"%id + " %16.8f"%enr + " %3i"%deg + \
    #                  f"   F {f}   Sym {sym}  {q_rv}" + "\n")

    with open(statesFile, 'w') as fl:
        for id, state in states.items():
            enr, deg, f, sym, quanta = state
            (spin, j, sym_rv, (q_rv, e_rv)), e = quanta
            fl.write(" %6i"%id + " %16.8f"%enr + " %3i"%deg + \
                     f"   F {f}   Sym {sym}  I {spin}  J {j}" + \
                     f"   Sym_rv {sym_rv}   E_rv {e_rv}   {q_rv}" + "\n")

    with open(transFile, 'w') as fl:
        for (id1, id2, acoef, nu) in trans:
            if acoef > 1e-20:
                fl.write(" %6i"%id1 + " %6i"%id2 + "%16.8e"%acoef + "%16.8f"%nu + "\n")
