"""Example of calculation of hyperfine states and hyperfine dipole spectrum
of water molecule, using rovibrational states and rovibrational matrix
elements of spin-spin, spin-rotation, and dipole moment operators
computed in TROVE (https://github.com/Trovemaster/TROVE)
and stored in a richmol database file.
"""
import numpy as np
from richmol.convert_units import MHz_to_invcm
from richmol.field import CarTens, filter
from richmol.hyperfine import Hyperfine
from richmol.hyperfine import LabTensor as HyperLabTensor
from scipy import constants


kHz_to_invcm = MHz_to_invcm(1/1000)[0] # conversion factor from kHz to cm^-1


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


def dipoleSpectrum(h0, dipole, temp=296, partFunc=None, zpe=None, filt=lambda **kw: True):
    """Computes Einstein A-coefficients and absorption coefficients
    for rovibrational transitions between states in `h0`,
    using dipole matrix elements in `dipole`.
    It is assumed that dipole matrix elements in `dipole` are in units of Debye
    and state energies in `h0` are in units of cm^-1.
    Function `filt` can be specified to filter out unwanted states in `h0`.
    Parameters `temp`, `partFunc`, and `zpe` specify temperature in Kelvin,
    partition sum, and zero-point energy in cm^-1.
    """
    acoef_fac = np.pi**4 / (3 * constants.h * 1e7) * 64.0e-36
    abs_fac = 8 * np.pi**3 / (3 * constants.h * constants.c * 1e9) * 64.0e-36
    boltz_fac = constants.h * 1e2 * constants.c / (constants.k * temp)

    # compute partition function and zpe if necessary

    enr = [e for j in h0.Jlist1 for sym in h0.symlist1[j]
           for e in h0.kmat[(j, j)][(sym, sym)][0].diagonal()]
    if not zpe:
        zpe = min(enr)
    if not partFunc:
        partFunc = sum([np.exp(-boltz_fac * (e - zpe)) for e in enr])
    print(f"partition function: {partFunc}\nzpe: {zpe}")

    # filter out unnecessary states

    h0_ = filter(h0, filt, filt)
    dipole_ = filter(dipole, filt, filt)

    # generate id-labeled states data

    states = {}
    id_num = {}
    id = 0
    assign, _ = h0_.assign()
    for j in h0_.Jlist1:
        for sym in h0_.symlist1[j]:
            enr = h0_.kmat[(j, j)][(sym, sym)][0].diagonal()
            quanta = h0_.quanta_k1[j][sym]
            deg = int(2 * j + 1)
            id0 = id + 1
            for e, q in zip(enr, quanta):
                id += 1
                states[id] = (e, deg, j, sym, q)
            id_num[(j, sym)] = np.arange(id0, id+1, 1)

    # compute dipole transition data for selected states

    irrep = 1 # for dipole moment only (rank-1 tensor)
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
            nnz_ind = np.array(ls.nonzero())
            id1 = id_num[(j1, sym1)][nnz_ind[0]]
            id2 = id_num[(j2, sym2)][nnz_ind[1]]
            nu = np.abs([states[i][0] - states[j][0] for i, j in zip(id1, id2)])
            elow = np.array([min([states[i][0], states[j][0]]) for i, j in zip(id1, id2)])
            acoef = ls.data * acoef_fac * nu**3
            intens = ls.data * abs_fac * nu * np.exp(-boltz_fac * elow) \
                   * (1 - np.exp(-boltz_fac * nu)) / partFunc
            trans += [(i, j, a1, a2, n) for i, j, a1, a2, n in zip(id1, id2, acoef, intens, nu)]

    return states, trans


def stateFilter(**kwargs):
    passE = True
    passJ = True
    if 'enr' in kwargs:
        passE = float(kwargs['enr']) < 15000.0
    if 'J' in kwargs:
        passJ = round(float(kwargs['J']), 1) <= fmax
    return passE * passJ



if __name__ == '__main__':

    """Compute hyperfine energies and spectrum of water (H2^16O)"""

    spins = [1/2, 1/2]  # nuclear spins I(H1) and I(H2)
    fmin = 0   # min value of F = I + J
    fmax = 10  # max value of F

    partFunc = 174.5813  # partition sum for water at T = 296 K  (taken from https://www.exomol.com/db//H2O/1H2-16O/POKAZATEL/1H2-16O__POKAZATEL.pf)

    storeHyperfine = True  # compute and store (True) or read from file (False) hyperfine states
    storeTensors = True    # compute and store (True) or read from file (False) hyperfine matrix elements of operators
    tensorNames = ['dipole', 'quad']  # list of names of operators to compute hyperfine matrix elements of
    storeSpectrum = True  # if True, compute spectrum and store in ExoMol two-file format
    statesFile = "h2o_exomol.states"  # name of the output ExoMol states file
    transFile = "h2o_exomol.trans"  # name of the output ExoMol transitions file

    # richmol database file with rovibrational solutions and matrix elements
    # of various Cartesian tensor operators for water molecule
    richmolFile = "/gpfs/cfel/group/cmi/data/Theory_H2O_hyperfine/H2O-16/basis_p48/richmol_database_rovib/h2o_p48_j40_rovib.h5"

    # richmol database file with hyperfine solutions and matrix elements
    # of Cartesian tensor operators for water molecule
    richmolHyperFile = "h2o_p48_j40_hyper.h5"

    # obtain hyperfine solutions and store them in `richmolHyperFile`
    if storeHyperfine:

        # read rovibrational Hamiltonian from `richmolFile`
        h0 = CarTens(richmolFile, name='h0')

        # read spin-spin matrix elements from `richmolFile`
        ss = CarTens(richmolFile, name='spin-spin H1-H2')

        # read spin-rotation matrix elements for H1 and H2 from `richmolFile`
        sr1 = CarTens(richmolFile, name='spin-rot H1')
        sr2 = CarTens(richmolFile, name='spin-rot H2')

        # convert spin-rotation and spin-spin from kHz to cm^-1
        ss *= kHz_to_invcm
        sr1 *= -kHz_to_invcm
        sr2 *= -kHz_to_invcm

        # solve hyperfine problem and return result in `CarTens` object
        h0 = Hyperfine(fmin, fmax, spins, h0, ss={(0, 1): ss}, sr={0: sr1, 1: sr2},
                       symmetryRules=c2vOrthoParaRules)

        # store hyperfine solutions in `richmolHyperFile`
        print(f"store hyperfine solutions into file '{richmolHyperFile}'")
        h0.store(richmolHyperFile, name='h0', replace=True)

    # compute and store hyperfine matrix elements of tensor operators in `richmolHyperFile`
    if storeTensors:

        # read hyperfine solutions from `richmolHyperFile`
        h0 = CarTens(richmolHyperFile, name='h0')

        for name in tensorNames:

            print(f"compute hyperfine matrix elements of '{name}' operator")
            # read rovibrational matrix elements of tensor operator from `richmolFile`
            tens = CarTens(richmolFile, name=name)

            # compute hyperfine matrix elements of tensor operator
            tens = HyperLabTensor(h0, tens)

            # store matrix elements in `richmolHyperFile`
            print(f"store matrix elements into file '{richmolHyperFile}'")
            tens.store(richmolHyperFile, name=name, replace=True)

    # compute spectrum (Einstein A-coefficients and absorption coefficients)
    # and store it in ExoMol format, i.e. files with states and transitions
    if storeSpectrum:

        # read hyperfine solutions from `richmolHyperFile`
        h0 = CarTens(richmolHyperFile, name='h0')

        # read hyperfine matrix elements of dipole from `richmolHyperFile`
        mu = CarTens(richmolHyperFile, name='dipole')

        # compute states and transitions data in ExoMol format
        states, trans = dipoleSpectrum(h0, mu, partFunc=partFunc, filt=stateFilter)

        # store states data in `statesFile` and transitions data in `transFile`
        with open(statesFile, 'w') as fl:
            for id, state in states.items():
                enr, deg, f, sym, quanta = state
                (spin, j, sym_rv, (q_rv, e_rv)), e = quanta
                fl.write(" %6i"%id + " %16.8f"%enr + " %3i"%deg + \
                         f"   F {f}   Sym {sym}  I {spin}  J {j}" + \
                         f"   Sym_rv {sym_rv}   E_rv {e_rv}   {q_rv}" + "\n")
        with open(transFile, 'w') as fl:
            for (id1, id2, acoef, intens, nu) in trans:
                if intens > 1e-36:
                    fl.write(" %6i"%id1 + " %6i"%id2 + " %16.8e"%acoef + \
                             " %16.8e"%intens + " %16.8f"%nu + "\n")
