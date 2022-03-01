import numpy as np
from scipy.sparse import csr_matrix
from reduced_me import spinMe_IxI, spinMe
from basis import nearEqualCoupling, spinNearEqualCoupling
from richmol.field import CarTens
import py3nj
from collections import defaultdict


def defaultSymmetryRules(spin, rovibSym):
    return "A"


def hamiltonian(f, spins, h0, quad=None, sr=None, ss=None, eQ=None,
                totSym="A", symmetryRules=defaultSymmetryRules, verbose=False):
    """Builds hyperfine Hamiltonian matrix, given value of quantum number
    of total angular momentum F, values of nuclear spins, symmetry selection
    rules, and hyperfine interaction tensor operators

    Args:
        f : float
            Value of quantum number of the total angular momentum operator F = I + J
        spins : list
            List of nuclear spin values
        h0 : :py:class:`richmol.field.CarTens` class
            Rovibrational Hamiltonian
        quad : dictionary with elements :py:class:`richmol.field.CarTens` class
            Nuclear quadrupole tensor operator quad[i] for each i-th spin
            center in `spins`
        sr : dictionary with elements :py:class:`richmol.field.CarTens` class
            Nuclear spin-rotation tensor operator sr[i] for each i-th spin
            center in `spins`
        ss : dictionary with elements :py:class:`richmol.field.CarTens` class
            Nuclear spin-spin tensor operator ss[(i, j)] for different pairs
            of spin centers in `spins`
        eQ : list
            List of nuclear quadrupole constants for each spin center
            in `spin` parameter
        sym : str
            Total spin-rovibrational symmetry
        symmetryRules : function(**kw)
            State filter function, take as parameters full spin state description
            `spinState` (total value of nuclear spin is `spinState[-1]`) and
            rovibrational symmetry label `symmetry` and returns total spin-rovibrational
            symmetry label. It may return `None` or empty string if certain
            symmetries need to be excluded from calculation.

    Return:
        hamMat : matrix
            Hyperfine interaction Hamiltonian matrix
        hamMat0 : matrix
            Rovibrational Hamiltonian matrix
        quanta : list
            List of spin-rovibrational quanta. Each element of `quanta` is a tuple
            (spin, j, sym, k), where `spin` contains list of spin quanta in coupled
            basis set (spin[-1] is the total spin quantum number), `j` is rotational
            quantum number, `sym` is symmetry of rovibrational wave function,
            and `k=h0.quanta_k1[j][sym]` consists of list of rovibrational quantum
            numbers (sometimes including rovibrational energy)
    """
    if verbose:
        print(f"Build hyperfine Hamiltonian for F = {f}")
        model = " + ".join(selem for selem, elem in zip(("H0", "quadrupole", "spin-rotation", "spin-spin"),
                           (h0, quad, sr, ss)) if elem)
        print(f"model: {model}")

    # coupling of spin and J quanta

    spinQuanta, jQuanta = nearEqualCoupling(f, spins)
    spinBasis = spinNearEqualCoupling(spins)

    if any(j not in h0.Jlist1 for j in jQuanta):
        raise ValueError(
            f"Some of J quanta necessary for F = {f}, J = {jQuanta}, " + \
            f"are not spanned by the bra-component of the basis, J = {Jlist1}") from None

    if any(j not in h0.Jlist2 for j in jQuanta):
        raise ValueError(
            f"Some of J quanta necessary for F = {f}, J = {jQuanta}, " + \
            f"are not spanned by the ket-component of the basis, J = {Jlist2}") from None

    # dictionary of allowed rovibrational symmetries for different spin states

    if verbose:
        print(f"total spin-rovibrational symmetry: {totSym}")
        print("spin quanta  |  J  |  allowed rovibrational symmetry for bra and ket states")
    allowedSym1 = dict()
    allowedSym2 = dict()
    for spin, j in zip(spinQuanta, jQuanta):
        key = (spin, j)
        allowedSym1[key] = [sym for sym in h0.symlist1[j] if symmetryRules(spin, sym).lower() == totSym.lower()]
        allowedSym2[key] = [sym for sym in h0.symlist2[j] if symmetryRules(spin, sym).lower() == totSym.lower()]
        print(spin, " %3i"%j, "  ", allowedSym1[key], "  ", allowedSym2[key])

    # spin-rotation intermediates

    if sr:

        if verbose:
            print(f"list of spin-rotation centers: {list(sr.keys())}")

        tol = 1e-8
        Nl = np.array([1, -np.sqrt(3), np.sqrt(5)], dtype=np.float64)
        Nl1 = Nl * np.array([(-1)**l for l in range(3)])
        Nl2 = Nl

        if any(n < 0 or n > len(spins) for n in sr.keys()):
            raise ValueError(
                f"Bad spin indices in keys of 'sr' parameter = {list(sr.keys())}, " + \
                f"these must be positive integers not exceeding total number of " + \
                f"spins = {len(spins)}") from None

        coef1 = np.zeros((len(jQuanta), len(jQuanta), 3), dtype=np.float64)
        coef2 = np.zeros((len(jQuanta), len(jQuanta), 3), dtype=np.float64)

        for i, (spin1, j1) in enumerate(zip(spinQuanta, jQuanta)):
            for j, (spin2, j2) in enumerate(zip(spinQuanta, jQuanta)):
                fac = spin2[-1] + f
                assert (float(fac).is_integer()), f"Non-integer power in (-1)**f: '(-1)**{fac}'"
                fac = int(fac)
                prefac = (-1)**fac * np.sqrt((2*j1+1) * (2*j2+1)) \
                       * py3nj.wigner6j(int(spin1[-1]*2), int(j1*2), int(f*2), 
                                        int(j2*2), int(spin2[-1]*2), 2)
                if j2 > 0:
                    threej = py3nj.wigner3j(int(j2*2), 2, int(j2*2), -int(j2*2), 0, int(j2*2))
                    if abs(threej) < tol:
                        raise ValueError(f"can't divide by 3j-symbol (J 1 J)(-J 0 J) = {threej}") from None
                    coef1[i, j, :] = prefac * j2 / threej * Nl1 \
                                   * py3nj.wigner6j([l*2 for l in range(3)], [2]*3, [2]*3,
                                                    [int(j2*2)]*3, [int(j1*2)]*3, [int(j2*2)]*3)
                if j1 > 0:
                    threej = py3nj.wigner3j(int(j1*2), 2, int(j1*2), -int(j1*2), 0, int(j1*2))
                    if abs(threej) < tol:
                        raise ValueError(f"can't divide by 3j-symbol (J' 1 J')(-J' 0 J') = {threej}") from None
                    coef2[i, j, :] = prefac * j1 / threej * Nl1 \
                                   * py3nj.wigner6j([2]*3, [l*2 for l in range(3)], [2]*3,
                                                    [int(j2*2)]*3, [int(j1*2)]*3, [int(j1*2)]*3)

        rme = spinMe(spinQuanta, spinQuanta, spins, 1, 'spin')
        rme_sr = 0.5 * ( np.einsum('kij,ijl->klij', rme, coef1) \
                       + np.einsum('kij,ijl->klij', rme, coef2) )

    # spin-spin intermediates

    if ss:

        if verbose:
            print(f"list of spin-spin centers: {list(ss.keys())}")

        if any(n < 0 or n > len(spins) for n12 in ss.keys() for n in n12):
            raise ValueError(
                f"Bad spin indices in keys of 'ss' parameter = {list(ss.keys())}, " + \
                f"these must be positive integers not exceeding total number of " + \
                f"spins = {len(spins)}") from None

        if any(n[0] == n[1] for n in ss.keys()):
            raise ValueError(
                f"Same-center spin-spin coupling is not allowed, " + \
                f"check keys of 'ss' parameter") from None

        # rank of spin-spin tensor
        ss_rank = 2

        # 6j-symbol prefactor
        coef = np.zeros((len(jQuanta), len(jQuanta)), dtype=np.float64)
        for i, (spin1, j1) in enumerate(zip(spinQuanta, jQuanta)):
            for j, (spin2, j2) in enumerate(zip(spinQuanta, jQuanta)):
                fac = spin2[-1] + j1 + j2 + f + ss_rank
                assert (float(fac).is_integer()), f"Non-integer power in (-1)**f: '(-1)**{fac}'"
                fac = int(fac)
                coef[i, j] = (-1)**fac * np.sqrt((2*j1+1) * (2*j2+1)) \
                           * py3nj.wigner6j(int(spin1[-1]*2), int(j1*2), int(f*2),
                                            int(j2*2), int(spin2[-1]*2), ss_rank*2)

        # nuclear spin reduced matrix elements x 6j coefficient
        rme_ss = spinMe_IxI(spinQuanta, spinQuanta, spinBasis, spins, ss_rank) * coef

    # quadrupole interaction intermediates

    if quad:
        raise NotImplementedError(f"Quadrupole interaction is not yet implemented") from None

    # build Hamiltonian matrix

    hamMat = []
    hamMat0 = []
    quanta = []
    quantaSpinJSym = []

    for i, (spin1, j1) in enumerate(zip(spinQuanta, jQuanta)):
        for sym1 in allowedSym1[(spin1, j1)]:

            dim1 = h0.dim_k1[j1][sym1]

            quanta += [(spin1, j1, sym1, k) for k in h0.quanta_k1[j1][sym1]]
            quantaSpinJSym.append((spin1, j1, sym1, dim1))

            hamMat_ = []
            hamMat0_ = []

            for j, (spin2, j2) in enumerate(zip(spinQuanta, jQuanta)):
                for sym2 in allowedSym2[(spin2, j2)]:

                    dim2 = h0.dim_k2[j2][sym2]

                    # diagonal rovibrational part

                    mat0 = np.zeros((dim1, dim2), dtype=np.float64)
                    if all(s1 == s2 for s1, s2 in zip(spin1, spin2)):
                        try:
                            mat0 = h0.kmat[(j1, j2)][(sym1, sym2)][0].toarray()
                        except (AttributeError, KeyError):
                            pass
                    hamMat0_.append(mat0)

                    # hyperfine terms

                    mat = np.zeros((dim1, dim2), dtype=np.float64)

                    # spin-spin part

                    if ss:
                        for (n1, n2) in ss.keys():
                            try:
                                kmat = ss[(n1, n2)].kmat[(j1, j2)][(sym1, sym2)][ss_rank].toarray()
                                mat += kmat * rme_ss[n1, n2, i, j]
                            except (AttributeError, KeyError):
                                continue

                    # spin-rotation part

                    if sr:
                        for n1 in sr.keys():
                            kmat = sr[n1].kmat[(j1, j2)][(sym1, sym2)]
                            for l, kmat_l in kmat.items():
                                try:
                                    mat += kmat_l.toarray() * rme_sr[n1, l, i, j]
                                except AttributeError:
                                    continue

                    hamMat_.append(mat)

            hamMat0.append(hamMat0_)
            hamMat.append(hamMat_)

    hamMat0 = np.bmat(hamMat0)
    hamMat = np.bmat(hamMat)

    if verbose:
        print(f"matrix dimensions: {hamMat.shape}")

    return hamMat, hamMat0, quanta, quantaSpinJSym


class Hyperfine(CarTens):

    def __init__(self, fmin, fmax, spins, h0, quad=None, sr=None, ss=None, eQ=None,
                 symmetryRules=defaultSymmetryRules):

        assert (round(float(fmin), 1) <= round(float(fmax), 1)), \
            f"fmax = {fmax} < fmin = {fmin}"
        fList = [round(elem, 1) for elem in np.arange(fmin, fmax + 1)]

        mydict = lambda: defaultdict(mydict)

        self.cart = ['0']
        self.os = [(0, 0)]
        self.rank = 0
        self.Jlist1 = fList
        self.Jlist2 = fList
        self.quantaSpinJSym = mydict()

        self.symlist1 = mydict()
        self.symlist2 = mydict()
        self.dim_m1 = mydict()
        self.dim_m2 = mydict()
        self.dim_k1 = mydict() 
        self.dim_k2 = mydict()
        self.dim1 = mydict()
        self.dim2 = mydict()
        self.quanta_k1 = mydict()
        self.quanta_k2 = mydict()
        self.quanta_m1 = mydict()
        self.quanta_m2 = mydict()
        self.kmat = mydict()
        self.mmat = mydict()
        self.vec = mydict()

        for f in fList:
            spinQuanta, jQuanta = nearEqualCoupling(f, spins)
            symList = [symmetryRules(spin, sym) for spin, j in zip(spinQuanta, jQuanta)
                       for sym in h0.symlist1[j] + h0.symlist2[j]]
            symList = list(set(sym for sym in symList if sym))
            for sym in symList:
                hmat, hmat0, quanta, quantaSpinJSym = \
                    hamiltonian(f, spins, h0, quad, sr, ss, eQ, sym,
                                symmetryRules, verbose=True)
                enr, vec = np.linalg.eigh(hmat + hmat0)

                self.dim_k1[f][sym], self.dim_k2[f][sym] = hmat.shape
                self.dim_m1[f][sym] = int(2 * f) + 1
                self.dim_m2[f][sym] = int(2 * f) + 1
                self.dim1[f][sym] = self.dim_k1[f][sym] * self.dim_m1[f][sym]
                self.dim2[f][sym] = self.dim_k2[f][sym] * self.dim_m2[f][sym]
                self.quanta_k1[f][sym] = [(q, e) for q, e in zip(quanta, enr)]
                self.quanta_k2[f][sym] = [(q, e) for q, e in zip(quanta, enr)]
                self.quanta_m1[f][sym] = [m for m in np.arange(-f, f+1)]
                self.quanta_m2[f][sym] = [m for m in np.arange(-f, f+1)]
                self.kmat[(f, f)][(sym, sym)] = {0 : csr_matrix(np.diag(enr))}
                self.mmat[(f, f)][(sym, sym)] = {0 {'0' : csr_matrix(np.eye(int(2 * f) + 1))}}

                self.vec[f][sym] = vec
                self.quantaSpinJSym[f][sym] = quantaSpinJSym

            if len(symList) > 0:
                self.symlist1[f] = symList
                self.symlist2[f] = symList



if __name__ == '__main__':
    from richmol.convert_units import MHz_to_invcm


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


    def c2vOrthoRules(spin, rovibSym):
        """Example of selection rules for water molecule
        where only all ortho states are included
        """
        assert (rovibSym.lower() in ('a1', 'a2', 'b1', 'b2')), \
                f"unknown symmetry: '{rovibSym}'"
        sym = { (1.0, 'b1') : 'b1',
                (1.0, 'b2') : 'b2'
              }
        I = round(float(spin[-1]), 1)
        key = (I, rovibSym.lower())
        try:
            return sym[key]
        except KeyError:
            return None


    def c2vParaRules(spin, rovibSym):
        """Example of selection rules for water molecule
        where only all para states are included
        """
        assert (rovibSym.lower() in ('a1', 'a2', 'b1', 'b2')), \
                f"unknown symmetry: '{rovibSym}'"
        sym = { (0.0, 'a1') : 'b2',
                (0.0, 'a2') : 'b1',
              }
        I = round(float(spin[-1]), 1)
        key = (I, rovibSym.lower())
        try:
            return sym[key]
        except KeyError:
            return None


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


    kHz_to_invcm = MHz_to_invcm(1/1000)[0]

    spins = [1/2, 1/2]

    richmolFile = "/gpfs/cfel/group/cmi/data/Theory_H2O_hyperfine/H2O-16/basis_p48/richmol_database_rovib/h2o_p48_j40_rovib.h5"
    h0 = CarTens(richmolFile, name='h0')
    ss = CarTens(richmolFile, name='spin-spin H1-H2')
    sr1 = CarTens(richmolFile, name='spin-rot H1')
    sr2 = CarTens(richmolFile, name='spin-rot H2')

    ss *= kHz_to_invcm
    sr1 *= -kHz_to_invcm
    sr2 *= -kHz_to_invcm

    hyper = Hyperfine(0, 10, spins, h0, ss={(0, 1): ss}, sr={0: sr1, 1: sr2},
                      symmetryRules=c2vOrthoParaRules)

