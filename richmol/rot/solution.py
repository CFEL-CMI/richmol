import numpy as np
from richmol.rot.basis import SymtopBasis
from richmol.rot.symmetry import symmetrize
from richmol.rot.J import Jxx, Jyy, Jzz, Jxy, Jyx, Jxz, Jzx, Jyz, Jzy, JJ, Jp, Jm
from scipy.sparse import csr_matrix


_hamiltonians = dict() # Watson-type effective Hamiltonians
_constants = dict() # parameters of effective Hamiltonians

class dummyMolecule:
    const = []
    def __getattr__(self, name):
        self.const.append(name)
        return 1

def register_ham(func):
    """Registers Hamiltonian and a set of molecular attributes it is using""" 
    bas = SymtopBasis(0)
    mol = dummyMolecule()
    func(Jzz(bas), mol, bas)
    _hamiltonians[func.__name__] = func
    _constants[func.__name__] = mol.const
    return func


class H0Tensor():
    """Casts matrix elements of rotational Hamiltonian into M- and K-tensor form, similar to labtens.LabTensor

    Args:
        mol : Molecule
            Molecular parameters.
        basis : nested dict
            Wave functions in symmetric-top basis (SymtopBasis class) for different values
            of J quantum number and different symmetries, i.e., basis[J][sym] -> SymtopBasis.
        thresh : float
            Threshold for neglecting matrix elements.

    Attrs:
        kmat : nested dict
            See labtens.LabTensor.kmat, here irrep = 0
        mmat : nested dict
            See labtens.LabTensor.mmat, here cart = "0" and irrep = 0
    """
    def __init__(self, mol, basis, thresh=1e-12):
        Jlist = [J for J in basis.keys()]
        symlist = list(set([sym for J in basis.keys() for sym in basis[J].keys()]))
        self.kmat = {(J, J) : {(sym, sym) : {} for sym in symlist} for J in Jlist}
        self.mmat = {(J, J) : {(sym, sym) : { "0" : {} } for sym in symlist} for J in Jlist}
        for J, bas_J in basis.items():
            for sym, bas_sym in bas_J.items():
                H = hamiltonian(mol, bas_sym)
                kmat, mmat = bas_sym.overlap(H)
                kmat[np.abs(kmat) < thresh] = 0
                mmat[np.abs(mmat) < thresh] = 0
                kmat_csr = csr_matrix(kmat)
                mmat_csr = csr_matrix(mmat)
                self.kmat[(J,J)][(sym,sym)][0] = kmat_csr
                self.mmat[(J,J)][(sym,sym)]["0"][0] = mmat_csr


def solve(mol, Jmin=0, Jmax=10, verbose=False):
    """Solves rotational eigenvalue problem

    Args:
        mol : Molecule
            Molecular parameters.
        Jmin, Jmax : int
            Min and max values of J quantum number.
        verbose : bool
            If True, some log will be printed.

    Returns:
        sol : nested dict
            Wave functions in symmetric-top basis (SymtopBasis class) for different values
            of J=Jmin..Jmax and different symmetries, i.e., sol[J][sym] -> SymtopBasis.
    """
    assert(Jmax >= Jmin), f"Jmax = {Jmax} < Jmin = {Jmin}"
    Jlist = [J for J in range(Jmin, Jmax+1)]

    sol = {}
    for J in Jlist:
        symbas = symmetrize(SymtopBasis(J, linear=mol.linear()), sym=mol.sym)
        sol[J] = {}
        for sym,bas in symbas.items():
            if verbose is True:
                print(f"solve for J = {J} and symmetry {sym}")
            H = hamiltonian(mol, bas, verbose=verbose)
            hmat, _ = bas.overlap(H)
            enr, vec = np.linalg.eigh(hmat.real)
            bas = bas.rotate(krot=(vec.T, enr))
            bas.sym = sym
            bas.abc = mol.abc
            sol[J][sym] = bas
    return sol


def hamiltonian(mol, bas, verbose=False):
    """Computes action of Hamiltonian operator on wave function

    Builds Hamiltonian from the user-defined rotational constants (mol.ABC_exp or mol.B_exp)
    when available, otherwise uses the rotational kinetic energy matrix G.

    Args:
        mol : Molecule
            Molecular parameters.
        bas : SymtopBasis
            Rotational wave functions in symmetric-top basis.
    """
    if hasattr(mol, 'B_exp') or mol.linear == True:
        # linear molecule
        B = mol.B
        if verbose is True:
            print(f"build rigid-rotor Hamiltonian for linear molecule, B = {B} cm-1")
        H = B * JJ(bas)
    elif hasattr(mol, 'ABC_exp'):
        # nonlinear molecule, build Hamiltonian from constants
        A, B, C = mol.ABC
        if verbose is True:
            print(f"build rigid-rotor Hamiltonian from rotational constants, A, B, C = {A, B, C} cm-1")
        try:
            ind = [("x","y","z").index(s) for s in list(mol.abc.lower())]
        except ValueError:
            raise ValueError(f"illegal value for abc = '{abc}'") from None
        # construct Hamiltonian
        Jxyz = [Jxx(bas), Jyy(bas), Jzz(bas)]
        H = A * Jxyz[ind[0]] + B * Jxyz[ind[1]] + C * Jxyz[ind[2]]
    else:
        # nonlinear molecule, build Hamiltonian from kinetic energy matrix
        if verbose is True:
            print(f"build rigid-rotor Hamiltonian from G-matrix")
        gmat = mol.gmat()
        H = 0.5 * ( gmat[0,0] * Jxx(bas) + \
                    gmat[0,1] * Jxy(bas) + \
                    gmat[0,2] * Jxz(bas) + \
                    gmat[1,0] * Jyx(bas) + \
                    gmat[1,1] * Jyy(bas) + \
                    gmat[1,2] * Jyz(bas) + \
                    gmat[2,0] * Jzx(bas) + \
                    gmat[2,1] * Jzy(bas) + \
                    gmat[2,2] * Jzz(bas) )

    # add centrifugal distortion terms or custom Hamiltonian
    try:
        watson = mol.watson
        if watson in _hamiltonians:
            H = _hamiltonians[watson](H, mol, bas, verbose=verbose)
        else:
            raise TypeError(f"Hamiltonian '{watson}' is not available") from None
    except AttributeError:
        pass

    return H


@register_ham
def watson_a(H0, mol, bas, verbose=False):
    """Watson-type asymmetric top Hamiltonian in A standard reduced form
    (J. K. G. Watson in "Vibrational Spectra and Structure" (Ed: J. Durig) Vol 6 p 1, Elsevier, Amsterdam, 1977).

    :math:`H = H_{rr} - D_{J} * J^{4} - D_{JK} * J^{2} * J_{z}^{2} - D_{K} * J_{z}^{4}`
    :math:`-\\frac{1}{2} * [ d_{1} * J^{2} + d_{2} * J_{z}^{2}, J_{+}^{2} + J_{-}^{2} ]_{+}`
    :math:`+ H_{J} * J^{6} + H_{JK} * J^{4} * J_{z}^{2} + H_{KJ} * J^{2} * J_{z}^{4} + H_{K} * J_{z}^{6}`
    :math:`+ \\frac{1}{2} * [ h_{1} * J^{4} + h_{2} * J^{2} * J_{z}^{2} + h_{3} * J_{z}^{4}, J_{+}^{2} + J_{-}^{2} ]_{+}`
    """
    J2 = JJ(bas)
    J4 = J2 * J2
    J6 = J2 * J4
    Jz2 = Jzz(bas)
    Jz4 = Jz2 * Jz2
    Jz6 = Jz2 * Jz4
    J2Jz2 = J2 * Jz2
    J4Jz2 = J2 * J2Jz2
    J2Jz4 = J2 * Jz4
    Jp2 = Jp() * Jp(bas)
    Jm2 = Jm() * Jm(bas)

    expr = {}
    expr['dj']  = (-1) * J4
    expr['djk'] = (-1) * J2Jz2
    expr['dk']  = (-1) * Jz4
    expr['d1'] = (-0.5) * (J2*(Jp2+Jm2)+Jp2*J2+Jm2*J2)
    expr['d2'] = (-0.5) * (Jz2*(Jp2+Jm2)+Jp2*Jz2+Jm2*Jz2)
    expr['hj']  = J6
    expr['hjk'] = J4Jz2
    expr['hkj'] = J2Jz4
    expr['hk']  = Jz6
    expr['h1'] = (0.5) * (J4*(Jp2+Jm2)+Jp2*J4+Jm2*J4)
    expr['h2'] = (0.5) * (J2Jz2*(Jp2+Jm2)+Jp2*J2Jz2+Jm2*J2Jz2)
    expr['h3'] = (0.5) * (Jz4*(Jp2+Jm2)+Jp2*Jz4+Jm2*Jz4)

    H = H0

    for key, val in expr.items():
        try:
            const = getattr(mol, key)
            if verbose is True:
                print(f"add 'watson_s' term '{key}'")
            H = H + const * val
        except AttributeError:
            pass

    return H


@register_ham
def watson_s(H0, mol, bas, verbose=False):
    """Watson-type asymmetric top Hamiltonian in S standard reduced form
    (J. K. G. Watson in "Vibrational Spectra and Structure" (Ed: J. Durig) Vol 6 p 1, Elsevier, Amsterdam, 1977).

    :math:`H = H_{rr} - D_{J} * J^{4} - D_{JK} * J^{2} * J_{z}^{2} - D_{K} * J_{z}^{4}`
    :math:`+ d_{1} * J^{2} * (J_{+}^{2} + J_{-}^{2}) + d_{2} * (J_{+}^{4} + J_{-}^{4})`
    :math:`+ H_{J} * J^{6} + H_{JK} * J^{4} * J_{z}^{2} + H_{KJ} * J^{2} * J_{z}^{4} + H_{K} * J_{z}^{6}`
    :math:`+ h_{1} * J^{4} * (J_{+}^{2} + J_{-}^{2}) + h_{2} * J^{2} * (J_{+}^{4} + J_{-}^{4})`
    :math:`+ h_{3} * (J_{+}^{6} + J_{-}^{6})`
    """
    J2 = JJ(bas)
    J4 = J2 * J2
    J6 = J2 * J4
    Jz2 = Jzz(bas)
    Jz4 = Jz2 * Jz2
    Jz6 = Jz2 * Jz4
    J2Jz2 = J2 * Jz2
    J4Jz2 = J2 * J2Jz2
    J2Jz4 = J2 * Jz4
    Jp2 = Jp() * Jp(bas)
    Jm2 = Jm() * Jm(bas)
    Jp4 = Jp2 * Jp2
    Jm4 = Jm2 * Jm2
    Jp6 = Jp2 * Jp4
    Jm6 = Jm2 * Jm4

    expr = {}

    expr['dj']  = (-1) * J4
    expr['djk'] = (-1) * J2Jz2
    expr['dk']  = (-1) * Jz4
    expr['d1']  = J2 * (Jp2 + Jm2)
    expr['d2']  = (Jp4 + Jm4)
    expr['hj']  = J6
    expr['hjk'] = J4Jz2
    expr['hkj'] = J2Jz4
    expr['hk']  = Jz6
    expr['h1']  = J4 * (Jp2 + Jm2)
    expr['h2']  = J2 * (Jp4 + Jm4)
    expr['h3']  = (Jp6 + Jm6)

    H = H0

    for key, val in expr.items():
        try:
            const = getattr(mol, key)
            if verbose is True:
                print(f"add 'watson_s' term '{key}'")
            H = H + const * val
        except AttributeError:
            pass

    return H

