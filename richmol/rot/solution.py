import numpy as np
from richmol.rot.basis import SymtopBasis
from richmol.rot.symmetry import symmetrize
from richmol.rot.J import Jxx, Jyy, Jzz, Jxy, Jyx, Jxz, Jzx, Jyz, Jzy, JJ, Jp, Jm
from scipy.sparse import csr_matrix
from richmol.field import CarTens
import itertools


_hamiltonians = dict() # Watson-type effective Hamiltonians
_constants = dict()    # parameters of effective Hamiltonians

class dummyMolecule:
    const = []
    def __getattr__(self, name):
        self.const.append(name)
        return 1

def register_ham(func):
    """Decorator function to register Hamiltonian function and a set of molecular
    attributes it is dependent on
    """
    bas = SymtopBasis(0)
    mol = dummyMolecule()
    func(Jzz(bas), mol, bas)
    _hamiltonians[func.__name__] = func
    _constants[func.__name__] = mol.const
    return func


def solve(mol, Jmin=0, Jmax=10, only={}, verbose=False):
    """Solves rotational eigenvalue problem

    Args:
        mol : Molecule
            Molecular parameters.
        Jmin, Jmax : int
            Min and max values of J quantum number.
        only : dict
            Contains various state filters to control the basis set and solution:
            only['m'][J] for J in range(Jmin, Jmax+1) contains list of m quantum
                numbers, by default, m runs form -J to +J
            only['sym'][J] for J in range(Jmin, Jmax+1) contains lits of symmetries
                for which solution to be obtained, by default, all symmetries
                are considered
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

        # m-quanta filter
        if 'm' in only:
            try:
                try:
                    m_list = list(only['m'][J])
                except IndexError:
                    raise IndexError(f"bad argument, only['m'][J] for J = {J} looks like an empty list") from None
            except KeyError:
                raise KeyError(f"bad argument, only['m'][J] for J = {J} does not exist") from None
        else:
            m_list = []

        # symmetry-adapted basis sets
        symbas = symmetrize(SymtopBasis(J, linear=mol.linear(), m_list=m_list), sym=mol.sym)

        sol[J] = {}
        for sym,bas in symbas.items():

            # symmetry filter
            if 'sym' in only:
                try:
                    if sym not in list(only['sym'][J]):
                        continue
                except KeyError:
                    raise KeyError(f"bad argument, only['sym'][J] for J = {J} does not exist") from None

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

    :math:`H = H_{rr} - \Delta_{J} * J^{4} - \Delta_{JK} * J^{2} * J_{z}^{2} - \Delta_{K} * J_{z}^{4}`
    :math:`-\\frac{1}{2} * [ \delta_J_{1} * J^{2} + \delta_{k} * J_{z}^{2}, J_{+}^{2} + J_{-}^{2} ]_{+}`
    :math:`+ H_{J} * J^{6} + H_{JK} * J^{4} * J_{z}^{2} + H_{KJ} * J^{2} * J_{z}^{4} + H_{K} * J_{z}^{6}`
    :math:`+ \\frac{1}{2} * [ \phi_{J} * J^{4} + \phi_{JK} * J^{2} * J_{z}^{2} + \phi_{K} * J_{z}^{4}, J_{+}^{2} + J_{-}^{2} ]_{+}`
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
    expr['DeltaJ']  = (-1) * J4
    expr['DeltaJK'] = (-1) * J2Jz2
    expr['DeltaK']  = (-1) * Jz4
    expr['deltaJ'] = (-0.5) * (J2*(Jp2+Jm2)+Jp2*J2+Jm2*J2)
    expr['deltaK'] = (-0.5) * (Jz2*(Jp2+Jm2)+Jp2*Jz2+Jm2*Jz2)
    expr['HJ']  = J6
    expr['HJK'] = J4Jz2
    expr['HKJ'] = J2Jz4
    expr['HK']  = Jz6
    expr['phiJ'] = (0.5) * (J4*(Jp2+Jm2)+Jp2*J4+Jm2*J4)
    expr['phiJK'] = (0.5) * (J2Jz2*(Jp2+Jm2)+Jp2*J2Jz2+Jm2*J2Jz2)
    expr['phiK'] = (0.5) * (Jz4*(Jp2+Jm2)+Jp2*Jz4+Jm2*Jz4)

    H = H0

    for key, val in expr.items():
        try:
            const = getattr(mol, key)
            if verbose is True:
                print(f"add 'watson_s' term '{key}' = {const}")
            H = H + const * val
        except AttributeError:
            pass

    return H


@register_ham
def watson_s(H0, mol, bas, verbose=False):
    """Watson-type asymmetric top Hamiltonian in S standard reduced form
    (J. K. G. Watson in "Vibrational Spectra and Structure" (Ed: J. Durig) Vol 6 p 1, Elsevier, Amsterdam, 1977).

    :math:`H = H_{rr} - \Delta_{J} * J^{4} - \Delta_{JK} * J^{2} * J_{z}^{2} - \Delta_{K} * J_{z}^{4}`
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

    expr['DeltaJ']  = (-1) * J4
    expr['DeltaJK'] = (-1) * J2Jz2
    expr['DeltaK']  = (-1) * Jz4
    expr['d1']  = J2 * (Jp2 + Jm2)
    expr['d2']  = (Jp4 + Jm4)
    expr['HJ']  = J6
    expr['HJK'] = J4Jz2
    expr['HKJ'] = J2Jz4
    expr['HK']  = Jz6
    expr['h1']  = J4 * (Jp2 + Jm2)
    expr['h2']  = J2 * (Jp4 + Jm4)
    expr['h3']  = (Jp6 + Jm6)

    H = H0

    for key, val in expr.items():
        try:
            const = getattr(mol, key)
            if verbose is True:
                print(f"add 'watson_s' term '{key}' = {const}")
            H = H + const * val
        except AttributeError:
            pass

    return H

