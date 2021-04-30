import numpy as np
from richmol.rot.basis import SymtopBasis
from richmol.rot.symmetry import symmetrize
from richmol.rot.J import Jxx, Jyy, Jzz, Jxy, Jyx, Jxz, Jzx, Jyz, Jzy, JJ, Jp, Jm
from collections import UserDict
import inspect
import h5py
import datetime
import time
from richmol import json
from collections.abc import Mapping


_hamiltonians = dict() # Watson-type effective Hamiltonians
_constants = dict()    # parameters of effective Hamiltonians

class dummyMolecule:
    const = []
    def __getattr__(self, name):
        self.const.append(name)
        return 1

def register_ham(func):
    """Decorator function to register Hamiltonian function and a set of molecular
    attributes it depends on
    """
    bas = SymtopBasis(0)
    mol = dummyMolecule()
    func(Jzz(bas), mol, bas)
    _hamiltonians[func.__name__] = func
    _constants[func.__name__] = mol.const
    return func


class Solution(UserDict):
    """Field-free rotational solutions for different of J and symmetry
    An object of this class is returned by :py:func`solve` function

    This is a subclass of :py:class`collections.UserDict`, use it as dictionary,
    i.e., Solution[J][sym] -> :py:class`richmol.rot.basis.SymtopBasis`

    Methods:
        store(filename, name=None, comment=None, replace=False):
            Stores object into HDF5 file
        read(filename, name=None):
            Reads object from HDF5 file
    """
    def __init__(self, val=None):
        if val is None:
            val = {}
        super().__init__(val)

    def __setitem__(self, item, value):
        super().__setitem__(item, value)

    def __delitem__(self, item):
        super().__delitem__(item)

    def store(self, filename, name=None, comment=None, replace=False):
        """Stores object into HDF5 file

        Args:
            filename : str
                Name of HDF5 file
            name : str
                Name of the data group, by default name of the variable is used
            comment : str
                User comment
            replace : bool
                If True, the existing data set will be replaced
        """
        if name is None:
            name = retrieve_name(self)

        with h5py.File(filename, 'a') as fl:
            if name in fl:
                if replace is True:
                    del fl[name]
                else:
                    raise RuntimeError(f"found existing dataset with name '{name}' in file " + \
                        f"'{filename}', use replace=True to replace it") from None

            group = fl.create_group(name)
            group.attrs["__class_name__"] = self.class_name()

            # description of object
            doc = self.__doc__
            if doc is None:
                doc = ""

            # add date/time
            date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            doc += "\nstored in file " + filename + " date: " + date.replace('\n','')

            # add user comment
            if comment is not None:
                doc += "\ncomment: " + " ".join(elem for elem in comment.split())

            group.attrs['__doc__'] = doc

            # store attributes

            attrs = list(set(vars(self).keys()))
            for attr in attrs:
                val = getattr(self, attr)
                try:
                    group.attrs[attr] = val
                except TypeError:
                    jd = json.dumps(val)
                    group.attrs[attr + "__json"] = jd


    def read(self, filename, name=None):
        """Reads object from HDF5 file

        Args:
            filename : str
                Name of HDF5 file
            name : str
                Name of the data group, if None, the first group with matching
                "__class_name__"  attribute will be loaded
        """
        with h5py.File(filename, 'a') as fl:

            # select datagroup

            if name is None:
                # take the first datagroup that has the same type
                groups = [group for group in fl.values() if "__class_name__" in group.attrs.keys()]
                group = next((group for group in groups if group.attrs["__class_name__"] == self.class_name()), None)
                if group is None:
                    raise TypeError(f"file '{filename}' has no dataset of type '{self.class_name()}'") from None
            else:
                # find datagroup by name
                try:
                    group = fl[name]
                except KeyError:
                    raise KeyError(f"file '{filename}' has no dataset with the name '{name}'") from None
                # check if self and datagroup types match
                class_name = group.attrs["__class_name__"]
                if class_name != self.class_name():
                    raise TypeError(f"dataset with the name '{name}' in file '{filename}' " + \
                        f"has different type: '{class_name}'") from None

            # read attributes

            attr = {}
            for key, val in group.attrs.items():
                if key.find('__json') == -1:
                    attr[key] = val
                else:
                    jl = json.loads(val)
                    key = key.replace('__json', '')
                    attr[key] = jl
            self.__dict__.update(attr)


    def class_name(self):
        """Generates '__class_name__' attribute for the solution data group in HDF5 file"""
        return self.__module__ + '.' + self.__class__.__name__


def solve(mol, Jmin=0, Jmax=10, verbose=False, **kwargs):
    """Solves rotational eigenvalue problem

    Args:
        mol : Molecule
            Molecular parameters.
        Jmin, Jmax : int
            Min and max values of J quantum number.
        verbose : bool
            If True, some log will be printed.

    Kwargs:
        mmin, mmax : int or float
            Min and max values of quantum number m of Z projection of total
            angular momentum
        mlist : list
            List of m values, if present, overrides mmin and mmax
        mdict : dict
            Dictionary mdict[J] -> list, contains list of m quantum numbers
            for different values of J, if present, overrides mlist
        symlist : list
            List of state symmetries
        symdict : dict:
            Dictionary symdict[J] -> list, contains list of symmetries
            for different values of J, if present, overrides symlist

    Returns:
        sol : Solution
            Wave functions in symmetric-top basis (SymtopBasis class) for different values
            of J=Jmin..Jmax and different symmetries, i.e., sol[J][sym] -> SymtopBasis.
    """
    assert(Jmax >= Jmin), f"Jmax = {Jmax} < Jmin = {Jmin}"
    Jlist = [J for J in range(Jmin, Jmax+1)]

    # m-quanta filters, mdict, mlist, mmin and mmax

    if 'mdict' in kwargs:
        mdict = {J : [m for m in kwargs['mdict'][J]] for J in list(Jlist & kwargs['mdict'].keys)}
        # for J values that are not present in kwargs['mdict'] set m ranges -J..J
        j_out = [J for J in Jlist if J not in mdict]
        mdict += {J : [m for m in np.linalg(-J, J, 2*J+1)] for J in j_out}
    elif 'mlist' in kwargs:
        mdict = {J : [m for m in kwargs['mlist'] if abs(m)<=J] for J in Jlist}
    else:
        mdict = dict()
        mmin = None
        mmax = None
        if 'mmin' in kwargs:
            mmin = kwargs['mmin']
        if 'mmax' in kwargs:
            mmax = kwargs['mmax']
        if mmin is not None and mmax is not None:
            assert (mmin <= mmax), f"'mmin' = {mmin} > 'mmax' = {mmax}"
        for J in Jlist:
            if mmin is None:
                m1 = -J
            else:
                m1 = max([-J, mmin])
            if mmax is None:
                m2 = J
            else:
                m2 = min([J, mmax])
            if m1>m2: continue
            mdict[J] = [m for m in np.linspace(m1, m2, m2-m1+1)]

    # compute solutions for different J and symmetries

    sol = Solution()

    for J in Jlist:

        # symmetry-adapted basis sets
        symbas = symmetrize(SymtopBasis(J, linear=mol.linear(), m_list=mdict[J]),
                            sym=mol.sym)

        sol[J] = {}

        for sym, bas in symbas.items():

            # symmetry filter
            if 'symdict' in kwargs:
                try:
                    if sym.lower() not in [elem.lower() for elem in kwargs['symdict'][J]]:
                        continue
                except KeyError:
                    pass
            elif 'symlist' in kwargs:
                if sym not in kwargs['symlist']:
                    continue

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


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
