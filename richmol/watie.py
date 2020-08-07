"""Tools for computing rigid-molecule rotational energy levels, wave functions, and matrix elements
of various rotation-dependent operators, such as laboratory-frame Cartesian tensor operators.
"""
import numpy as np
import math
import sys
import dis
import os
from mendeleev import element
import re
import inspect
import copy
from ctypes import CDLL, c_double, c_int, POINTER, RTLD_GLOBAL
import warnings
from richmol.pywigxjpf import wig_table_init, wig_temp_init, wig3jj, wig_temp_free, wig_table_free
from functools import wraps


bohr_to_angstrom = 0.529177249    # converts distances from atomic units to Angstrom
planck = 6.62606896e-27           # Plank constant in erg a second
avogno = 6.0221415e+23            # Avogadro constant
vellgt = 2.99792458e+10           # Speed of light constant in centimetres per second
boltz = 1.380658e-16              # Boltzmann constant in erg per Kelvin
small = abs(np.finfo(float).eps)
large = abs(np.finfo(float).max)


# load Fortran library symtoplib
symtoplib_path = os.path.join(os.path.dirname(__file__), 'symtoplib')
fsymtop = np.ctypeslib.load_library('symtoplib', symtoplib_path)

# allow for repetitions of warning for the same source location
warnings.simplefilter('always', UserWarning)


class settings():
    """ Sets some control parameters """
    assign_nprim = 1 # number of primitive basis contributions printed in the state assignment (e.g. in Richmol states file)
    assign_ndig_c2 = 4 # number of digits printed for the assignment coefficient |c|^2



def atom_data_from_label(atom_label):
    """Given atom label, returns its properties, e.g. mass. Combine atom labels with integer mass
    numbers to specify different isotopologues, e.g., 'H2' (deuterium), 'C13', 'N15', etc.
    """
    r = re.compile("([a-zA-Z]+)([0-9]+)")
    m = r.match(atom_label)
    if m is None:
        atom = atom_label
        mass_number = 0
    else:
        atom = m.group(1)
        mass_number = int(m.group(2))
    elem = element(atom)
    if mass_number==0:
        mass_number = int(round(elem.mass,0))
    try:
        ind = [iso.mass_number for iso in elem.isotopes].index(mass_number)
    except ValueError:
        raise ValueError(f"Isotope '{mass_number}' of the element '{atom}' is not found in mendeleev " \
                +f"database") from None
    mass = [iso.mass for iso in elem.isotopes][ind]
    return {"mass":mass}



class RigidMolecule():

    @property
    def XYZ(self):
        try:
            x = self.atoms
        except AttributeError:
            raise AttributeError(f"'{retrieve_name(self)}.XYZ' was not initialized") from None
        res = self.atoms.copy()
        try:
            res['xyz'] = np.dot(res['xyz'], np.transpose(self.frame_rotation))
        except AttributeError:
            pass
        return res


    @XYZ.setter
    def XYZ(self, arg):

        to_angstrom = 1 # default distance units are Angstrom
        xyz = []
        mass = []
        label = []

        if isinstance(arg, str):

            # read from XYZ file

            fl = open(arg, 'r')
            line = fl.readline()
            natoms = float(line.split()[0])
            comment = fl.readline()
            for line in fl:
                w = line.split()
                atom_label = w[0]
                try:
                    x,y,z = (float(ww) for ww in w[1:])
                except ValueError:
                    raise ValueError(f"Atom specification '{atom_label}' in the XYZ file {arg} " \
                            +f"is not followed by the three floating-point values of x, y, and z " \
                            +f"atom coordinates") from None
                atom_mass = atom_data_from_label(atom_label.upper())["mass"]
                xyz.append([x,y,z])
                mass.append(atom_mass)
                label.append(atom_label)
            fl.close()

        elif isinstance(arg, (list, tuple)):

            # read from input iterable

            for ielem,elem in enumerate(arg):
                if isinstance(elem, str):
                    if elem[:4].lower()=="bohr":
                        to_angstrom = bohr_to_angstrom
                    elif elem[:4].lower()=="angs":
                        to_angstrom = 1
                    else:
                        atom_label = elem
                        atom_mass = atom_data_from_label(atom_label.upper())["mass"]
                        try:
                            x,y,z = (float(val) for val in arg[ielem+1:ielem+4])
                        except ValueError:
                            raise ValueError(f"Atom specification '{atom_label}' is not followed " \
                                    +f"by the three floating-point values of x, y, and z " \
                                    +f"atom coordinates") from None
                        xyz.append([float(val)*to_angstrom for val in (x,y,z)])
                        mass.append(atom_mass)
                        label.append(atom_label)
        else:
            raise TypeError(f"Unsupported argument type '{type(arg)}' for atoms' specification") from None

        self.atoms = np.array( [(lab, mass, cart) for lab,mass,cart in zip(label,mass,xyz)], \
                               dtype=[('label','U10'),('mass','f8'),('xyz','f8',(3))] )


    @property
    def tensor(self):
        """ Returns a dict of all initialised tensors """
        try:
            x = self.tens
        except AttributeError:
            raise AttributeError(f"'{retrieve_name(self)}.tensor' was not initialized") from None
        tens = copy.deepcopy(self.tens)
        try:
            sa = "abcdefgh"
            si = "ijklmnop"
            for name,array in tens.items():
                ndim = array.ndim
                if ndim>len(sa):
                    raise ValueError(f"Number of dimensions for tensor '{name}' is equal to {ndim} " \
                            +f"and it exceeds the maximum {len(sa)}") from None
                key = "".join(sa[i]+si[i]+"," for i in range(ndim)) \
                    + "".join(si[i] for i in range(ndim)) + "->" \
                    + "".join(sa[i] for i in range(ndim))
                rot_mat = [self.frame_rotation for i in range(ndim)]
                tens[name] = np.einsum(key, *rot_mat, array)
        except AttributeError:
            pass
        return tens


    @tensor.setter
    def tensor(self, arg):
        """Defines Cartesian tensor in the molecule-fixed frame

        Examples:
            tensor = ("mu", [0.5, -0.1, 0]) to add a permanent dipole moment vector.
            tensor = ("my_alpha", [[10,0,0],[0,20,0],[0,0,30]]) to add a rank-2 tensor, such as,
                for example, polarizability.
        """
        # check if input is (name, tensor)
        try:
            name, tens = arg
            name = name.strip()
        except ValueError:
            raise ValueError(f"Pass an iterable with two items, tensor = ('name', tensor)") from None
        # check if name and tensor have proper types
        if not isinstance(name, str):
            raise TypeError(f"Unsupported argument type '{type(name)}' for tensor name, must be 'str'") from None
        if isinstance(tens, (tuple, list)):
            tens = np.array(tens)
        elif isinstance(tens, (np.ndarray,np.generic)):
            pass
        else:
            raise TypeError(f"Unsupported argument type '{type(tens)}' for tensor values, " \
                    +f"must be one of: 'list', 'numpy.ndarray'") from None
        # check if name and tensor have proper values
        if "," in name or len(name)==0:
            raise ValueError(f"Illegal tensor name '{name}', it must not contain commas and must not be empty")
        if not all(dim==3 for dim in tens.shape):
            raise ValueError(f"(Cartesian) tensor has bad shape: '{tens.shape}' != {[3]*tens.ndim}") from None
        if np.all(np.abs(tens)<small):
            raise ValueError(f"Tensor has all its elements equal to zero") from None
        if np.any(np.abs(tens)>large*0.1):
            raise ValueError(f"Tensor has too large values of its elements") from None
        if np.any(np.isnan(tens)):
            raise ValueError(f"Tensor has some values of its elements equal to NaN") from None
        # save tensor
        try:
            x = self.tens
        except AttributeError:
            self.tens = {}
        if name in self.tens:
            raise ValueError(f"Tensor with name '{name}' already exists") from None
        self.tens[name] = tens


    @property
    def frame(self):
        """ Returns type of molecular frame (str) and frame rotation matrix (array (3,3)) """
        try:
            rotmat = self.frame_rotation
            frame_type = self.frame_type
        except AttributeError:
            rotmat = np.eye(3, dtype=np.float64)
            frame_type = "I"
        return frame_type, rotmat


    @frame.setter
    def frame(self, arg):
        """Defines rotation of molecular frame

        Cartesian coordinates of atoms and all Cartesian tensors will be rotated to a new frame.

        Examples:
            frame = "pas" will rotate to a principal axes system with x,y,z = a,b,c.
            frame = "tens_name" will rotate to a principal axes system of a 3x3 tensor with the name 
                "tens_name", this tensor must be initialized before using a command
                tensor = (tens_name, [[x,x,x],[x,x,x],[x,x,x]]).
            frame = "zxy" will permute axes x-->z, y-->x, and y-->z.
            frame = "zxy,pas" will rotate to "pas" and permute x-->z, y-->x, and y-->z.
        """
        if isinstance(arg, str):

            try:
                x = self.frame_rotation
            except AttributeError:
                self.frame_rotation = np.eye(3, dtype=np.float64)

            for fr in reversed([v.strip() for v in arg.split(',')]):

                assert (len(fr)>0), f"Illegal frame type specification: '{arg}'"

                if fr.lower()=="pas":
                    # principal axes system
                    try:
                        diag, rotmat = np.linalg.eigh(self.imom())
                    except np.linalg.LinAlgError:
                        raise RuntimeError("Eigenvalues did not converge") from None
                    self.frame_rotation = np.dot(np.transpose(rotmat), self.frame_rotation)

                elif "".join(sorted(fr.lower()))=="xyz":
                    # axes permutation
                    ind = [("x","y","z").index(s) for s in list(fr.lower())]
                    rotmat = np.zeros((3,3), dtype=np.float64)
                    for i in range(3):
                        rotmat[i,ind[i]] = 1.0
                    self.frame_rotation = np.dot(rotmat, self.frame_rotation)

                else:
                    # axes system defined by to-diagonal rotation of arbitrary rank-2 (3x3) tensor
                    # the tensor must be initialized before, with the name matching fr
                    try:
                        tens = self.tensor[fr]
                    except KeyError:
                        raise KeyError(f"Tensor '{fr}' was not initialised") from None
                    if tens.ndim!=2:
                        raise ValueError(f"Tensor '{fr}' has inappropriate rank: {tens.ndim} " \
                                +f"is not equal to 2") from None
                    if np.any(np.abs(tens-tens.T)>small*10.0):
                        raise ValueError(f"Tensor '{fr}' is not symmetric") from None
                    try:
                        diag, rotmat = np.linalg.eigh(tens)
                    except np.linalg.LinAlgError:
                        raise RuntimeError("Eigenvalues did not converge") from None
                    self.frame_rotation = np.dot(np.transpose(rotmat), self.frame_rotation)

        else:
            raise TypeError(f"Unsupported argument type '{type(arg)}' for frame specification, " \
                    +f"must be 'str'") from None

        # update string that keeps track of all frame rotations
        try:
            self.frame_type += "," + arg
        except AttributeError:
            self.frame_type = arg


    @property
    def B(self):
        """ Returns Bx, By, Bz rotational constants in units of cm^-1 """
        imom = self.imom()
        tol = 1e-12
        if np.any(np.abs(np.diag(np.diag(imom))-imom)>tol):
            raise RuntimeError("Can't compute rotational constants for the current frame = " \
                    +f"'{self.frame_type}', inertia tensor is not diagonal = {imom}, " \
                    +f"max offdiag = {np.max(np.abs(np.diag(np.diag(imom))-imom))}") from None
        convert_to_cm = planck * avogno * 1e+16 / (8.0 * np.pi * np.pi * vellgt) 
        return [convert_to_cm/val for val in np.diag(imom)]


    @B.setter
    def B(self, val):
        raise AttributeError(f"You can't set {retrieve_name(self)}.B") from None


    @property
    def kappa(self):
        """ Returns asymmtery parameter kappa = (2*B-A-C)/(A-C) """
        A, B, C = reversed(sorted(self.B))
        return (2*B-A-C)/(A-C)


    @kappa.setter
    def kappa(self, val):
        raise AttributeError(f"You can't set {retrieve_name(self)}.kappa") from None


    def imom(self):
        """ Inertia tensor """
        xyz = self.XYZ['xyz']
        mass = self.XYZ['mass']
        cm = np.sum([x*m for x,m in zip(xyz,mass)], axis=0)/np.sum(mass)
        xyz0 = xyz - cm[np.newaxis,:]
        imat = np.zeros((3,3), dtype=np.float64)
        natoms = xyz0.shape[0]
        # off-diagonals
        for i in range(3):
            for j in range(3):
                if i==j: continue
                imat[i,j] = -np.sum([ xyz0[iatom,i]*xyz0[iatom,j]*mass[iatom] for iatom in range(natoms) ])
        # diagonals
        imat[0,0] = np.sum([ (xyz0[iatom,1]**2+xyz0[iatom,2]**2)*mass[iatom] for iatom in range(natoms) ])
        imat[1,1] = np.sum([ (xyz0[iatom,0]**2+xyz0[iatom,2]**2)*mass[iatom] for iatom in range(natoms) ])
        imat[2,2] = np.sum([ (xyz0[iatom,0]**2+xyz0[iatom,1]**2)*mass[iatom] for iatom in range(natoms) ])
        return imat


    def gmat(self):
        """ Rotational kinetic energy matrix """
        convert_to_cm = planck*avogno*1e+16/(4.0*np.pi*np.pi*vellgt)
        xyz = self.XYZ['xyz']
        mass = self.XYZ['mass']
        natoms = xyz.shape[0]

        # Levi-Civita tensor
        levi_civita = np.zeros((3,3,3),dtype=np.float64)
        levi_civita[0,1,2] = 1
        levi_civita[0,2,1] =-1
        levi_civita[1,0,2] =-1
        levi_civita[1,2,0] = 1
        levi_civita[2,0,1] = 1
        levi_civita[2,1,0] =-1

        # rotational t-vector
        tvec = np.zeros((3,natoms*3), dtype=np.float64)
        tvec_m = np.zeros((natoms*3,3), dtype=np.float64)
        for irot in range(3):
            ialpha = 0
            for iatom in range(natoms):
                for alpha in range(3):
                    tvec[irot,ialpha] = np.dot(levi_civita[alpha,irot,:], xyz[iatom,:])
                    tvec_m[ialpha,irot] = tvec[irot,ialpha] * mass[iatom]
                    ialpha+=1

        # rotational g-matrix
        gsmall = np.dot(tvec, tvec_m)

        # invert g-matrix to obtain rotational G-matrix
        umat, sv, vmat = np.linalg.svd(gsmall, full_matrices=True)

        dmat = np.zeros((3,3), dtype=np.float64)
        no_sing = 0
        ifstop = False
        for i in range(len(sv)):
            if sv[i]>small*np.linalg.norm(gsmall):
                dmat[i,i] = 1.0/sv[i]
            else:
                no_sing += 1
                print(f"Warning: rotational kinetic energy matrix is singular, " \
                        +f"singular element index = {i}, singular value = {sv[i]}")
                if no_sing==1 and self.linear() is True:
                    print(f"this is fine for linear molecule: set 1/{sv[i]}=0")
                    dmat[i,i] = 0
                else:
                    ifstop = True
        if ifstop is True:
            raise RuntimeError(f"Rotational kinetic energy matrix g-small:\n{gsmall}\ncontains " \
                    +f"singular elements:\n{sv}\nplease check your input geometry " \
                    +f"in {retrieve_name(self)}.XYZ")

        gbig = np.dot(umat, np.dot(dmat, vmat))
        gbig *= convert_to_cm
        return gbig


    def linear(self):
        """ Returns True/False if molecule is linear/non-linear """
        xyz = self.XYZ['xyz']
        imom = self.imom()
        d, rotmat = np.linalg.eigh(imom)
        xyz2 = np.dot(xyz, rotmat)
        tol = 1e-14
        if (np.all(abs(xyz2[:,0])<tol) and np.all(abs(xyz2[:,1])<tol)) or \
            (np.all(abs(xyz2[:,0])<tol) and np.all(abs(xyz2[:,2])<tol)) or \
            (np.all(abs(xyz2[:,1])<tol) and np.all(abs(xyz2[:,2])<tol)):
            return True
        else:
            return False



class PsiTable():
    """ Basic class for operations on wavefunction coefficients """

    def __init__(self, prim, stat, coefs=None):
        """Initialize PsiTable from sets of primitive and state quanta and, if provided,
        matrix of state coefficients
        """
        if not isinstance(prim, (list, tuple, np.ndarray)):
            raise TypeError(f"Unsupported argument type '{type(prim)}'") from None
        if not isinstance(stat, (list, tuple, np.ndarray)):
            raise TypeError(f"Unsupported argument type '{type(stat)}'") from None
        try:
            x = [int(val) for elem in prim for val in elem]
        except ValueError:
            raise ValueError(f"Failed to convert element into integer")

        nprim = len(prim)
        nstat = len(stat)
        assert (nprim>0), f"Number of primitives nprim = 0"
        assert (nstat>0), f"Number of states nstat = 0"
        assert (nprim>=nstat), f"nprim < nstat: {nprim} < {nstat}"

        nelem_stat = list(set(len(elem) for elem in stat))
        if len(nelem_stat)>1:
            raise ValueError(f"Different lengths for different elements in 'stat'")
        nelem_prim = list(set(len(elem) for elem in prim))
        if len(nelem_prim)>1:
            raise ValueError(f"Different lengths for different elements in 'prim'")

        # check for duplicates in prim and stat
        if len(list(set(tuple(p) for p in prim))) != len(prim):
            raise ValueError(f"Found duplicate elements in 'prim'")
        # if len(list(set(tuple(s) for s in stat))) != len(stat):
        #     raise ValueError(f"found duplicate elements in 'stat'")

        dt = [('prim', 'i4', (nelem_prim)), ('stat', 'U10', (nelem_stat)), ('c', np.complex128, [nstat])]
        self.table = np.zeros(nprim, dtype=dt)
        self.table['prim'] = prim
        self.table['stat'][:nstat] = stat

        if coefs is not None:
            try:
                shape = coefs.shape
            except AttributeError:
                raise AttributeError(f"Unsupported argument type for coefficients matrix '{type(coefs)}'") from None
            if any(x!=y for x,y in zip(shape,[nprim,nstat])):
                raise ValueError(f"Shape of coefficients matrix = {shape} is not aligned with the " \
                        +f"number of primitives = {nprim} and number of states = {nstat}") from None
            self.table['c'][:,:] = coefs


    @classmethod
    def fromPsiTable(cls, arg):
        """ Initialize PsiTable from an argument of PsiTable type """
        if not isinstance(arg, PsiTable):
            raise TypeError(f"Unexpected type for argument '{type(arg)}', expected 'PsiTable'")
        cls = copy.deepcopy(arg)
        return cls


    def __add__(self, arg):
        try:
            x = arg.table
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'table'") from None

        if not np.array_equal(self.table['prim'], arg.table['prim']):
            raise ValueError(f"'{type(self)}' objects under sum work on different basis sets " \
                    +f"('table['prim']' attributes do not match)") from None

        if not np.array_equal(self.table['stat'], arg.table['stat']):
            raise ValueError(f"'{type(self)}' objects under sum work on different basis sets " \
                    +f"('table['stat']' attributes do not match)") from None

        nprim, nstat = self.table['c'].shape
        prim = self.table['prim']
        stat = self.table['stat'][:nstat]
        coefs = np.zeros((nprim, nstat), dtype=np.complex128)
        coefs = self.table['c'] + arg.table['c']
        return PsiTable(prim, stat, coefs)


    def __sub__(self, arg):
        try:
            x = arg.table
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'table'") from None

        if not np.array_equal(self.table['prim'], arg.table['prim']):
            raise ValueError(f"'{type(self)}' objects under subtr work on different basis sets " \
                    +f"('table['prim']' attributes do not match)") from None

        if not np.array_equal(self.table['stat'], arg.table['stat']):
            raise ValueError(f"'{type(self)}' objects under subtr work on different basis sets " \
                    +f"('table['stat']' attributes do not match)") from None

        nprim, nstat = self.table['c'].shape
        prim = self.table['prim']
        stat = self.table['stat'][:nstat]
        coefs = np.zeros((nprim, nstat), dtype=np.complex128)
        coefs = self.table['c'] - arg.table['c']
        return PsiTable(prim, stat, coefs)


    def __mul__(self, arg):
        if np.isscalar(arg):
            nprim, nstat = self.table['c'].shape
            prim = self.table['prim']
            stat = self.table['stat'][:nstat]
            coefs = self.table['c'].copy()
            coefs *= arg
        else:
            raise TypeError(f"unsupported operand type(s) for '*': '{self.__class__.__name__}' and " \
                    +f"'{arg.__class__.__name__}'") from None
        return PsiTable(prim, stat, coefs)


    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__


    def append(self, arg, del_duplicate_stat=False, del_zero_stat=False, del_zero_prim=False, tol=1e-12):
        try:
            x = arg.table
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'table'") from None

        nstat1 = self.table['c'].shape[1]
        nstat2 = arg.table['c'].shape[1]
        prim1 = [tuple(x) for x in self.table['prim']]
        stat1 = [tuple(x) for x in self.table['stat'][:nstat1]]
        prim2 = [tuple(x) for x in arg.table['prim']]
        stat2 = [tuple(x) for x in arg.table['stat'][:nstat2]]

        if len(stat1[0]) != len(stat2[0]):
            raise ValueError(f"two tables in append have different length of the elements in 'stat'")
        if len(prim1[0]) != len(prim2[0]):
            raise ValueError(f"two tables in append have different length of the elements in 'prim'")

        prim = list(set(prim1 + prim2))
        stat = stat1 + stat2
        coefs = np.zeros((len(prim),len(stat)), dtype=np.complex128)
        nstat = len(stat)

        for i,p in enumerate(prim):
            try:
                i1 = prim1.index(tuple(p))
                coefs[i,:nstat1] += self.table['c'][i1,:]
            except ValueError:
                pass
            try:
                i2 = prim2.index(tuple(p))
                coefs[i,nstat1:nstat] += arg.table['c'][i2,:]
            except ValueError:
                pass
        if del_zero_stat==True:
            prim, stat, coefs = self.del_zero_stat(prim, stat, coefs, tol)
        if del_zero_prim==True:
            prim, stat, coefs = self.del_zero_prim(prim, stat, coefs, tol)
        if del_duplicate_stat==True:
            prim, stat, coefs = self.del_duplicate_stat(prim, stat, coefs, tol)

        # check for duplicates in 'stat'
        if len(list(set(tuple(s) for s in stat))) != len(stat):
            raise ValueError(f"two tables in append have overlapping 'stat' elements that correspond " \
                    +f"to different coefficient vectors")

        return PsiTable(prim, stat, coefs)


    def overlap(self, arg):
        """ Computes overlap < self | arg > """
        try:
            x = arg.table
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'table'") from None
        prim1 = [tuple(x) for x in self.table['prim']]
        coefs1 = self.table['c'].conj().T
        prim2 = [tuple(x) for x in arg.table['prim']]
        coefs2 = arg.table['c']
        # find overlapping primitive states in both sets
        both = list(set(prim1) & set(prim2))
        # both = set(prim1).intersection(prim2)
        if len(both)==0:
            warnings.warn(f"functions have no overlapping primitive quanta, the overlap is zero!")
        ind1 = [prim1.index(x) for x in both]
        ind2 = [prim2.index(x) for x in both]
        # dot product across overlapping primitive quanta
        return np.dot(coefs1[:,ind1], coefs2[ind2,:])


    def rotate(self, arg, stat=None):
        """Applies a unitary transformation

        arg = rotmat, here 'rotmat' is a unitary transformation matrix, the number of columns
            in 'rotmat' must be equal to the number of basis states, i.e., self.table['c'].shape[1].

        arg = (rotmat, enr), here 'rotmat' is a unitary transformation matrix (as above) and 'enr'
            is a set of associated energies. For example, if a unitary transformed basis diagonalizes
            some Hamiltonian, its eigenvalues could be stored in 'enr'.
            If 'enr' is provided, it is stored in a new attribute PsiTable.enr.

        stat (list): A user-defined assignment of unitary-transformed states.
        """
        try:
            rotmat, enr = arg
        except ValueError:
            rotmat = arg
            enr = None

        try:
            shape = rotmat.shape
        except AttributeError:
            raise AttributeError(f"Bad type for a rotation matrix '{type(rotmat)}' (use numpy array)") from None

        if enr is None:
            pass
        elif isinstance(enr, (list, tuple, np.ndarray)):
            if shape[0] != len(enr):
                raise ValueError(f"Number of elements in energy array = {len(enr)} is not aligned " \
                        +f"with the number of rows in rotation matrix = {shape[0]}") from None
        else:
            raise ValueError(f"Bad type for a set of associated energies '{type(enr)}' (use list, tuple, " \
                    +f"or numpy array)") from None

        nstat = self.table['c'].shape[1]
        if shape[1] != nstat:
            raise ValueError(f"Number of columns in rotation matrix = {shape[1]} is not aligned with " \
                    +f"the number of basis states = {nstat}") from None

        if np.all(np.abs(rotmat)<small):
            raise ValueError(f"Rotation matrix has all its elements equal to zero") from None
        if np.any(np.abs(rotmat)>large*0.1):
            raise ValueError(f"Rotation matrix has too large values of its elements") from None
        if np.any(np.isnan(rotmat)):
            raise ValueError(f"Rotation matrix has some values of its elements equal to NaN") from None

        coefs = np.dot(self.table['c'][:,:], rotmat.T)

        # state assignments
        if stat is None:
            stat = []
            ndig = settings.assign_ndig_c2 # number of digits in |c|^2 to be kept for assignment
            c2_form = "%"+str(ndig+3)+"."+str(ndig)+"f"
            for v in rotmat:
                n = settings.assign_nprim # number of primitive states to be used for assignment
                ind = (-abs(v)**2).argsort()[:n]
                elem_stat = self.table['stat'][ind]
                c2 = [c2_form%abs(v[i])**2 for i in ind]
                ll = [ elem for i in range(len(ind)) for elem in list(elem_stat[i])+[c2[i]] ]
                stat.append(ll)
        elif len(stat) != rotmat.shape[0]:
            raise ValueError(f"Number of elements in state assignment = {len(stat)} is not aligned " \
                    +f"with the number of rows in rotation matrix = {rotmat.shape[0]}") from None
        prim = [elem for elem in self.table['prim']]
        res = PsiTable(prim, stat, coefs)

        if enr is not None:
            res.enr = np.array(enr, dtype=np.float64)

        return res


    def del_zero_stat(self, prim=None, stat=None, coefs=None, tol=1e-12):
        """ Deletes states with zero coefficients """
        if all(x is None for x in (prim,stat,coefs)):
            prim = self.table['prim'].copy()
            stat = self.table['stat'].copy()
            coefs = self.table['c'].copy()
            freturn = lambda prim, stat, coefs: PsiTable(prim, stat, coefs)
        elif all(x is not None for x in (prim,stat,coefs)):
            freturn = lambda prim, stat, coefs: (prim, stat, coefs)
        else:
            raise ValueError(f"expecting either all 'prim', 'stat', and 'coefs' arguments " \
                    +f"to be defined or none of them")
        nstat = coefs.shape[1]
        ind = [istat for istat in range(nstat) if all(abs(val)<tol for val in coefs[:,istat])]
        coefs2 = np.delete(coefs, ind, 1)
        stat2 = np.delete(stat[:nstat], ind, 0)
        prim2 = [elem for elem in prim]
        if len(stat2)==0:
            return None # somehow deleted all states
        return freturn(prim2, stat2, coefs2)


    def del_zero_prim(self, prim=None, stat=None, coefs=None, tol=1e-12):
        """ Deletes primitives that are not coupled by states
        """
        if all(x is None for x in (prim,stat,coefs)):
            prim = self.table['prim'].copy()
            stat = self.table['stat'].copy()
            coefs = self.table['c'].copy()
            freturn = lambda prim, stat, coefs: PsiTable(prim, stat, coefs)
        elif all(x is not None for x in (prim,stat,coefs)):
            freturn = lambda prim, stat, coefs: (prim, stat, coefs)
        else:
            raise ValueError(f"expecting either all 'prim', 'stat', and 'coefs' arguments " \
                    +f"to be defined or none of them")
        nprim = coefs.shape[0]
        nstat = coefs.shape[1]
        ind = [iprim for iprim in range(nprim) if all(abs(val)<tol for val in coefs[iprim,:])]
        coefs2 = np.delete(coefs, ind, 0)
        prim2 = np.delete(prim, ind, 0)
        stat2 = [elem for elem in stat[:nstat]]
        if len(prim2)==0:
            return None # somehow deleted all primitives
        return freturn(prim2, stat2, coefs2)


    def del_duplicate_stat(self, prim=None, stat=None, coefs=None, tol=1e-12):
        """ Deletes duplicate states """
        if all(x is None for x in (prim,stat,coefs)):
            prim = self.table['prim'].copy()
            stat = self.table['stat'].copy()
            coefs = self.table['c'].copy()
            freturn = lambda prim, stat, coefs: PsiTable(prim, stat, coefs)
        elif all(x is not None for x in (prim,stat,coefs)):
            freturn = lambda prim, stat, coefs: (prim, stat, coefs)
        else:
            raise ValueError(f"expecting either all 'prim', 'stat', and 'coefs' arguments " \
                    +f"to be defined or none of them")
        nstat = coefs.shape[1]
        ind = []
        for istat in range(nstat):
            ind += [jstat for jstat in range(istat+1,nstat) \
                   if all(abs(val1-val2)<tol for val1,val2 in zip(coefs[:,istat],coefs[:,jstat]))]
        coefs2 = np.delete(coefs, ind, 1)
        stat2 = np.delete(stat[:nstat], ind, 0)
        prim2 = [elem for elem in prim]
        if len(stat2)==0:
            return None # somehow deleted all states
        return freturn(prim2, stat2, coefs2)



class PsiTableMK():
    """ Basic class for operations on rotational wavefunction, based on PsiTable,
    keeps separately J,k and J,m spaces of quantum numbers
    """

    def __init__(self, psik, psim):
        if not isinstance(psik, PsiTable):
            raise TypeError(f"unexpected type for argument '{type(psik)}', expected 'PsiTable'")
        if not isinstance(psim, PsiTable):
            raise TypeError(f"unexpected type for argument '{type(psim)}', expected 'PsiTable'")
        # nstat = psim.table['c'].shape[1]
        # self.m = PsiTable(psim.table['prim'], psim.table['stat'][:nstat], psim.table['c'])
        # nstat = psik.table['c'].shape[1]
        # self.k = PsiTable(psik.table['prim'], psik.table['stat'][:nstat], psik.table['c'])
        self.m = PsiTable.fromPsiTable(psim)
        self.k = PsiTable.fromPsiTable(psik)


    def __add__(self, arg):
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        res_m = self.m + arg.m
        res_k = self.k + arg.k
        return PsiTableMK(res_k, res_m)


    def __sub__(self, arg):
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        res_m = self.m - arg.m
        res_k = self.k - arg.k
        return PsiTableMK(res_k, res_m)


    def __mul__(self, arg):
        if np.isscalar(arg):
            res_m = self.m
            res_k = self.k * arg
        else:
            raise TypeError(f"unsupported operand type(s) for '*': '{self.__class__.__name__}' and " \
                    +f"'{arg.__class__.__name__}'") from None
        return PsiTableMK(res_k, res_m)

    __rmul__ = __mul__


    def append(self, arg, del_duplicate_stat=False, del_zero_stat=False, del_zero_prim=False, tol=1e-12):
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        res_m = self.m.append(arg.m, del_duplicate_stat, del_zero_stat, del_zero_prim, tol)
        res_k = self.k.append(arg.k, del_duplicate_stat, del_zero_stat, del_zero_prim, tol)
        return PsiTableMK(res_k, res_m)


    def overlap(self, arg):
        howmany = expecting(offset=2)
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        ovlp_m = self.m.overlap(arg.m)
        ovlp_k = self.k.overlap(arg.k)
        if howmany == 1:
            return ovlp_k
        elif howmany > 1:
            return ovlp_k, ovlp_m


    def overlap_k(self, arg):
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        ovlp_k = self.k.overlap(arg.k)
        return ovlp_k


    def overlap_m(self, arg):
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        ovlp_m = self.m.overlap(arg.m)
        return ovlp_m


    def rotate(self, krot=None, mrot=None, kstat=None, mstat=None):
        if mrot is not None:
            res_m = self.m.rotate(mrot, mstat)
        else:
            res_m = self.m
        if krot is not None:
            res_k = self.k.rotate(krot, kstat)
        else:
            res_k = self.k
        return PsiTableMK(res_k, res_m)


    def store_richmol(self, name, append=False):
        """Stores state energies and assignments in Richmol energies file

        Args:
            name (str): Name of file to store energies.
            append (str): If True, the energies of states will be appended to existing file.
        """
        Jk = list(set(J for J in self.k.table['prim'][:,0]))
        Jm = list(set(J for J in self.m.table['prim'][:,0]))
        if len(Jk)>1 or len(Jm)>1:
            raise ValueError(f"Multiple values of J quanta = {Jk} and {Jm} in k- and m-parts") from None
        else:
            if Jk[0] != Jm[0]:
                raise ValueError(f"Non-equal values of J quanta = {Jk[0]} and {Jm[0]} in k- and m-parts") from None
            J = Jk[0]

        nstat = self.k.table['c'].shape[1]
        assign = self.k.table['stat'][:nstat]

        try:
            enr = self.k.enr[:nstat]
        except AttributeError:
            raise AttributeError(f"States set has no associated energies") from None

        try:
            sym = self.k.sym[:nstat]
        except AttributeError:
            sym = ["A" for i in range(nstat)]

        if append==True:
            mode = "a+"
        else:
            mode = "w"

        with open(name, mode) as fl:
            for istat in range(nstat):
                id = istat + 1
                fl.write(" %3i"%J + " %6i"%id + " %4s"%sym[istat] + "  1" + " %20.12f"%enr[istat] \
                        + " ".join(" %s"%elem for elem in assign[istat]) + "\n")



class SymtopBasis(PsiTableMK):
    """Basis of symmetric top functions for selected J

    Args:
        J (int): Quantum number of the rotational angular momentum.
        linear (bool): set True if molecule is linear, in this case quantum number k is kept at zero.
    """

    def __init__(self, J, linear=False):

        try:
            self.J = int(round(J))
        except TypeError:
            raise TypeError(f"J = '{J}' is not a number") from None
        assert (self.J>=0), f"J = {J} is smaller than zero"

        # generate keys (j,k) for columns representing primitive functions
        if linear:
            prim = [(int(J),0)]
        else:
            prim = [(int(J),int(k)) for k in range(-J,J+1)]

        # generate keys (j,k,tau) for rows representing symmetrized functions
        if linear:
            bas = [(J,0,np.fmod(J, 2))]
        else:
            bas = []
            for k in range(0,J+1):
                if k==0:
                    tau = [int(math.fmod(J, 2))]
                else:
                    tau = [0,1]
                for t in tau:
                    bas.append( (int(J),int(k),int(t)) )

        assert (len(prim)==len(bas)), "len(prim)!=len(bas)"

        # generate Wang-type linear combinations
        coefs = np.zeros((len(prim),len(bas)), dtype=np.complex128)
        for ibas,(J,k,tau) in enumerate(bas):
            c, kval = self.wang_coefs(J, k, tau)
            for kk,cc in zip(kval,c):
                iprim = prim.index((J,kk))
                coefs[iprim,ibas] = cc

        # generate m-quanta
        prim_m = [(int(J),int(m)) for m in range(-J,J+1)]
        coefs_m = np.eye(len(prim_m), dtype=np.complex128)

        # initialize
        PsiTableMK.__init__(self, PsiTable(prim, bas, coefs), PsiTable(prim_m, prim_m, coefs_m))


    def wang_coefs(self, j, k, tau):
        """Wang's symmetrization coefficients c1 and c2 for symmetric-top function in the form
        |J,k,tau> = c1|J,k> + c2|J,-k>

        Args:
            j, k, tau (int): J, k, and tau quantum numbers, where k can take values between 0 and J
                and tau=0 or 1 is parity defined as (-1)^tau.

        Returns:
            coefs (list): Wang's symmetrization coefficients, coefs=[c1,c2] for k>0 and coefs=[c1] for k=0.
            kval (list): List of k-values, kval=[k,-k] for k0 and kval=[k] for k=0.
        """
        assert (k>=0), f"k = {k} < 0"
        assert (j>=0), f"J = {j} < 0"
        assert (k<=j), f"k = {k} > J = {J}"
        assert (tau<=1 and tau>=0), f"tau = {tau} is not equal to 0 or 1"

        sigma = math.fmod(k, 3) * tau
        fac1 = pow(-1.0,sigma)/math.sqrt(2.0)
        fac2 = fac1 * pow(-1.0,(j+k))
        kval = [k, -k]
        if tau==0:
            if k==0:
                coefs = [1.0]
            elif k>0:
                coefs = [fac1, fac2]
        elif tau==1:
            if k==0:
                coefs = [complex(0.0,1.0)]
            elif k>0:
                coefs = [complex(0.0,fac1), complex(0.0,-fac2)]
        return coefs, kval



def symmetrize(arg, sym="D2", tol=1e-12):
    """Returns dictionary of symmetry-adapted objects 'arg' for different irreps (as dict keys)
    of the symmetry group defined by 'sym'

    Args:
        arg (PsiTableMK): Basis of symmetric-top functions.
        sym (str): Point symmetry group, defaults to "D2".
        tol (float): Tolerance for treating symmetrization and basis-set coefs as zero, defaults to 1e-12.
    """
    try:
        x = arg.k
    except AttributeError:
        raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
    try:
        x = arg.m
    except AttributeError:
        raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None

    try:
        sym_ = getattr(sys.modules[__name__], sym)
    except:
        raise NotImplementedError(f"symmetry '{sym}' is not implemented") from None

    # list of J quanta spanned by arg
    Jlist = list(set(j for (j,k) in arg.k.table['prim']))

    # create copies of arg for different irreps
    symmetry = sym_(Jlist[0])
    res = {sym_lab : copy.deepcopy(arg) for sym_lab in symmetry.sym_lab}
    for elem in res.values():
        elem.k.table['c'] = 0

    nstat = arg.k.table['c'].shape[1]
    prim = [tuple(x) for x in arg.k.table['prim']]

    for J in Jlist:
        symmetry = sym_(J)
        proj = symmetry.proj()

        # mapping between (J,k) quanta in arg.k.table and k=-J..J in symmetry.proj array
        ind_k = []
        ind_p = []
        for ik,k in enumerate(range(-J,J+1)):
            try:
                ind = prim.index((J,k))
                ind_k.append(ik)
                ind_p.append(ind)
            except ValueError:
                if np.any(abs(proj[:,:,ik])>tol) or np.any(abs(proj[:,ik,:])>tol):
                    raise ValueError(f"input set {retrieve_name(arg)} is missing primitive functions " \
                            +f"that are required for symmetrization, e.g., (J,k) = {(J,k)}") from None

        for irrep,sym_lab in enumerate(symmetry.sym_lab):
            pmat = np.dot(proj[irrep,:,ind_k], arg.k.table['c'][ind_p,:])
            res[sym_lab].k.table['c'][ind_p,:] = pmat[ind_k,:]

    # remove states with zero coefficients
    remove = []
    for sym_lab,elem in res.items():
        elem.k = elem.k.del_zero_stat(tol=1e-12)
        if elem.k is None:
            remove.append(sym_lab)
    for sym_lab in remove: del res[sym_lab]

    # check if the total number of states remains the same
    nstat_sym = sum(elem.k.table['c'].shape[1] for elem in res.values()) 
    if nstat_sym != nstat:
        raise RuntimeError(f"total number of states before symmetrization = {nstat} is different " \
                +f"from the total number of states summed over all irreps = {nstat_sym}")

    return res



class SymtopSymmetry():

    def __init__(self, J):

        try:
            self.J = int(round(J))
        except TypeError:
            raise TypeError(f"J = '{J}' is not a number") from None
        assert (self.J>=0), f"J = {J} is smaller than zero"

        # compute symmetrisation coefficients for symmetric-top functions

        jmin = J
        jmax = J
        npoints = self.noper
        npoints_c = c_int(npoints)
        grid = np.asfortranarray(self.euler_rotation, dtype=np.float64)

        jmin_c = c_int(jmin)
        jmax_c = c_int(jmax)
        symtop_grid_r = np.asfortranarray(np.zeros((npoints,2*jmax+1,2*jmax+1,jmax-jmin+1), dtype=np.float64))
        symtop_grid_i = np.asfortranarray(np.zeros((npoints,2*jmax+1,2*jmax+1,jmax-jmin+1), dtype=np.float64))

        fsymtop.symtop_3d_grid.argtypes = [ \
            c_int, \
            c_int, \
            c_int, \
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=4, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=4, flags='F') ]

        fsymtop.symtop_3d_grid.restype = None
        fsymtop.symtop_3d_grid(npoints_c, jmin_c, jmax_c, grid, symtop_grid_r, symtop_grid_i)

        self.coefs = symtop_grid_r.reshape((npoints,2*jmax+1,2*jmax+1,jmax-jmin+1)) \
                   + symtop_grid_i.reshape((npoints,2*jmax+1,2*jmax+1,jmax-jmin+1))*1j

        # Wigner D-functions [D_{m,k}^{(j)}]^* from symmetric-top functions |j,k,m>
        for j in range(jmin,jmax+1):
            self.coefs[:,:,:,j-jmin] = self.coefs[:,:,:,j-jmin] / np.sqrt((2*j+1)/(8.0*np.pi**2))


    def proj(self):
        J = self.J
        proj = np.zeros((self.nirrep,2*J+1,2*J+1), dtype=np.complex128)
        for irrep in range(self.nirrep):
            for ioper in range(self.noper):
                Chi = float(self.characters[ioper,irrep]) # needs to be complex conjugate, fix if characters can be imaginary
                fac = Chi/self.noper
                for k1 in range(-J,J+1):
                    for k2 in range(-J,J+1):
                        proj[irrep,k1+J,k2+J] += fac * self.coefs[ioper,k1+J,k2+J,0]
        return proj


class D2(SymtopSymmetry):
    def __init__(self, J):

        self.noper = 4
        self.nirrep = 4
        self.ndeg = [1,1,1,1]

        self.characters = np.zeros((self.nirrep,self.noper), dtype=np.float64)
        self.euler_rotation = np.zeros((3,self.noper), dtype=np.float64)

        # E  C2(z)  C2(y)  C2(x)
        self.characters[:,0] = [1,1,1,1]    # A
        self.characters[:,1] = [1,1,-1,-1]  # B1 
        self.characters[:,2] = [1,-1,1,-1]  # B2 
        self.characters[:,3] = [1,-1,-1,1]  # B3 

        self.sym_lab=['A','B1','B2','B3']

        pi = np.pi
        # order of angles in euler_rotation[0:3,:] is [phi, theta, chi]
        #self.euler_rotation[:,0] = [0,0,0]        # E
        #self.euler_rotation[:,1] = [pi,-pi,-2*pi] # C2(x)
        #self.euler_rotation[:,2] = [pi,-pi,-pi]   # C2(y)
        #self.euler_rotation[:,3] = [0,0,pi]       # C2(z)
        self.euler_rotation[:,0] = [0,0,0]             # E
        self.euler_rotation[:,1] = [pi,0,0]            # C2(z)
        self.euler_rotation[:,2] = [0,pi,0]            # C2(y)
        self.euler_rotation[:,3] = [0.5*pi,pi,1.5*pi]  # C2(x)

        SymtopSymmetry.__init__(self, J)


class D2h(SymtopSymmetry):
    def __init__(self, J):

        self.noper = 8
        self.nirrep = 8
        self.ndeg = [1,1,1,1,1,1,1,1]

        self.characters = np.zeros((self.nirrep,self.noper), dtype=np.float64)
        self.euler_rotation = np.zeros((3,self.noper), dtype=np.float64)

        # E  C2(z)  C2(y)  C2(x)  i  sxy  sxz  syz  
        self.characters[:,0] = [1, 1, 1, 1, 1, 1, 1, 1]  # Ag
        self.characters[:,1] = [1, 1, 1, 1,-1,-1,-1,-1]  # Au
        self.characters[:,2] = [1, 1,-1,-1, 1, 1,-1,-1]  # B1g
        self.characters[:,3] = [1, 1,-1,-1,-1,-1, 1, 1]  # B1u
        self.characters[:,4] = [1,-1, 1,-1, 1,-1, 1,-1]  # B2g
        self.characters[:,5] = [1,-1, 1,-1,-1, 1,-1, 1]  # B2u
        self.characters[:,6] = [1,-1,-1, 1, 1,-1,-1, 1]  # B3g
        self.characters[:,7] = [1,-1,-1, 1,-1, 1, 1,-1]  # B3u

        self.sym_lab=['Ag','Au','B1g','B1u','B2g','B2u','B3g','B3u']

        pi = np.pi
        # order of angles in euler_rotation[0:3,:] is [phi, theta, chi]
        # this needs to be checked
        self.euler_rotation[:,0] = [0,0,0]             # E
        self.euler_rotation[:,1] = [pi,0,0]            # C2(z)
        self.euler_rotation[:,2] = [0,pi,0]            # C2(y)
        self.euler_rotation[:,3] = [0.5*pi,pi,1.5*pi]  # C2(x)
        self.euler_rotation[:,4] = [0,0,0]             # i
        self.euler_rotation[:,5] = [pi,0,0]            # sxy
        self.euler_rotation[:,6] = [0,pi,0]            # sxz
        self.euler_rotation[:,7] = [0.5*pi,pi,1.5*pi]  # syz

        SymtopSymmetry.__init__(self, J)


class C2v(SymtopSymmetry):
    def __init__(self, J):

        self.noper = 4
        self.nirrep = 4
        self.ndeg = [1,1,1,1]

        self.characters = np.zeros((self.nirrep,self.noper), dtype=np.float64)
        self.euler_rotation = np.zeros((3,self.noper), dtype=np.float64)

        # E  C2(z)  C2(y)  C2(x)
        self.characters[:,0] = [1,1,1,1]    # A1
        self.characters[:,1] = [1,1,-1,-1]  # B2 
        self.characters[:,2] = [1,-1,1,-1]  # B1 
        self.characters[:,3] = [1,-1,-1,1]  # B2 

        self.sym_lab=['A1','A2','B1','B2']

        pi = np.pi
        # order of angles in euler_rotation[0:3,:] is [phi, theta, chi]
        self.euler_rotation[:,0] = [0,0,0]             # E
        self.euler_rotation[:,1] = [pi,0,0]            # C2(z)
        self.euler_rotation[:,2] = [0,pi,0]            # C2(y)
        self.euler_rotation[:,3] = [0.5*pi,pi,1.5*pi]  # C2(x)

        SymtopSymmetry.__init__(self, J)



class J(PsiTableMK):
    """ Basic class for rotational angular momentum operators """

    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            try:
                x = arg.m
            except AttributeError:
                raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
            try:
                x = arg.k
            except AttributeError:
                raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
            PsiTableMK.__init__(self, arg.k, arg.m)


    def __mul__(self, arg):
        try:
            x = arg.m
            y = arg.k
            res = self.__class__(arg)
        except AttributeError:
            if np.isscalar(arg):
                res = J(self)
                res.k = res.k * arg
            else:
                raise TypeError(f"unsupported operand type(s) for '*': '{self.__class__.__name__}' " \
                        +f"and '{arg.__class__.__name__}'") from None
        return res


    __rmul__ = __mul__



class mol_Jp(J):
    """ Molecular-frame J+ = Jx + iJy """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            table = self.k.table.copy()
            self.k.table['c'] = 0
            for ielem,(j,k) in enumerate(table['prim']):
                if abs(k-1) <= j:
                    fac = np.sqrt( j*(j+1) - k*(k-1) )
                    k2 = k - 1
                    jelem = np.where((table['prim']==(j,k2)).all(axis=1))[0][0]
                    self.k.table['c'][jelem,:] = table['c'][ielem,:] * fac
        except AttributeError:
            pass

Jp = mol_Jp


class mol_Jm(J):
    """ Molecular-frame J- = Jx - iJy """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            table = self.k.table.copy()
            self.k.table['c'] = 0
            for ielem,(j,k) in enumerate(table['prim']):
                if abs(k+1) <= j:
                    fac = np.sqrt( j*(j+1) - k*(k+1) )
                    k2 = k + 1
                    jelem = np.where((table['prim']==(j,k2)).all(axis=1))[0][0]
                    self.k.table['c'][jelem,:] = table['c'][ielem,:] * fac
        except AttributeError:
            pass

Jm = mol_Jm


class mol_Jz(J):
    """ Molecular-frame Jz """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            for ielem,(j,k) in enumerate(self.k.table['prim']):
                self.k.table['c'][ielem,:] *= k
        except AttributeError:
            pass

Jz = mol_Jz


class mol_JJ(J):
    """ Molecular-frame J^2 """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            for ielem,(j,k) in enumerate(self.k.table['prim']):
                self.k.table['c'][ielem,:] *= j*(j+1)
        except AttributeError:
            pass

JJ = mol_JJ


class mol_Jxx(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = 0.25 * ( mol_Jm(arg) * mol_Jm(arg) +  mol_Jm(arg) * mol_Jp(arg) \
                +  mol_Jp(arg) * mol_Jm(arg) +  mol_Jp(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jxx = mol_Jxx


class mol_Jxy(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.25) * ( mol_Jm(arg) * mol_Jm(arg) -  mol_Jm(arg) * mol_Jp(arg) \
                +  mol_Jp(arg) * mol_Jm(arg) -  mol_Jp(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jxy = mol_Jxy


class mol_Jyx(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.25) * ( mol_Jm(arg) * mol_Jm(arg) +  mol_Jm(arg) * mol_Jp(arg) \
                -  mol_Jp(arg) * mol_Jm(arg) -  mol_Jp(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jyx = mol_Jyx


class mol_Jxz(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = 0.5 * ( mol_Jm(arg) * mol_Jz(arg) +  mol_Jp(arg) * mol_Jz(arg) )
            J.__init__(self, res)

Jxz = mol_Jxz


class mol_Jzx(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = 0.5 * ( mol_Jz(arg) * mol_Jm(arg) +  mol_Jz(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jzx = mol_Jzx


class mol_Jyy(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = -0.25 * ( mol_Jm(arg) * mol_Jm(arg) -  mol_Jm(arg) * mol_Jp(arg) \
                -  mol_Jp(arg) * mol_Jm(arg) +  mol_Jp(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jyy = mol_Jyy


class mol_Jyz(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.5) * ( mol_Jm(arg) * mol_Jz(arg) -  mol_Jp(arg) * mol_Jz(arg) )
            J.__init__(self, res)

Jyz = mol_Jyz


class mol_Jzy(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.5) * ( mol_Jz(arg) * mol_Jm(arg) -  mol_Jz(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jzy = mol_Jzy


class mol_Jzz(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = mol_Jz(arg) * mol_Jz(arg)
            J.__init__(self, res)

Jzz = mol_Jzz



class CartTensor():
    """ Basic class for laboratory-frame Cartesian tensor operators

    Args:
        arg (np.ndarray, list or tuple): Array with tensor elements in the molecule-fixed frame.
    """

    # transformation matrix for tensors of rank in tmat_s.keys()
    # from Cartesian to spherical-tensor representation
    tmat_s = {1 : np.array([ [np.sqrt(2.0)/2.0, -math.sqrt(2.0)*1j/2.0, 0], \
                             [0, 0, 1.0], \
                             [-math.sqrt(2.0)/2.0, -math.sqrt(2.0)*1j/2.0, 0] ], dtype=np.complex128),
              2 : np.array([ [-1.0/math.sqrt(3.0), 0, 0, 0, -1.0/math.sqrt(3.0), 0, 0, 0, -1.0/math.sqrt(3.0)], \
                             [0, 0, -0.5, 0, 0, 0.5*1j, 0.5, -0.5*1j, 0], \
                             [0, 1.0/math.sqrt(2.0)*1j, 0, -1.0/math.sqrt(2.0)*1j, 0, 0, 0, 0, 0], \
                             [0, 0, -0.5, 0, 0, -0.5*1j, 0.5, 0.5*1j, 0], \
                             [0.5, -0.5*1j, 0, -0.5*1j, -0.5, 0, 0, 0, 0], \
                             [0, 0, 0.5, 0, 0, -0.5*1j, 0.5, -0.5*1j, 0], \
                             [-1.0/math.sqrt(6.0), 0, 0, 0, -1.0/math.sqrt(6.0), 0, 0, 0, (1.0/3.0)*math.sqrt(6.0)], \
                             [0, 0, -0.5, 0, 0, -0.5*1j, -0.5, -0.5*1j, 0], \
                             [0.5, 0.5*1j, 0, 0.5*1j, -0.5, 0, 0, 0, 0] ], dtype=np.complex128) }

    # inverse spherical-tensor to Cartesian transformation matrix
    tmat_x = {key : np.linalg.pinv(val) for key,val in tmat_s.items()}

    cart_ind = {1 : ["x","y","z"], 2 : ["xx","xy","xz","yx","yy","yz","zx","zy","zz"]}
    irrep_ind = {1 : [(1,-1),(1,0),(1,1)], 2 : [(o,s) for o in range(3) for s in range(-o,o+1)] }


    def __init__(self, arg):

        # check input tensor
        if isinstance(arg, (tuple, list)):
            tens = np.array(arg)
        elif isinstance(arg, (np.ndarray,np.generic)):
            tens = arg
        else:
            raise TypeError(f"Unsupported argument type '{type(arg)}' for tensor values, " \
                    +f"must be one of: 'list', 'numpy.ndarray'") from None
        if not all(dim==3 for dim in tens.shape):
            raise ValueError(f"(Cartesian) tensor has bad shape: '{tens.shape}' != {[3]*tens.ndim}") from None
        if np.all(np.abs(tens)<small):
            raise ValueError(f"Tensor has all its elements equal to zero") from None
        if np.any(np.abs(tens)>large*0.1):
            raise ValueError(f"Tensor has too large values of its elements") from None
        if np.any(np.isnan(tens)):
            raise ValueError(f"Tensor has some values of its elements equal to NaN") from None

        rank = tens.ndim
        self.rank = rank
        try:
            self.Us = self.tmat_s[rank]
            self.Ux = self.tmat_x[rank]
            self.os = self.irrep_ind[rank]
            self.cart = self.cart_ind[rank]
        except KeyError:
            raise NotImplementedError(f"Tensor of rank = {rank} is not implemented") from None

        # save mol-fixed tensor in flatted form with the elements following the order in self.cart
        self.tens_flat = np.zeros(len(self.cart), dtype=type(tens))
        for ix,sx in enumerate(self.cart):
            s = [ss for ss in sx]    # e.g. split "xy" into ["x","y"]
            ind = ["xyz".index(ss) for ss in s]    # e.g. convert ["x","y"] into [0,1]
            self.tens_flat[ix] = tens.item(tuple(ind))

        # special cases if tensor is symmetric and traceless
        if self.rank==2:
            symmetric = lambda tens, tol=1e-12: np.all(np.abs(tens-tens.T) < tol)
            traceless = lambda tens, tol=1e-12: abs(np.sum(np.diag(tens)))<tol
            if symmetric(tens)==True and traceless(tens)==True:
                # for symmetric and traceless tensor the following rows in tmat_s and columns in tmat_x
                # will be zero: (0,0), (1,-1), (1,0), and (1,1)
                self.Us = np.delete(self.Us, [0,1,2,3], 0)
                self.Ux = np.delete(self.Ux, [0,1,2,3], 1)
                self.os = [(omega,sigma) for (omega,sigma) in self.os if omega==2]
            elif symmetric(tens)==True and traceless(tens)==False:
                # for symmetric tensor the following rows in tmat_s and columns in tmat_x
                # will be zero: (1,-1), (1,0), and (1,1)
                self.Us = np.delete(self.Us, [1,2,3], 0)
                self.Ux = np.delete(self.Ux, [1,2,3], 1)
                self.os = [(omega,sigma) for (omega,sigma) in self.os if omega in (0,2)]


    def __call__(self, arg):
        """ Computes |psi'> = CartTensor|psi>

        Args:
            arg (PsiTableMK): |psi>, set of linear combinations of symmetric-top functions.

        Returns:
            res (dict of PsiTableMK): |psi'>, resulting tensor-projected set of linear combinations
                of symmetric-top functions.
        """
        try:
            jm_table = arg.m.table
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            jk_table = arg.k.table
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None

        irreps = set(omega for (omega,sigma) in self.os)
        dj_max = max(irreps)    # selection rules |j1-j2|<=omega
        os_ind = {omega : [ind for ind,(o,s) in enumerate(self.os) if o==omega] for omega in irreps}

        # generate tables for tensor-projected set of basis functions

        jmin = min([min(j for (j,k) in jk_table['prim']), min(j for (j,m) in jm_table['prim'])])
        jmax = max([max(j for (j,k) in jk_table['prim']), max(j for (j,m) in jm_table['prim'])])

        prim_k = [(int(J),int(k)) for J in range(max([0,jmin-dj_max]),jmax+1+dj_max) for k in range(-J,J+1)]
        prim_m = [(int(J),int(m)) for J in range(max([0,jmin-dj_max]),jmax+1+dj_max) for m in range(-J,J+1)]

        nstat_k = jk_table['c'].shape[1]
        nstat_m = jm_table['c'].shape[1]

        stat_m = jm_table['stat'][:nstat_m]
        stat_k = jk_table['stat'][:nstat_k]

        # output dictionary
        res = { (cart, irrep) : PsiTableMK(PsiTable(prim_k, stat_k), PsiTable(prim_m, stat_m)) \
                for irrep in irreps for cart in self.cart }

        # some initializations in pywigxjpf module for computing 3j symbols
        wig_table_init((jmax+dj_max)*2, 3)
        wig_temp_init((jmax+dj_max)*2)

        # compute K|psi>
        cart0 = self.cart[0]
        for ind1,(j1,k1) in enumerate(jk_table['prim']):
            for ind2,(j2,k2) in enumerate(prim_k):
                fac = (-1)**abs(k2)
                # compute <j2,k2|K-tensor|j1,k1>
                threeJ = np.array([wig3jj([j1*2, o*2, j2*2, k1*2, s*2, -k2*2]) for (o,s) in self.os])
                for irrep in irreps:
                    ind = os_ind[irrep]
                    me = np.dot(threeJ[ind], np.dot(self.Us[ind,:], self.tens_flat)) * fac
                    res[(cart0,irrep)].k.table['c'][ind2,:] += me * jk_table['c'][ind1,:]

        # for (cart,irrep),val in res.items():
        #     if cart != cart0:
        #         val.k = res[(cart0,irrep)].k

        # compute M|psi>
        for ind1,(j1,m1) in enumerate(jm_table['prim']):
            for ind2,(j2,m2) in enumerate(prim_m):
                fac = np.sqrt((2*j1+1)*(2*j2+1)) * (-1)**abs(m2)
                # compute <j2,m2|M-tensor|j1,m1>
                threeJ = np.array([wig3jj([j1*2, o*2, j2*2, m1*2, s*2, -m2*2]) for (o,s) in self.os])
                for irrep in irreps:
                    ind = os_ind[irrep]
                    me = np.dot(self.Ux[:,ind], threeJ[ind]) * fac
                    for icart,cart in enumerate(self.cart):
                        res[(cart,irrep)].m.table['c'][ind2,:] += me[icart] * jm_table['c'][ind1,:]

        # free memory in pywigxjpf module
        wig_temp_free()
        wig_table_free()

        return res


    def me(self, psi_bra, psi_ket):
        """ Computes matrix elements of Cartesian tensor operator <psi_bra|CartTensor|psi_ket>

        Args:
            psi_bra, psi_ket (PsiTableMK): Set of linear combinations of symmetric-top functions.

        Returns:
            res (dict of 4D arrays): Matrix elements between states in psi_bra and psi_ket sets
                of functions, for different lab-fixed Cartesian components of the tensor (as dict keys).
                For each Cartesian component, matrix elements are stored in a 4D matrix
                [k2,m2,k1,m1], where k2 and k1 are the k-state indices in psi_bra and psi_ket
                respectively, and m2 and m1 are the corresponding m-state indices.
        """
        try:
            x = psi_bra.m.table
        except AttributeError:
            raise AttributeError(f"'{psi_bra.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = psi_bra.k.table
        except AttributeError:
            raise AttributeError(f"'{psi_bra.__class__.__name__}' has no attribute 'k'") from None
        try:
            x = psi_ket.m.table
        except AttributeError:
            raise AttributeError(f"'{psi_ket.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = psi_ket.k.table
        except AttributeError:
            raise AttributeError(f"'{psi_ket.__class__.__name__}' has no attribute 'k'") from None

        tens_psi = self(psi_ket)

        dim_bra_k = psi_bra.k.table['c'].shape[1]
        dim_ket_k = psi_ket.k.table['c'].shape[1]
        dim_bra_m = psi_bra.m.table['c'].shape[1]
        dim_ket_m = psi_ket.m.table['c'].shape[1]

        irreps = list(set(omega for (omega,sigma) in self.os))
        nirrep = len(irreps)

        ovlp_k = np.zeros((nirrep, dim_bra_k, dim_ket_k), dtype=np.complex128)
        ovlp_m = { cart : np.zeros((nirrep, dim_bra_m, dim_ket_m), dtype=np.complex128) \
                   for cart in self.cart }

        cart0 = self.cart[0]
        for (cart,irrep_),val in tens_psi.items():
            irrep = irreps.index(irrep_)
            ovlp_m[cart][irrep,:,:] = psi_bra.overlap_m(val)
            if cart == cart0:
                ovlp_k[irrep,:,:] = psi_bra.overlap_k(val)

        res = {key : np.einsum('ijk,ilm->jlkm', ovlp_k, val) for key,val in ovlp_m.items()}
        return res


    def store_richmol(self, psi_bra, psi_ket, name=None, fname=None, thresh=1e-12):
        """ Stores tensor matrix elements in Richmol file

        Args:
            psi_bra, psi_ket (PsiTableMK): Set of linear combinations of symmetric-top functions.
            name (str): Name of the tensor operator (single word), defaults to "tens"+str(self.rank).
            fname (str): Name of the file for storing tensor matrix elements, defaults to 'name'.
                The actual file name will be fname+"_j"+str(J_ket)+"_j"+str(J_bra)+".rchm", where
                J_bra and J_ket are respective J quanta of bra and ket states.
            thresh (float): Threshold for storing matrix elements in file.
        """
        zero_tol = small*10

        try:
            x = psi_bra.m.table
        except AttributeError:
            raise AttributeError(f"'{psi_bra.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = psi_bra.k.table
        except AttributeError:
            raise AttributeError(f"'{psi_bra.__class__.__name__}' has no attribute 'k'") from None
        try:
            x = psi_ket.m.table
        except AttributeError:
            raise AttributeError(f"'{psi_ket.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = psi_ket.k.table
        except AttributeError:
            raise AttributeError(f"'{psi_ket.__class__.__name__}' has no attribute 'k'") from None

        # determine J quanta for bra and ket states

        Jk_bra = list(set(J for J in psi_bra.k.table['prim'][:,0]))
        Jm_bra = list(set(J for J in psi_bra.m.table['prim'][:,0]))
        if len(Jk_bra)>1 or len(Jm_bra)>1:
            raise ValueError(f"Function {retrieve_name(psi_bra)} couples states with different " \
                    +f"J quanta = {Jk_bra} and {Jm_bra} for k- and m-parts, richmol matrix element " \
                    +f"file cannot be created")
        else:
            if Jk_bra[0] != Jm_bra[0]:
                raise ValueError(f"Inconsistent J quanta = {Jk_bra[0]} and {Jm_bra[0]} in the k- and " \
                        +f"m-dependent parts of function {retrieve_name(psi_bra)}")
            J2 = Jk_bra[0]

        Jk_ket = list(set(J for J in psi_ket.k.table['prim'][:,0]))
        Jm_ket = list(set(J for J in psi_ket.m.table['prim'][:,0]))
        if len(Jk_ket)>1 or len(Jm_ket)>1:
            raise ValueError(f"Function {retrieve_name(psi_ket)} couples states with different " \
                    +f"J quanta = {Jk_ket} and {Jm_ket} for k- and m-parts, richmol matrix element " \
                    +f"file cannot be created")
        else:
            if Jk_ket[0] != Jm_ket[0]:
                raise ValueError(f"Inconsistent J quanta = {Jk_ket[0]} and {Jm_ket[0]} in the k- and " \
                        +f"m-dependent parts of function {retrieve_name(psi_ket)}")
            J1 = Jk_ket[0]

        # compute matrix elements for M and K tensors

        tens_psi = self(psi_ket)

        dim_bra_k = psi_bra.k.table['c'].shape[1]
        dim_ket_k = psi_ket.k.table['c'].shape[1]
        dim_bra_m = psi_bra.m.table['c'].shape[1]
        dim_ket_m = psi_ket.m.table['c'].shape[1]

        irreps = list(set(omega for (omega,sigma) in self.os))
        nirrep = len(irreps)
        ncart = len(self.cart)

        kmat = np.zeros((nirrep, dim_bra_k, dim_ket_k), dtype=np.complex128)
        mmat = { cart : np.zeros((nirrep, dim_bra_m, dim_ket_m), dtype=np.complex128) \
                   for cart in self.cart }

        cart0 = self.cart[0]
        for (cart,irrep_),val in tens_psi.items():
            irrep = irreps.index(irrep_)
            mmat[cart][irrep,:,:] = psi_bra.overlap_m(val)
            if cart == cart0:
                kmat[irrep,:,:] = psi_bra.overlap_k(val)

        # check if elements of K-tensor are all purely real or purely imaginary

        if np.all(abs(kmat[:,:,:].real) < zero_tol) and np.any(abs(kmat[:,:,:].imag) >= zero_tol):
            kmat_cmplx = -1
        elif np.any(abs(kmat[:,:,:].real) >= zero_tol) and np.all(abs(kmat[:,:,:].imag) < zero_tol):
            kmat_cmplx = 0
        elif np.all(abs(kmat[:,:,:].real) < zero_tol) and np.all(abs(kmat[:,:,:].imag) < zero_tol):
            kmat_cmplx = None
            return
        else:
            raise RuntimeError(f"Elements of K-tensor are complex-valued, expected purely real " \
                    +f"or purely imaginary numbers\nK = {kmat}")

        # check if elements of M-tensor are all purely real or imaginary

        mmat_cmplx = {cart : None for cart in self.cart}

        for key,elem in mmat.items():
            if np.all(abs(elem[:,:,:].real) < zero_tol) and np.any(abs(elem[:,:,:].imag) >= zero_tol):
                mmat_cmplx[key] = -1
            elif np.any(abs(elem[:,:,:].real) >= zero_tol) and np.all(abs(elem[:,:,:].imag) < zero_tol):
                mmat_cmplx[key] = 0
            elif np.all(abs(elem[:,:,:].real) < zero_tol) and np.all(abs(elem[:,:,:].imag) < zero_tol):
                mmat_cmplx[key] = None
            else:
                raise RuntimeError(f"Elements of {key} M-tensor are complex-valued, expected purely " \
                        +f"real or purely imaginary numbers\nM = {elem}")

        if all(cmplx is None for cmplx in mmat_cmplx.values()):
            return

        # decide if total M*K matrix element is purely real or imaginary

        assert (kmat_cmplx in [0,-1]), f"kmat_cmplx = {kmat_cmplx} is not in [0,-1]"
        assert (all(elem in [0,-1,None] for elem in mmat_cmplx.values())), \
                f"mmat_cmplx = {mmat_cmplx} is not in [0,-1,None]"

        icmplx = {cart : 0 for cart in self.cart}
        icmplx_coef = {cart : 1 for cart in self.cart}

        for cart in self.cart:
            if mmat_cmplx[cart] is None:
                icmplx[cart] = 0
            elif mmat_cmplx[cart]==0:
                if kmat_cmplx==0:
                    icmplx[cart] = 0
                elif kmat_cmplx==-1:
                    icmplx[cart] = -1
            elif mmat_cmplx[cart]==-1:
                if kmat_cmplx==0:
                    icmplx[cart] = -1
                elif kmat_cmplx==-1:
                    icmplx[cart] = 0
                    icmplx_coef[cart] = -1

        assert (all(ind in [0,-1,None] for ind in icmplx.values())), f"icmplx = {icmplx} is not in [0,-1,None]"

        # ready to write matrix elements into file

        if name is not None:
            name_ = name
        else:
            name_ = "tens"+str(self.rank)

        if fname is not None:
            fname_ = fname
        else:
            fname_ = name_

        with open(fname_+"_j"+str(J1)+"_j"+str(J2)+".rchm", "w") as fl:
            fl.write("Start richmol format\n")
            fl.write("%s"%name_ + "  %4i"%nirrep + "  %4i"%ncart + "\n")
            fl.write("M-tensor\n")
            icart = 1
            for key,val in mmat.items():
                fl.write("alpha" + "  %4i"%icart + "  %4i"%icmplx[key] + "  %s"%key + "\n")
                for i1,(j,m1) in enumerate(psi_ket.m.table['stat']):
                    for i2,(j,m2) in enumerate(psi_bra.m.table['stat']):
                        if mmat_cmplx[key] == 0:
                            me = val[:,i2,i1].real * icmplx_coef[key]
                        elif mmat_cmplx[key] == -1:
                            me = val[:,i2,i1].imag * icmplx_coef[key]
                        else:
                            continue
                        if np.any(abs(me)>thresh):
                            fl.write(" %4i"%m1 + " %4i"%m2 + " ".join("  %20.12e"%elem for elem in me) + "\n")
                icart+=1
            fl.write("K-tensor\n")
            for i1 in range(psi_ket.k.table['c'].shape[1]):
                for i2 in range(psi_bra.k.table['c'].shape[1]):
                    if kmat_cmplx == 0:
                        me = kmat[:,i2,i1].real
                    elif kmat_cmplx==-1:
                        me = kmat[:,i2,i1].imag
                    id1 = i1 + 1
                    id2 = i2 + 1
                    if np.any(abs(me)>thresh):
                        fl.write(" %6i"%id1 + " %6i"%id2 + "    1 1 " + " ".join("  %20.12e"%elem for elem in me) + "\n")
            fl.write("End richmol format")



def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def expecting(offset=0):
    """ Return how many values the caller is expecting """
    f = inspect.currentframe().f_back.f_back
    i = f.f_lasti + offset
    bytecode = f.f_code.co_code
    instruction = bytecode[i]
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        return bytecode[i + 1]
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    else:
        return 1

