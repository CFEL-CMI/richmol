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
    imom_offdiag_tol = 1e-12
    tens_symm_tol = 1e-14
    tens_small_elem = small
    tens_large_elem = large/10.0
    rotmat_small_elem = small
    rotmat_large_elem = large/10.0
    gmat_svd_small = 1e-12
    assign_nprim = 1 # number of primitive basis contributions printed in the state assignment (e.g. in Richmol states file)
    assign_ndig_c2 = 4 # number of digits printed for the assignment coefficient |c|^2


def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


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
        """To set and return Cartesian coordinates of atoms in molecule

        >>> d2s = RigidMolecule()
        >>> d2s.XYZ = ( "angstrom", \
                        "S",   0.00000000,        0.00000000,        0.10358697, \
                        "H2", -0.96311715,        0.00000000,       -0.82217544, \
                        "H2",  0.96311715,        0.00000000,       -0.82217544 )
        >>> print(d2s.XYZ['mass']) # print masses of atoms
        [31.97207117  2.01410178  2.01410178]
        >>> print(d2s.XYZ['label']) # print names of atoms
        ['S' 'H2' 'H2']
        >>> print(d2s.XYZ['xyz']) # print Cartesian coordinates of atoms (in Angstrom)
        [[ 0.          0.          0.10358697]
         [-0.96311715  0.         -0.82217544]
         [ 0.96311715  0.         -0.82217544]]

        >>> d2s.frame = "zxy" # change molecular frame where the axes are swapped places, e.g., xyz --> zxy
        >>> print(d2s.XYZ['xyz'].round(8)) # print Cartesian coordinates of atoms in the new molecular frame
        [[ 0.10358697  0.          0.        ]
         [-0.82217544 -0.96311715  0.        ]
         [-0.82217544  0.96311715  0.        ]]

        """
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
            raise TypeError(f"Bad argument type '{type(arg)}' for atoms' specification") from None

        self.atoms = np.array( [(lab, mass, cart) for lab,mass,cart in zip(label,mass,xyz)], \
                               dtype=[('label','U10'),('mass','f8'),('xyz','f8',(3))] )


    @property
    def tensor(self):
        """To set and return molecule-fixed Cartesian tensors

        >>> mol = RigidMolecule()
        >>> # specify Cartesian coordinates of atoms (water molecule)
        >>> mol.XYZ = ("angstrom", "O", 0,0,0, "H", 0.8, 0.6, 0, "H", -1,2, 0.6, 0)
        >>> # add new rank-1 tensor with name "mu"
        >>> mol.tensor = ("mu", [0.5, -0.1, 0])
        >>> # add rank-2 tensor with name "ccsd(t) alpha"
        >>> mol.tensor = ("ccsd(t) alpha", [[10,0,0],[0,20,0],[0,0,30]])
        >>> print( mol.tensor["mu"] )
        [ 0.5 -0.1  0. ]
        >>> print( mol.tensor["ccsd(t) alpha"] )
        [[10  0  0]
         [ 0 20  0]
         [ 0  0 30]]

        >>> # change molecular frame to the principal axes system
        >>> mol.frame = "pas"
        >>> # print tensors "mu" and "ccsd(t) alpha" in the new molecular frame
        >>> print(mol.tensor["mu"].round(6))
        [-0.299156  0.400636  0.099985]
        >>> print(mol.tensor["ccsd(t) alpha"].round(6))
        [[18.865071  3.663798  3.167346]
         [ 3.663798 12.076468 -1.865857]
         [ 3.167346 -1.865857 29.058462]]

        >>> # try to add new tensor with the name identical to the one added before
        >>> mol.tensor = ("ccsd(t) alpha", [[10,1,0],[2,20,0],[3,0,30]])
        Traceback (most recent call last):
            ...
        ValueError: Tensor with the name 'ccsd(t) alpha' already exists

        """
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
                    raise ValueError(f"Number of dimensions for tensor '{name}' = {ndim} " \
                            +f"exceeds the maximum = {len(sa)}") from None
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
        # check if input is (name, tensor)
        try:
            name, tens = arg
            name = name.strip()
        except ValueError:
            raise ValueError(f"Pass an iterable with two items, tensor = ('name', tensor)") from None
        # check if name and tensor have proper types
        if not isinstance(name, str):
            raise TypeError(f"Bad argument type '{type(name)}' for tensor name") from None
        if isinstance(tens, (tuple, list)):
            tens = np.array(tens)
        elif isinstance(tens, (np.ndarray,np.generic)):
            pass
        else:
            raise TypeError(f"Bad argument type '{type(tens)}' for tensor values ") from None
        # check if name and tensor have proper values
        if "," in name or len(name) == 0:
            raise ValueError(f"Illegal tensor name '{name}'")
        if not all(dim==3 for dim in tens.shape):
            raise ValueError(f"Cartesian tensor has bad shape: '{tens.shape}' != {[3]*tens.ndim}") from None
        if np.all(np.abs(tens) < settings.tens_small_elem):
            raise ValueError(f"Tensor has all its elements equal to zero") from None
        if np.any(np.abs(tens) > settings.tens_large_elem):
            raise ValueError(f"Tensor has too large values of its elements") from None
        if np.any(np.isnan(tens)):
            raise ValueError(f"Tensor has some values of its elements equal to NaN") from None
        # save tensor
        try:
            x = self.tens
        except AttributeError:
            self.tens = {}
        if name in self.tens:
            raise ValueError(f"Tensor with the name '{name}' already exists") from None
        self.tens[name] = tens


    @property
    def frame(self):
        """To define and change molecule-fixed frame (embedding)

        >>> d2s = RigidMolecule()
        >>> d2s.XYZ = ( "angstrom", \
                        "S",   0.00000000,        0.00000000,        0.10358697, \
                        "H2", -0.96311715,        0.00000000,       -0.82217544, \
                        "H2",  0.96311715,        0.00000000,       -0.82217544 )
        >>> d2s.tensor = ("dipole moment", [0, 0, -9.70662418E-01])
        >>> d2s.tensor = ("polarizability", [[30, -0.5, 0.03], \
                                             [-0.5, 20, 1.3], \
                                             [0.03, 1.3, 35]] )
        >>> # change frame to principal axes system (PAS)
        >>> d2s.frame = "pas"
        >>> # print Cartesian coordinates of atoms in PAS
        >>> print(d2s.XYZ['xyz'])
        [[ 0.          0.10358697  0.        ]
         [-0.96311715 -0.82217544  0.        ]
         [ 0.96311715 -0.82217544  0.        ]]
        >>> # print tensor "dipole moment" in PAS
        >>> print(d2s.tensor['dipole moment'])
        [ 0.         -0.97066242  0.        ]
        >>> # print tensor "polarizability" in PAS
        >>> print(d2s.tensor['polarizability'])
        [[ 3.0e+01  3.0e-02 -5.0e-01]
         [ 3.0e-02  3.5e+01  1.3e+00]
         [-5.0e-01  1.3e+00  2.0e+01]]
        >>> # print type of frame (str) and rotation matrix
        >>> frame_type, rotation_matrix = d2s.frame
        >>> print(frame_type)
        pas
        >>> print(rotation_matrix)
        [[1. 0. 0.]
         [0. 0. 1.]
         [0. 1. 0.]]

        >>> # add frame that is rotated to principal axes of tensor "pol"
        >>> d2s.frame = "pol"
        Traceback (most recent call last):
            ...
        KeyError: "Tensor 'pol' was not initialised"
        >>> d2s.frame = "polarizability" # wait, we don't have tensor "pol" but "polarizability"
        >>> # print frame type and total (collective) rotation matrix
        >>> frame_type, rotation_matrix = d2s.frame
        >>> print(frame_type)
        pas,polarizability
        >>> print(rotation_matrix.round(6))
        [[-0.049338 -0.99511   0.085563]
         [ 0.998779 -0.048939  0.006765]
         [-0.002544  0.085792  0.99631 ]]
        >>> # check if "polarizability" tensor is diagonal in new frame
        >>> print(d2s.tensor["polarizability"].round(6))
        [[19.863432  0.        0.      ]
         [ 0.       30.024702  0.      ]
         [ 0.        0.       35.111866]]

        """
        try:
            rotmat = self.frame_rotation
            frame_type = self.frame_type
        except AttributeError:
            rotmat = np.eye(3, dtype=np.float64)
            frame_type = "I"
        return frame_type, rotmat


    @frame.setter
    def frame(self, arg):
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

                elif "".join(sorted(fr.lower())) == "xyz":
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
                    if tens.ndim != 2:
                        raise ValueError(f"Tensor '{fr}' has inappropriate rank: {tens.ndim} " \
                                +f"is not equal to 2") from None
                    if np.any(np.abs(tens - tens.T) > settings.tens_symm_tol):
                        raise ValueError(f"Tensor '{fr}' is not symmetric") from None
                    try:
                        diag, rotmat = np.linalg.eigh(tens)
                    except np.linalg.LinAlgError:
                        raise RuntimeError("Eigenvalues did not converge") from None
                    self.frame_rotation = np.dot(np.transpose(rotmat), self.frame_rotation)

        else:
            raise TypeError(f"Bad argument type '{type(arg)}' for frame specification") from None

        # update string that keeps track of all frame rotations
        try:
            self.frame_type += "," + arg
        except AttributeError:
            self.frame_type = arg


    @property
    def B(self):
        """Returns Bx, By, Bz rotational constants in units of cm^-1

        >>> d2s = RigidMolecule()
        >>> d2s.XYZ = ( "angstrom", \
                        "S",   0.00000000,        0.00000000,        0.10358697, \
                        "H2", -1.26311715,        0.00000000,       -0.82217544, \
                        "H2",  0.96311715,        0.00000000,       -0.82217544 )
        >>> Bx, By, Bz = d2s.B
        Traceback (most recent call last):
            ...
        RuntimeError: Can't compute rotational constants inertia tensor is not diagonal = [[ 3.066  -0.     -0.4968]
         [-0.      8.1376 -0.    ]
         [-0.4968 -0.      5.0716]], max offdiag = 0.4968

        >>> # to compute rotational constants we must use principal axes frame
        >>> d2s.frame = "pas"
        >>> Bx, By, Bz = d2s.B
        >>> print(round(Bx,4), round(By,4), round(Bz,4))
        5.715 3.2494 2.0716

        """
        imom = self.imom()
        tol = settings.imom_offdiag_tol
        if np.any(np.abs( np.diag(np.diag(imom)) - imom ) > tol):
            raise RuntimeError("Can't compute rotational constants " \
                    +f"inertia tensor is not diagonal = {imom.round(4)}, " \
                    +f"max offdiag = {np.max(np.abs(np.diag(np.diag(imom))-imom)).round(4)}") from None
        convert_to_cm = planck * avogno * 1e+16 / (8.0 * np.pi * np.pi * vellgt) 
        return [convert_to_cm/val for val in np.diag(imom)]


    @B.setter
    def B(self, val):
        raise AttributeError(f"You can't set {retrieve_name(self)}.B") from None


    @property
    def kappa(self):
        """Returns asymmtery parameter kappa = (2*B-A-C)/(A-C)

        >>> d2s = RigidMolecule()
        >>> d2s.XYZ = ( "angstrom", \
                        "S",   0.00000000,        0.00000000,        0.10358697, \
                        "H2", -1.26311715,        0.00000000,       -0.82217544, \
                        "H2",  0.96311715,        0.00000000,       -0.82217544 )
        >>> d2s.frame = "pas" # to compute rotational constants we must use principal axes frame
        >>> asymmetry = d2s.kappa
        >>> print(round(asymmetry,4))
        -0.3534

        """
        A, B, C = reversed(sorted(self.B))
        return (2*B-A-C)/(A-C)


    @kappa.setter
    def kappa(self, val):
        raise AttributeError(f"You can't set {retrieve_name(self)}.kappa") from None


    def imom(self):
        """Computes moments of inertia tensor

        >>> camphor = RigidMolecule()
        >>> camphor.XYZ = ("angstrom", \
                    "O",     -2.547204,    0.187936,   -0.213755, \
                    "C",     -1.382858,   -0.147379,   -0.229486, \
                    "C",     -0.230760,    0.488337,    0.565230, \
                    "C",     -0.768352,   -1.287324,   -1.044279, \
                    "C",     -0.563049,    1.864528,    1.124041, \
                    "C",      0.716269,   -1.203805,   -0.624360, \
                    "C",      0.929548,    0.325749,   -0.438982, \
                    "C",      0.080929,   -0.594841,    1.638832, \
                    "C",      0.791379,   -1.728570,    0.829268, \
                    "C",      2.305990,    0.692768,    0.129924, \
                    "C",      0.730586,    1.139634,   -1.733020, \
                    "H",     -1.449798,    1.804649,    1.756791, \
                    "H",     -0.781306,    2.571791,    0.321167, \
                    "H",      0.263569,    2.255213,    1.719313, \
                    "H",      1.413749,   -1.684160,   -1.316904, \
                    "H",     -0.928638,   -1.106018,   -2.110152, \
                    "H",     -1.245108,   -2.239900,   -0.799431, \
                    "H",      1.816886,   -1.883799,    1.170885, \
                    "H",      0.276292,   -2.687598,    0.915376, \
                    "H",     -0.817893,   -0.939327,    2.156614, \
                    "H",      0.738119,   -0.159990,    2.396232, \
                    "H",      3.085409,    0.421803,   -0.586828, \
                    "H",      2.371705,    1.769892,    0.297106, \
                    "H",      2.531884,    0.195217,    1.071909, \
                    "H",      0.890539,    2.201894,   -1.536852, \
                    "H",      1.455250,    0.830868,   -2.487875, \
                    "H",     -0.267696,    1.035608,   -2.160680)
        >>> imom = camphor.imom()
        >>> print(imom.round(6))
        [[ 3.49280826e+02 -9.59405000e-01  1.09125800e+00]
         [-9.59405000e-01  4.27045792e+02 -8.05120000e-02]
         [ 1.09125800e+00 -8.05120000e-02  4.60626510e+02]]

        >>> # rotate to principal axes system
        >>> camphor.frame = "pas"
        >>> imom = camphor.imom()
        >>> # check if moments of inertia tensor is now diagonal
        >>> print(imom.round(6))
        [[349.25832    0.         0.      ]
         [  0.       427.057364   0.      ]
         [  0.         0.       460.637444]]

        """
        xyz = self.XYZ['xyz']
        mass = self.XYZ['mass']
        cm = np.sum([x*m for x,m in zip(xyz,mass)], axis=0) / np.sum(mass)
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
        """Rotational kinetic energy matrix

        >>> camphor = RigidMolecule()
        >>> camphor.XYZ = ("angstrom", \
                    "O",     -2.547204,    0.187936,   -0.213755, \
                    "C",     -1.382858,   -0.147379,   -0.229486, \
                    "C",     -0.230760,    0.488337,    0.565230, \
                    "C",     -0.768352,   -1.287324,   -1.044279, \
                    "C",     -0.563049,    1.864528,    1.124041, \
                    "C",      0.716269,   -1.203805,   -0.624360, \
                    "C",      0.929548,    0.325749,   -0.438982, \
                    "C",      0.080929,   -0.594841,    1.638832, \
                    "C",      0.791379,   -1.728570,    0.829268, \
                    "C",      2.305990,    0.692768,    0.129924, \
                    "C",      0.730586,    1.139634,   -1.733020, \
                    "H",     -1.449798,    1.804649,    1.756791, \
                    "H",     -0.781306,    2.571791,    0.321167, \
                    "H",      0.263569,    2.255213,    1.719313, \
                    "H",      1.413749,   -1.684160,   -1.316904, \
                    "H",     -0.928638,   -1.106018,   -2.110152, \
                    "H",     -1.245108,   -2.239900,   -0.799431, \
                    "H",      1.816886,   -1.883799,    1.170885, \
                    "H",      0.276292,   -2.687598,    0.915376, \
                    "H",     -0.817893,   -0.939327,    2.156614, \
                    "H",      0.738119,   -0.159990,    2.396232, \
                    "H",      3.085409,    0.421803,   -0.586828, \
                    "H",      2.371705,    1.769892,    0.297106, \
                    "H",      2.531884,    0.195217,    1.071909, \
                    "H",      0.890539,    2.201894,   -1.536852, \
                    "H",      1.455250,    0.830868,   -2.487875, \
                    "H",     -0.267696,    1.035608,   -2.160680)
        >>> gmat = camphor.gmat()
        >>> for i in range(3):
        ...     for j in range(3):
        ...         print("G("+"xyz"[i]+","+"xyz"[j]+") = ", gmat[i,j].round(6))
        G(x,x) =  0.096529
        G(x,y) =  0.000217
        G(x,z) =  -0.000229
        G(y,x) =  0.000217
        G(y,y) =  0.07895
        G(y,z) =  1.3e-05
        G(z,x) =  -0.000229
        G(z,y) =  1.3e-05
        G(z,z) =  0.073195

        >>> # test for linear molecule
        >>> HF = RigidMolecule()
        >>> HF.XYZ = ("angstrom", "F", 0,0,0, "H", 0,0,0.91)
        >>> gmat = HF.gmat()
        Warning: rotational kinetic energy matrix is singular, singular element index = 2, singular value = 0.0
        this is fine for linear molecule: set 1/0.0=0

        """
        convert_to_cm = planck * avogno * 1e+16 / (4.0 * np.pi * np.pi * vellgt)
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
            if sv[i] > settings.gmat_svd_small:
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
    """Basic class to handle operations on rotational wavefunctions, which are represented
    by a table of superposition coefficients with different primitive basis functions in rows
    and different states in columns.

    Attributes:
        table : structured numpy array
            table['c'][:,:] is a numpy.complex128 matrix, holds wafefunctions' superposition
            coefficients, with primitive basis functions stored in rows and states in columns.
            table['prim'][:] is an array of tuples of integer quantum numbers (q1, q2, ...)
            used for assignment of different primitive basis functions.
            table['stat'][:] is an array of tuples of U10 quantum numbers (s1, s2, ...)
            used for assignment of different states.
        enr : numpy array, enr.shape = table['c'].shape[1]
            Contains states' energies. This attribute is dynamically added as part of unitary
            transformation of the wavefunction set in rotate().
        sym : numpy array, sym.shape = table['c'].shape[1]
            Contains states' symmetry labels. This attribute is dynamically added as part of ....
    """

    def __init__(self, prim, stat, coefs=None):
        """Initialize PsiTable from the list of primitive quanta 'prim', state assignments 'stat',
        and, if provided, matrix of superposition coefficients 'coefs' (zero matrix by default).
        """
        if not isinstance(prim, (list, tuple, np.ndarray)):
            raise TypeError(f"Bad argument type '{type(prim)}' for set of primitive quanta, expected " \
                    +f"list, tuple or numpy array") from None
        if not isinstance(stat, (list, tuple, np.ndarray)):
            raise TypeError(f"Bad argument type '{type(stat)}' for set of state quanta, expected " \
                    +f"list, tuple or numpy array") from None
        try:
            x = [int(val) for elem in prim for val in elem]
        except ValueError:
            raise ValueError(f"Failed to convert element in 'prim' into integer") from None

        nprim = len(prim)
        nstat = len(stat)
        assert (nprim>0), f"Number of primitives = 0"
        assert (nstat>0), f"Number of states = 0"
        assert (nprim>=nstat), f"Number of primitives is smaller than number of states, i.e. " \
                +f"nprim < nstat: {nprim} < {nstat}"

        nelem_stat = list(set(len(elem) for elem in stat))
        if len(nelem_stat)>1:
            raise ValueError(f"Different lengths for different elements in 'stat'") from None
        nelem_prim = list(set(len(elem) for elem in prim))
        if len(nelem_prim)>1:
            raise ValueError(f"Different lengths for different elements in 'prim'") from None

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
                raise TypeError(f"Bad argument type '{type(coefs)}' for coefficients matrix, expected " \
                        +f"numpy array") from None
            if any(x!=y for x,y in zip(shape,[nprim,nstat])):
                raise ValueError(f"Shape of the coefficients matrix = {shape} is not aligned with the " \
                        +f"number of primitives = {nprim} and number of states = {nstat}") from None
            self.table['c'][:,:] = coefs


    @classmethod
    def fromPsiTable(cls, arg):
        """Initialize PsiTable from an argument of PsiTable type using deepcopy
        """
        if not isinstance(arg, PsiTable):
            raise TypeError(f"Bad argument type '{type(arg)}', expected 'PsiTable'") from None
        cls = copy.deepcopy(arg)
        return cls


    def __add__(self, arg):
        try:
            x = arg.table
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'table'") from None

        if not np.array_equal(self.table['prim'], arg.table['prim']):
            raise ValueError(f"'{type(self)}' objects under sum have different sets of primitive quanta " \
                    +f"(table['prim'] attributes do not match)") from None

        if not np.array_equal(self.table['stat'], arg.table['stat']):
            raise ValueError(f"'{type(self)}' objects under sum have different sets of state quanta " \
                    +f"(table['stat'] attributes do not match)") from None

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
            raise ValueError(f"'{type(self)}' objects under subtr have different sets of primitive quanta " \
                    +f"(table['prim'] attributes do not match)") from None

        if not np.array_equal(self.table['stat'], arg.table['stat']):
            raise ValueError(f"'{type(self)}' objects under subtr have different sets of state quanta " \
                    +f"(table['stat'] attributes do not match)") from None

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
            raise TypeError(f"Unsupported operand type(s) for '*': '{self.__class__.__name__}' and " \
                    +f"'{arg.__class__.__name__}'") from None
        return PsiTable(prim, stat, coefs)


    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__


    def append(self, arg, del_duplicate_stat=False, del_zero_stat=False, del_zero_prim=False, thresh=1e-12):
        """ Appends two wavefunction sets together """
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
            raise ValueError(f"Two sets in append have different length of the elements in 'stat'") \
                    from None
        if len(prim1[0]) != len(prim2[0]):
            raise ValueError(f"Two sets in append have different length of the elements in 'prim'") \
                    from None

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
            prim, stat, coefs = self.del_zero_stat(prim, stat, coefs, thresh)
        if del_zero_prim==True:
            prim, stat, coefs = self.del_zero_prim(prim, stat, coefs, thresh)
        if del_duplicate_stat==True:
            prim, stat, coefs = self.del_duplicate_stat(prim, stat, coefs, thresh)

        # check for duplicates in 'stat'
        if len(list(set(tuple(s) for s in stat))) != len(stat):
            raise ValueError(f"Two sets in append have overlapping 'stat' elements that correspond " \
                    +f"to different coefficient vectors, try option 'del_duplicate_stat=True'") from None

        return PsiTable(prim, stat, coefs)


    def overlap(self, arg):
        """Computes overlap < self | arg >

        Args:
            arg : PsiTable
                Wavefunction set to compute overlap with
        """
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
            warnings.warn(f"Functions have no overlapping primitive quanta, the overlap is zero!")
        ind1 = [prim1.index(x) for x in both]
        ind2 = [prim2.index(x) for x in both]
        # dot product across overlapping primitive quanta
        return np.dot(coefs1[:,ind1], coefs2[ind2,:])


    def rotate(self, arg, stat=None):
        """Applies unitary transformation to wavefunction set

        Args:
            arg : numpy.ndarray or tuple (numpy.ndarray, numpy.ndarray)
                Contains unitary transformation matrix and, if provided, a set of associated
                state energies (second in tuple).
                The number of columns in unitary transformation matrix must be equal to
                the number of states self.table['c'].shape[1].
            stat : list
                User-defined assignments for resulting unitary-transformed wavefunctions.

        Returns:
            Resulting unitary-transformed wavefunction set as PsiTable object.
            If state energies are provided in 'arg', they are stored in self.enr.
        """
        try:
            rotmat, enr = arg
        except ValueError:
            rotmat = arg
            enr = None

        try:
            shape = rotmat.shape
        except AttributeError:
            raise AttributeError(f"Bad argument type '{type(rotmat)}' for rotation matrix, expected " \
                    +f"numpy array") from None

        if enr is None:
            pass
        elif isinstance(enr, (list, tuple, np.ndarray)):
            if shape[0] != len(enr):
                raise ValueError(f"Number of elements in the energy array = {len(enr)} is not aligned " \
                        +f"with the number of rows in the rotation matrix = {shape[0]}") from None
        else:
            raise ValueError(f"Bad argument type '{type(enr)}' for a set of associated energies, " \
                    +f"expected list, tuple, numpy array") from None

        nstat = self.table['c'].shape[1]
        if shape[1] != nstat:
            raise ValueError(f"Number of columns in the rotation matrix = {shape[1]} is not aligned " \
                    +f"with the number ofstates = {nstat}") from None

        if np.all(np.abs(rotmat) < settings.rotmat_small_elem):
            raise ValueError(f"Rotation matrix has all its elements equal to zero") from None
        if np.any(np.abs(rotmat) > settings.rotmat_large_elem):
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
            raise ValueError(f"Number of elements in the state assignment = {len(stat)} is not aligned " \
                    +f"with the number of rows in the rotation matrix = {rotmat.shape[0]}") from None
        prim = [elem for elem in self.table['prim']]
        res = PsiTable(prim, stat, coefs)

        if enr is not None:
            res.enr = np.array(enr, dtype=np.float64)

        return res


    def del_zero_stat(self, prim=None, stat=None, coefs=None, thresh=1e-12):
        """ Deletes states with zero coefficients """
        if all(x is None for x in (prim,stat,coefs)):
            prim = self.table['prim'].copy()
            stat = self.table['stat'].copy()
            coefs = self.table['c'].copy()
            freturn = lambda prim, stat, coefs: PsiTable(prim, stat, coefs)
        elif all(x is not None for x in (prim,stat,coefs)):
            freturn = lambda prim, stat, coefs: (prim, stat, coefs)
        else:
            raise ValueError(f"Expecting either all 'prim', 'stat', and 'coefs' arguments " \
                    +f"to be defined or none of them") from None
        nstat = coefs.shape[1]
        ind = [istat for istat in range(nstat) if all(abs(val) < thresh for val in coefs[:,istat])]
        coefs2 = np.delete(coefs, ind, 1)
        stat2 = np.delete(stat[:nstat], ind, 0)
        prim2 = [elem for elem in prim]
        if len(stat2)==0:
            return None # somehow deleted all states
        return freturn(prim2, stat2, coefs2)


    def del_zero_prim(self, prim=None, stat=None, coefs=None, thresh=1e-12):
        """ Deletes primitives that are not coupled by states """
        if all(x is None for x in (prim,stat,coefs)):
            prim = self.table['prim'].copy()
            stat = self.table['stat'].copy()
            coefs = self.table['c'].copy()
            freturn = lambda prim, stat, coefs: PsiTable(prim, stat, coefs)
        elif all(x is not None for x in (prim,stat,coefs)):
            freturn = lambda prim, stat, coefs: (prim, stat, coefs)
        else:
            raise ValueError(f"Expecting either all 'prim', 'stat', and 'coefs' arguments " \
                    +f"to be defined or none of them") from None
        nprim = coefs.shape[0]
        nstat = coefs.shape[1]
        ind = [iprim for iprim in range(nprim) if all(abs(val) < thresh for val in coefs[iprim,:])]
        coefs2 = np.delete(coefs, ind, 0)
        prim2 = np.delete(prim, ind, 0)
        stat2 = [elem for elem in stat[:nstat]]
        if len(prim2)==0:
            return None # somehow deleted all primitives
        return freturn(prim2, stat2, coefs2)


    def del_duplicate_stat(self, prim=None, stat=None, coefs=None, thresh=1e-12):
        """ Deletes duplicate states """
        if all(x is None for x in (prim,stat,coefs)):
            prim = self.table['prim'].copy()
            stat = self.table['stat'].copy()
            coefs = self.table['c'].copy()
            freturn = lambda prim, stat, coefs: PsiTable(prim, stat, coefs)
        elif all(x is not None for x in (prim,stat,coefs)):
            freturn = lambda prim, stat, coefs: (prim, stat, coefs)
        else:
            raise ValueError(f"Expecting either all 'prim', 'stat', and 'coefs' arguments " \
                    +f"to be defined or none of them") from None
        nstat = coefs.shape[1]
        ind = []
        for istat in range(nstat):
            ind += [jstat for jstat in range(istat+1,nstat) \
                   if all(abs(val1-val2) < thresh for val1,val2 in zip(coefs[:,istat],coefs[:,jstat]))]
        coefs2 = np.delete(coefs, ind, 1)
        stat2 = np.delete(stat[:nstat], ind, 0)
        prim2 = [elem for elem in prim]
        if len(stat2)==0:
            return None # somehow deleted all states
        return freturn(prim2, stat2, coefs2)



class PsiTableMK():
    """Basic class to handle operations on rotational wavefunctions, which are represented
    by two tables of superposition coefficients (PsiTable class), one for k-subset and one for
    m-subset of quantum numbers.

    Attributes:
        k : PsiTable  class
            Holds table of superposition coefficients for k-subset.
        m : PsiTable  class
            Holds table of superposition coefficients for m-subset.
    """

    def __init__(self, psik, psim):
        if not isinstance(psik, PsiTable):
            raise TypeError(f"Bad argument type '{type(psik)}', expected 'PsiTable'")
        if not isinstance(psim, PsiTable):
            raise TypeError(f"Bad argument type '{type(psim)}', expected 'PsiTable'")

        # initialize using PsiTable.__init__
        # this way some of the attributes (added dynamically) in psik and psim will be lost
        #
        # nstat = psim.table['c'].shape[1]
        # self.m = PsiTable(psim.table['prim'], psim.table['stat'][:nstat], psim.table['c'])
        # nstat = psik.table['c'].shape[1]
        # self.k = PsiTable(psik.table['prim'], psik.table['stat'][:nstat], psik.table['c'])

        # initialize using PsiTable.fromPsiTable
        # this way psik and psim will be deep-copied keeping all dynamically added attributes
        #
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
            raise TypeError(f"Unsupported operand type(s) for '*': '{self.__class__.__name__}' and " \
                    +f"'{arg.__class__.__name__}'") from None
        return PsiTableMK(res_k, res_m)

    __rmul__ = __mul__


    def append(self, arg, del_duplicate_stat=False, del_zero_stat=False, del_zero_prim=False, thresh=1e-12):
        """Appends two wavefunction sets together

        If requested, can delete duplicate states ('del_duplicate_stat' = True),
        delete states with all coefficients below 'thresh' ('del_zero_stat' = True),
        and delete primitive functions that have negligible contribution (below 'thresh')
        to all states ('del_zero_prim' = True)

        Args:
            arg : PsiTableMK class
                Wavefunction set to be appended to the current set

        Examples:

        >>> J1 = 2
        >>> J2 = 3
        >>> bas1 = SymtopBasis(J1) # initialize basis, PsiTableMK type
        >>> bas2 = SymtopBasis(J2)
        >>> bas = bas1.append(bas1) # try to add two sets bas1 and bas2 that have identical states
        Traceback (most recent call last):
            ...
        ValueError: Two sets in append have overlapping 'stat' elements that correspond to different coefficient vectors, try option 'del_duplicate_stat=True'

        >>> # to avoid this problem, we ask to delete all duplicate states in the resulting set 'bas'
        >>> bas = bas1.append(bas1, del_duplicate_stat=True)
        >>> # since we expect that 'bas' == 'bas1', the number of state and primitive functions must be equal in two sets
        >>> no_prim, no_states = bas.k.table['c'].shape
        >>> no_prim1, no_states1 = bas1.k.table['c'].shape
        >>> print(no_prim, no_states, no_prim1, no_states1)
        5 5 5 5

        >>> # append different sets of functions 'bas1' and 'bas2'
        >>> bas = bas1.append(bas2)
        >>> no_prim, no_states = bas.k.table['c'].shape
        >>> no_prim1, no_states1 = bas1.k.table['c'].shape
        >>> no_prim2, no_states2 = bas2.k.table['c'].shape
        >>> print(no_prim1, no_states1)
        5 5
        >>> print(no_prim2, no_states2)
        7 7
        >>> print(no_prim, no_states)
        12 12

        """
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        res_m = self.m.append(arg.m, del_duplicate_stat, del_zero_stat, del_zero_prim, thresh)
        res_k = self.k.append(arg.k, del_duplicate_stat, del_zero_stat, del_zero_prim, thresh)
        return PsiTableMK(res_k, res_m)


    def overlap(self, arg):
        """Computes overlap between wavefunction sets < self | arg >

        If one return value is requested, computes only overlap between the k-subspaces,
        in case of two requested return values, computes overlaps between both k- and m-subspaces

        Args:
            arg : PsiTableMK class
                Wavefunction set to compute the overlap with

        Examples:

        >>> J = 10
        >>> bas1 = SymtopBasis(J) # initialize basis, PsiTableMK type
        >>> bas2 = SymtopBasis(J)
        >>> ovlp = bas1.overlap(bas2) # compute overlap between k-subspaces of bas1 and bas2
        >>> print(np.diag(ovlp).round(6)) # since bas1 == bas2, the overlap must be the identity matrix
        [1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j
         1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j
         1.+0.j]

        >>> # if two return values requested, it returns also overlap between the m-subspaces
        >>> ovlp, ovlp_m = bas1.overlap(bas2)
        >>> print(np.diag(ovlp_m).round(6))
        [1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j
         1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j 1.+0.j
         1.+0.j]

        """
        howmany = expecting(offset=2)
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        ovlp_k = self.k.overlap(arg.k)
        if howmany == 1:
            return ovlp_k
        elif howmany > 1:
            ovlp_m = self.m.overlap(arg.m)
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
        """Applies unitary transformation to wavefunction sets

        Args:
            krot : numpy.ndarray or tuple (numpy.ndarray, numpy.ndarray)
                Unitary transformation matrix for k-subspace wavefunctions and, if provided,
                a set of corresponding state energies.
                The number of columns in unitary transformation matrix must be equal to
                the number of states self.k.table['c'].shape[1].
            mrot : numpy.ndarray or tuple (numpy.ndarray, np.ndarray)
                Unitary transformation matrix for m-subspace wavefunctions and, if provided,
                a set of corresponding state energies.
                The number of columns in unitary transformation matrix must be equal to
                the number of states self.m.table['c'].shape[1].
            kstat : list
                User-defined assignment of the resulting unitary-transformed k-subspace wavefunctions.
            mstat : list
                User-defined assignment of the resulting unitary-transformed m-subspace wavefunctions.

        Returns:
            Unitary-transformed wavefunction set as PsiTableMK object.
            If state energies are provided in 'krot' or/and 'mrot', they are stored
            in self.k.enr and self.m.enr, respectively.

        Examples:

        >>> # set up symmetric-top basis for J=10, compute matrix representation
        >>> # of rigid-rotor Hamiltonian, compute its eigenvalues and eigenvectors
        >>> J = 10
        >>> bas = SymtopBasis(J)
        >>> hamiltonian = 10 * Jxx(bas) + 12 * Jyy(bas) + 14 * Jzz(bas)
        >>> ham_matrix = bas.overlap(hamiltonian)
        >>> eigenval, eigenvec = np.linalg.eigh(ham_matrix.real)

        >>> # rotate initial basis 'bas' using Hamiltonian eigenvector matrix
        >>> bas2 = bas.rotate(krot=(eigenvec.T, eigenval))

        >>> # compute matrix representation of the above Hamiltonian in the new 'rotated'
        >>> # basis 'bas2', check if it is diagonal, with diagonal elements given by the eigenvalues
        >>> hamiltonian = 10 * Jxx(bas2) + 12 * Jyy(bas2) + 14 * Jzz(bas2)
        >>> ham_matrix = bas2.overlap(hamiltonian)
        >>> diag = np.diag(ham_matrix) # diagonal elements
        >>> max_off_diag = np.max(np.abs(ham_matrix - np.diag(diag))) # maximal off-diagonal element
        >>> print(np.abs(eigenval - diag).round(6)) # check if diagonal is equal to eigenvalues
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        >>> print(max_off_diag.round(10)) # check if max off-diagonal element is close to zero
        0.0

        """
        if mrot is not None:
            res_m = self.m.rotate(mrot, mstat)
        else:
            res_m = self.m
        if krot is not None:
            res_k = self.k.rotate(krot, kstat)
        else:
            res_k = self.k
        return PsiTableMK(res_k, res_m)


    @property
    def nstates(self):
        """ Returns number of basis states (functions) """
        return self.k.table['c'].shape[1]

    @nstates.setter
    def nstates(self):
        raise AttributeError(f"You can't set {retrieve_name(self)}.nstates") from None


    @property
    def enr(self):
        """ Returns energies of basis states """
        nstat = self.k.table['c'].shape[1]
        try:
            enr = self.k.enr
        except AttributeError:
            raise AttributeError(f"Basis states have no associated energies, these are usually assigned " \
                    +f"at the step of unitary rotation, see PsiTableMK.rotate") from None
        return enr

    @enr.setter
    def enr(self):
        raise AttributeError(f"You can't set {retrieve_name(self)}.enr") from None


    @property
    def assign(self):
        """Returns assignment of basis states

        To control the number of primitive functions printed in the state assignment, change
        settings.assign_nprim (=1..6), to change the number of significant digits printed
        for squared modulus of primitive coefficients, change settings.assign_ndig_c2 (=1..10)
        """
        nstat = self.k.table['c'].shape[1]
        assign = self.k.table['stat'][:nstat]
        try:
            sym = self.k.sym[:nstat]
        except AttributeError:
            sym = ["A" for i in range(nstat)]
        return assign

    @assign.setter
    def assign(self):
        raise AttributeError(f"You can't set {retrieve_name(self)}.assign") from None


    @property
    def sym(self):
        """ Returns symmetry of basis states """
        nstat = self.k.table['c'].shape[1]
        try:
            sym = self.k.sym[:nstat]
        except AttributeError:
            sym = ["A" for i in range(nstat)]
        return sym

    @sym.setter
    def sym(self, val):
        if isinstance(val, str):
            nstat = self.k.table['c'].shape[1]
            self.k.sym = np.array([val for istate in range(nstat)])
        else:
            raise TypeError(f"Bad type for symmetry: '{type(val)}'") from None


    @counted
    def store_richmol(self, name, append=False):
        """Stores energies of wavefunction set in Richmol energies file

        To control the number of primitive functions printed for state assignment, change
        settings.assign_nprim (=1..6), to change the number of significant digits printed
        for squared modulus of primitive coefficients, change settings.assign_ndig_c2 (=1..10)

        Args:
            name : str
                Name of the file to store energies into.
            append : str
                If True, the data will be appended to existing file.
        """
        Jk = list(set(J for J in self.k.table['prim'][:,0]))
        Jm = list(set(J for J in self.m.table['prim'][:,0]))
        if len(Jk)>1 or len(Jm)>1:
            raise ValueError(f"Multiple values of J quanta = {Jk} and {Jm} in the k- and m-parts, " \
                    +f"because Richmol matrix elements files are generated for different pairs " \
                    +f"of J quanta, mixing states with different J in the states set is not a good " \
                    +f"idea when it comes to storing state energies and their ID numbers") from None
        else:
            if Jk[0] != Jm[0]:
                raise ValueError(f"Different values of J quanta = {Jk[0]} and {Jm[0]} in the k- " \
                        +f"and m-parts (no idea how this happened)") from None
            J = Jk[0]

        nstat = self.k.table['c'].shape[1]
        assign = self.k.table['stat'][:nstat]

        try:
            enr = self.k.enr[:nstat]
        except AttributeError:
            raise AttributeError(f"States set have no associated energies, these are usually assigned " \
                    +f"at the step of unitary rotation, see PsiTableMK.rotate") from None

        try:
            sym = self.k.sym[:nstat]
        except AttributeError:
            sym = ["A" for i in range(nstat)]

        if self.store_richmol.calls == 1:
            if append==True:
                mode = "a+"
            else:
                mode = "w"
        else:
            mode = "a"

        with open(name, mode) as fl:
            for istat in range(nstat):
                id = istat + 1
                fl.write(" %3i"%J + " %6i"%id + " %4s"%sym[istat] + "  1" + " %20.12f"%enr[istat] \
                        + " ".join(" %3s"%elem for elem in assign[istat]) + "\n")



class SymtopBasis(PsiTableMK):
    """Basis of symmetric top functions for selected J

    Args:
        J : int
            Quantum number of the rotational angular momentum.
        linear : bool
            Set True if molecule is linear, in this case quantum number k is kept at zero.
    """

    def __init__(self, J, linear=False):

        try:
            self.J = int(round(J))
        except TypeError:
            raise TypeError(f"J = '{J}' is not a number") from None
        assert (self.J>=0), f"J = {J} is smaller than zero"

        # generate keys (j,k) for columns representing primitive functions
        if linear:
            prim = [(int(J),int(0))]
        else:
            prim = [(int(J),int(k)) for k in range(-J,J+1)]

        # generate keys (j,k,tau) for rows representing symmetrized functions
        if linear:
            bas = [(int(J),int(0),int(np.fmod(J, 2)))]
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
            j, k, tau : int 
                J, k, and tau quantum numbers, where k can take values between 0 and J
                and tau=0 or 1 is parity defined as (-1)^tau.

        Returns:
            coefs : list
                Wang's symmetrization coefficients, coefs=[c1,c2] for k>0 and coefs=[c1] for k=0.
            kval : list
                List of k-values, kval=[k,-k] for k0 and kval=[k] for k=0.
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



def symmetrize(arg, sym="D2", thresh=1e-12):
    """Returns dictionary of symmetry-adapted objects 'arg' for different irreps (as dict keys)
    of the symmetry group defined by 'sym'

    Args:
        arg : PsiTableMK
            Basis of symmetric-top functions.
        sym : str
            Point symmetry group, defaults to "D2".
        thresh : float
            Threshold for treating symmetrization and basis-set coefs as zero, defaults to 1e-12.
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
                if np.any(abs(proj[:,:,ik]) > thresh) or np.any(abs(proj[:,ik,:]) > thresh):
                    raise ValueError(f"Input set {retrieve_name(arg)} is missing some primitive " \
                            +f"functions that are required for symmetrization, for example, " \
                            +f"(J,k) = {(J,k)} is missing") from None

        for irrep,sym_lab in enumerate(symmetry.sym_lab):
            pmat = np.dot(proj[irrep,:,ind_k], arg.k.table['c'][ind_p,:])
            res[sym_lab].k.table['c'][ind_p,:] = pmat[ind_k,:]

    # remove states with zero coefficients
    remove = []
    for sym_lab,elem in res.items():
        elem.k = elem.k.del_zero_stat(thresh=1e-12)
        if elem.k is None:
            remove.append(sym_lab)
    for sym_lab in remove: del res[sym_lab]

    # check if the total number of states remains the same
    nstat_sym = sum(elem.k.table['c'].shape[1] for elem in res.values()) 
    if nstat_sym != nstat:
        raise RuntimeError(f"Total number of states before symmetrization = {nstat} is different " \
                +f"from the total number of states across all irreps = {nstat_sym}")

    return res



class SymtopSymmetry():
    """Basic class to handle symmetry of rotational wavefunctions,
    generates projection operators for different irreps of chosen symmetry group.
    """

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
    """Basic class to handle matrix elements of laboratory-frame Cartesian tensor operators

    Attributes:
        rank : int
            Rank of tensor operator.
        Us : numpy.complex128 2D array
            Cartesian-to-spherical-tensor transformation matrix.
        Ux : numpy.complex128 2D array
            Spherical-tensor-to-Cartesian transformation matrix (Ux = (Ux^T)^*).
        cart : array of str
            Contains string labels of different Cartesian components in the order corresponding
            to the order of Cartesian components in rows of 'Ux' (columns of 'Us').
        os : [(omega,sigma) for omega in range(nirrep) for sigma in range(-omega,omega+1)]
            List of spherical-tensor indices (omega,sigma) in the order corresponding
            to the spherical-tensor components in columns of 'Ux' (rows of 'Us'),
            'nirrep' here is the number of tensor irreducible representations.
        tens_flat : 1D array
            Contains elements of Cartesian tensor in the molecular frame, flattened in the order
            corresponding to the order of Cartesian components in 'cart'.

    Args:
        arg : np.ndarray, list or tuple
            Contains elements of Cartesian tensor in the molecular frame.
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
            raise TypeError(f"Bad argument type '{type(arg)}' for tensor, expected list, tuple, " \
                    +f"or numpy array") from None
        if not all(dim==3 for dim in tens.shape):
            raise ValueError(f"(Cartesian) tensor has bad shape: '{tens.shape}' != {[3]*tens.ndim}") from None
        if np.all(np.abs(tens) < settings.tens_small_elem):
            raise ValueError(f"Tensor has all its elements equal to zero") from None
        if np.any(np.abs(tens) > settings.tens_large_elem):
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
        """Computes |psi'> = CartTensor|psi>

        Args:
            arg : PsiTableMK)
                |psi>, set of linear combinations of symmetric-top functions.

        Returns:
            res : dict of PsiTableMK
                |psi'>, resulting tensor-projected set of linear combinations of symmetric-top
                functions for different laboratory-frame Cartesian components and different
                irreducible components of the tensor as dict keys (cart,irrep),
                where cart is in self.cart and irrep is in set(omega for (omega,sigma) in self.os).
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
        """Computes matrix elements of Cartesian tensor operator < psi_bra | CartTensor | psi_ket >

        Args:
            psi_bra, psi_ket : PsiTableMK
                Set of linear combinations of symmetric-top functions.

        Returns:
            res : dict of 4D arrays
                Matrix elements between states in psi_bra and psi_ket sets of functions, for different
                laboratory-frame Cartesian components of the tensor (in self.cart) as dict keys.
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
        """Stores tensor matrix elements in Richmol file

        Args:
            psi_bra, psi_ket : PsiTableMK)
                Set of linear combinations of symmetric-top functions.
            name : str
                Name of the tensor operator (single word), defaults to "tens"+str(self.rank).
            fname : str
                Name of the file for storing tensor matrix elements, defaults to 'name'.
                The actual file name will be fname+"_j"+str(J_ket)+"_j"+str(J_bra)+".rchm", where
                J_bra and J_ket are respective J quanta of bra and ket states.
            thresh : float
                Threshold for neglecting matrix elements.
        """
        zero_tol = 1e-14 #small*10

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
                    +f"J quanta = {Jk_bra} and {Jm_bra} in the k- and m-parts, Richmol matrix element " \
                    +f"file cannot be created")
        else:
            if Jk_bra[0] != Jm_bra[0]:
                raise ValueError(f"Different J quanta = {Jk_bra[0]} and {Jm_bra[0]} in the k- and " \
                        +f"m-dependent parts of function {retrieve_name(psi_bra)}")
            J2 = Jk_bra[0]

        Jk_ket = list(set(J for J in psi_ket.k.table['prim'][:,0]))
        Jm_ket = list(set(J for J in psi_ket.m.table['prim'][:,0]))
        if len(Jk_ket)>1 or len(Jm_ket)>1:
            raise ValueError(f"Function {retrieve_name(psi_ket)} couples states with different " \
                    +f"J quanta = {Jk_ket} and {Jm_ket} in the k- and m-parts, Richmol matrix element " \
                    +f"file cannot be created")
        else:
            if Jk_ket[0] != Jm_ket[0]:
                raise ValueError(f"Different J quanta = {Jk_ket[0]} and {Jm_ket[0]} in the k- and " \
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
            raise RuntimeError(f"Elements of the K-tensor are complex-valued, expected purely real " \
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
                raise RuntimeError(f"Elements of the {key} M-tensor are complex-valued, expected purely " \
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
                            fl.write(" %4s"%m1 + " %4s"%m2 + " ".join("  %20.12e"%elem for elem in me) + "\n")
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



class WignerD(CartTensor):
    """ Wigner D-matrix D_{m,k}^{(J)} """
    def __init__(self, J, m, k):
        self.cart = ['0']
        self.tens_flat = np.array([1], dtype=np.float64)
        if m == k:
            self.os = [(J,m)]
            self.Ux = np.array([[1]], dtype=np.complex128)
            self.Us = np.array([[1]], dtype=np.complex128)
        else:
            self.os = [(J,m), (J,k)]
            self.Ux = np.array([[1, 0]], dtype=np.complex128)
            self.Us = np.array([[0],[1]], dtype=np.complex128)



class WignerExpand(CartTensor):
    """Arbitrary function of two Euler angles theta and phi expanded in terms of Wigner D-functions

    Args:
        func(theta, phi)
            Function of two Euler angles theta and phi.
        jmax : int
            Max value of the quantum number J spanned by the set of Wigner D-functions
            D_{m,k}^{(J)} (m,k=-J..J) used in the expansion of function 'func'.
        npoints_leb : int
            Size of the Lebedev angular quadrature used for numerical integration.
        thresh : float
            Threshold for neglecting small expansion coefficients.
    """

    def __init__(self, func, jmax=100, npoints_leb=5810, tol=1e-12):

        self.cart = ['0']
        self.rank = 0
        self.tens_flat = np.array([1], dtype=np.float64)

        # expansion coefficients

        overlap_me = overlap_integrals_func_symtop(func, jmax, npoints_leb)

        wcoef = {}
        nirrep = 0
        omega = []
        for j in range(jmax+1):
            fac = np.sqrt((2*j+1)/(8.0*np.pi**2))
            me = np.array([overlap_me[im,j] * fac for m,im in zip(range(-jmax,jmax+1),range(2*jmax+1)) if abs(m) <= j])
            if np.all(abs(me) < thresh): continue
            wcoef[j] = me
            nirrep += 1
            omega.append(j)

        # Cartesian <--> spherical tensor transformation matrices

        self.os = [(o,s) for o in omega for s in range(-o,o+1)]
        dim1 = 1
        dim2 = len(self.os)
        self.Ux = np.zeros((dim1,dim2), dtype=np.complex128)
        self.Us = np.zeros((dim2,dim1), dtype=np.complex128)

        ind_sigma = {(omega,sigma) : [s for s in range(-omega,omega+1)].index(sigma) \
                     for (omega,sigma) in self.os}

        for i,(omega,sigma) in enumerate(self.os):
            isigma = ind_sigma[(omega,sigma)]
            self.Ux[0,i] = wcoef[omega][isigma]
            if sigma == 0:
                self.Us[i,0] = 1.0


    def overlap(self, func, jmax=100, npoints=5810):
        """Computes overlap integrals between symmetric-top functions and input function 'func',
        of two Euler angles theta and phi, using Lebedev quadrature

        Args:
            func(theta, phi)
                Function of two Euler angles theta and phi.
            jmax : int
                Max value of the quantum number J spanned by the set of symmetric-top functions.
            npoints : int
                Size of the Lebedev angular quadrature used for numerical integration.

        Returns:
            ovlp : array (2*jmax+1,jmax)
                Overlap integrals ovlp(im,J) = <J,k=0,m|func>,
                where m,im in zip(range(-J,J+1),range(2J+1)) for J in range(0,jmax+1).
        """
        # initialize Lebedev quadrature

        npoints_c = c_int(npoints)
        grid = np.asfortranarray(np.zeros((2,npoints), dtype=np.float64))
        weight = np.zeros(npoints, dtype=np.float64)

        fsymtop.angular_lebedev.argtypes = [ \
            c_int, \
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='C') ]

        fsymtop.angular_lebedev.restype = None

        fsymtop.angular_lebedev( \
            npoints_c, \
            grid, \
            weight )

        # compute symmetric-top functions |j,k=0,m>
        # for j=0..jmax and m=-j..j on the Lebedev grid for theta and phi angles

        jmax_c = c_int(jmax)
        symtop_grid_r = np.asfortranarray(np.zeros((2*jmax+1,jmax+1,npoints), dtype=np.float64))
        symtop_grid_i = np.asfortranarray(np.zeros((2*jmax+1,jmax+1,npoints), dtype=np.float64))

        fsymtop.symtop_2d_grid_theta_phi.argtypes = [ \
            c_int, \
            c_int, \
            np.ctypeslib.ndpointer(np.float64, ndim=2, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=3, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=3, flags='F') ]

        fsymtop.symtop_2d_grid_theta_phi.restype = None

        fsymtop.symtop_2d_grid_theta_phi( \
            npoints_c, \
            jmax_c, grid, \
            symtop_grid_r, \
            symtop_grid_i )

        grid = grid.reshape((2,npoints)) # grid[0:1,ipoint] = theta,phi
        symtop_grid = np.array( symtop_grid_r.reshape((2*jmax+1,jmax+1,npoints)) \
                              - symtop_grid_i.reshape((2*jmax+1,jmax+1,npoints))*1j, \
                              dtype=np.complex128 )  # symtop_grid[m,j,ipoint] = |j,k=0,m>

        # input function on 2D grid of theta and phi angles
        # and matrix element <j,k=0,m|user_func> using Lebedev quadrature

        func_times_weight = np.array( [func(theta,phi) * wght \
                                       for theta,phi,wght in zip(grid[0,:],grid[1,:],weight)], \
                                    dtype=np.float64 )

        ovlp = np.sum(symtop_grid * func_times_weight, axis=2)

        twopi = np.pi*2
        ovlp *= twopi  # factor 2Pi comes from the implicit integration over the third Euler angle chi

        return ovlp



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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
