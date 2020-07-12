"""Tools for computing rigid-molecule rotational energy levels, wave functions, and matrix elements
of various rotation-dependent operators, such as laboratory-frame Cartesian tensor operators.
"""
import numpy as np
import sys
from mendeleev import element
import re
import inspect
import copy


bohr_to_angstrom_ = 0.529177249    # converts distances from atomic units to Angstrom
planck_ = 6.62606896e-27           # Plank constant in erg a second
avogno_ = 6.0221415e+23            # Avogadro constant
vellgt_ = 2.99792458e+10           # Speed of light constant in centimetres per second
boltz_ = 1.380658e-16              # Boltzmann constant in erg per Kelvin


def atom_data_from_label(atom_label):
    """ Given atom label, returns its properties, e.g. mass.
    Combine atom labels with integer mass numbers to specify different isotopologues,
    e.g., 'H2' (deuterium), 'C13', 'N15', etc.
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
        raise ValueError(f"Isotope {mass_number} of the element {atom} is not found in mendeleev " \
                +f"database") from None
    mass = [iso.mass for iso in elem.isotopes][ind]
    return {"mass":mass}


class RigidMolecule():

    @property
    def XYZ(self):
        try:
            x = self.atoms
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'XYZ'") from None
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
                        to_angstrom = bohr_to_angstrom_
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
            raise TypeError(f"Unsupported argument type {type(arg)} for atoms' specification") from None

        self.atoms = np.array( [(lab, mass, cart) for lab,mass,cart in zip(label,mass,xyz)], \
                               dtype=[('label','U10'),('mass','f8'),('xyz','f8',(3))] )


    @property
    def tensor(self):
        """ Returns a dict of all initialised tensors. """
        try:
            x = self.tens
        except AttributeError:
            raise AttributeError(f"'{retrieve_name(self)}' has no attribute 'tensor'") from None
        tens = copy.copy(self.tens)
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
                array = np.einsum(key, *rot_mat, array)
                tens[name] = array
        except AttributeError:
            pass
        return tens


    @tensor.setter
    def tensor(self, arg):
        """Defines Cartesian tensor.

        Examples:
            tensor = ("mu", [0.5, -0.1, 0]) to add a permanent dipole moment vector.
            tensor = ("my_alpha", [[10,0,0],[0,20,0],[0,0,30]]) to add a rank-2 tensor, such as,
                for example, polarizability.
        """
        try:
            name, tens = arg
        except ValueError:
            raise ValueError(f"Pass an iterable with two items, tensor = ('name', tensor)") from None
        if not isinstance(name, str):
            raise TypeError(f"Unsupported argument type {type(name)} for tensor name, must be 'str'") from None
        if isinstance(tens, (tuple, list)):
            tens = np.array(tens)
        elif isinstance(arg, np.ndarray):
            pass
        else:
            raise TypeError(f"Unsupported argument type {type(tens)} for tensor values, " \
                    +f"must be one of: 'list', 'np.ndarray'") from None
        if not all(dim==tens.shape[0] for dim in tens.shape):
            raise ValueError(f"Tensor dimensions are not all equal, shape = {tens.shape}") from None
        try:
            x = self.tens
        except AttributeError:
            self.tens = {}
        if name in self.tens:
            raise ValueError(f"Tensor with name '{name}' already exists")
        self.tens[name] = tens


    @property
    def frame(self):
        """Returns type of molecular frame (str) and frame rotation matrix (array (3,3)).
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
        """Defines rotation of molecular frame.
        Cartesian coordinates of atoms and all Cartesian tensors will be rotated to a new frame.

        Examples:
            frame = "pas" will rotate to a principal axes system with x,y,z = a,b,c.
            frame = "tens_name" will rotate to a principal axes system of a tensor with the name 
                "tens_name", this tensor must be initialized before using the command
                'tensor = ("name", array)'.
            frame = "zxy" will permute axes x-->z, y-->x, and y-->z.
            frame = "zxy,pas" will rotate to "pas" and permute x-->z, y-->x, and y-->z.
        """
        if isinstance(arg, str):
            rotmat0 = np.eye(3, dtype=np.float64)
            for fr in reversed([v.strip() for v in arg.split(',')]):
                if fr.lower()=="pas":
                    # principal axes system
                    diag, rotmat = np.linalg.eigh(self.imom())
                    rotmat0 = np.dot(np.transpose(rotmat), rotmat0)
                    self.frame_diag = diag
                    convert_to_cm = planck_ * avogno_ * 1e+16 / (8.0 * np.pi * np.pi * vellgt_) 
                    self.ABC = [convert_to_cm/val for val in diag]
                elif "".join(sorted(fr.lower()))=="xyz":
                    # axes permutation
                    ind = [("x","y","z").index(s) for s in list(fr.lower())]
                    rotmat = np.zeros((3,3), dtype=np.float64)
                    for i in range(3):
                        rotmat[i,ind[i]] = 1.0
                    rotmat0 = np.dot(rotmat, rotmat0)
                else:
                    # axes system defined by to-diagonal rotation of arbitrary rank-2 (3x3) tensor
                    # the tensor must be initialized before, with the name matching fr
                    try:
                        tens = self.tensor[fr]
                    except (AttributeError, ValueError):
                        raise AttributeError(f"Tensor '{fr}' intended for frame rotation was not " \
                                +f"initialised, try {retrieve_name(self)}.tensor " \
                                +f"= ('{fr}', [[x,x,x],[x,x,x],[x,x,x]])") from None
                    if tens.ndim!=2:
                        raise ValueError(f"Tensor '{fr}' intended for frame rotation has a bad rank: " \
                                +f"{tens.ndim} != 2") from None
                    if tens.shape!=(3,3):
                        raise ValueError(f"Tensor '{fr}' intended for frame rotation has a bad shape: " \
                                +f"{tens.shape} != (3, 3)") from None
                    diag, rotmat = np.linalg.eigh(tens)
                    rotmat0 = np.dot(np.transpose(rotmat), rotmat0)
                    self.frame_diag = diag
        else:
            raise TypeError(f"Unsupported argument type {type(arg)} for frame") from None
        try:
            self.frame_rotation = np.dot(rotmat0, self.frame_rotation)
        except AttributeError:
            self.frame_rotation = np.transpose(rotmat0)
        try:
            self.frame_type += "," + arg
        except AttributeError:
            self.frame_type = arg


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


def retrieve_name(var):
    """Gets the name of var. Does it from the out most frame inner-wards.
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]



if __name__=="__main__":


    camphor = RigidMolecule()

    camphor.XYZ = ("angstrom", \
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

    print(camphor.XYZ)
    camphor.tensor = ("alpha", ((1,2,3),(1,2,3),(3,4,5)))
    camphor.tensor = ("beta", ((1,2,3),(1,2,3),(3,4,5)))
    camphor.tensor = ("gamma", ((1,2,3),(1,2,3),(3,4,5)))
    print(camphor.tensor)
    camphor.frame = "zyx"
    camphor.frame = "pas"
    a,b = camphor.frame
    print(a,b)
