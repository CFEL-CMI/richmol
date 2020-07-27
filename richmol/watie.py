"""Tools for computing rigid-molecule rotational energy levels, wave functions, and matrix elements
of various rotation-dependent operators, such as laboratory-frame Cartesian tensor operators.
"""
import numpy as np
import math
import sys
import os
from mendeleev import element
import re
import inspect
import copy
from ctypes import CDLL, c_double, c_int, POINTER, RTLD_GLOBAL
import warnings
from richmol.pywigxjpf import wig_table_init, wig_temp_init, wig3jj, wig_temp_free, wig_table_free


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

        # update a string that keeps track of all frame rotations
        try:
            self.frame_type += "," + arg
        except AttributeError:
            self.frame_type = arg


    @property
    def ABC(self):
        """ Returns A, B, C rotational constants (where A>=B>=C), in units cm^-1 """
        imom = self.imom()
        tol = 1e-12
        if np.any(np.abs(np.diag(np.diag(imom))-imom)>tol):
            raise RuntimeError("Cannot compute rotational constants for the current frame = " \
                    +f"'{self.frame_type}', the inertia tensor is not diagonal = {imom}") from None
        convert_to_cm = planck * avogno * 1e+16 / (8.0 * np.pi * np.pi * vellgt) 
        ABC = [convert_to_cm/val for val in np.diag(imom)]
        return ABC


    @ABC.setter
    def ABC(self, val):
        pass


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



class SymtopBasis():
    """Basis of symmetric top functions for selected J

    Args:
        J (int): Quantum number of the rotational angular momentum.
        linear (bool): True if molecule is linear, in this case quantum number k is set to zero.
    """

    def __init__(self, J, linear=False):

        try:
            self.J = int(round(J))
        except TypeError:
            raise TypeError(f"J = '{J}' is not a number") from None

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

        assert (len(prim)==len(bas)),"len(prim)!=len(bas)"

        # create empty table of linear coefficients
        nbas = len(bas)
        assert (nbas>0), f"number of basis functions nbas = {nbas} for J = {j}"
        dt = [('jk', 'i4', (2)), ('c', np.complex128, [nbas])]
        self.jk_table = np.zeros(nbas, dtype=dt)
        self.jk_table['jk'] = prim
        self.jkt = np.array(bas)
        self.dim = nbas
        self.dim_k = nbas

        # generate Wang-type linear combinations
        for ibas,(J,k,tau) in enumerate(bas):
            coefs, kval = self.wang_coefs(J, k, tau)
            for kk,cc in zip(kval,coefs):
                iprim = np.where((self.jk_table['jk']==(J,kk)).all(axis=1))[0][0]
                self.jk_table['c'][iprim,ibas] = cc

        # Add m-dependent part of the basis
        # generate keys (j,m) and create a table of coefficients
        prim_m = [(int(J),int(m)) for m in range(-J,J+1)]
        nbas_m = len(prim_m)
        dt = [('jm', 'i4', (2)), ('c', np.complex128, [nbas_m])]
        self.jm_table = np.zeros(nbas, dtype=dt)
        self.jm_table['jm'] = prim_m
        for i in range(nbas_m):
            self.jm_table['c'][i,i] = 1
        self.dim_m = nbas_m


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
        assert (k>=0),f"k = {k} < 0"
        assert (j>=0),f"J = {j} < 0"
        assert (k<=j),f"k = {k} > J = {J}"
        assert (tau<=1 and tau>=0),f"tau = {tau} is not equal to 0 or 1"

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


    def overlap_k(self, arg):
        """ Computes overlap matrix elements <self|arg> in the k-dependent space of quanta
        ('m' quanta are ignored)

        Args:
            arg (SymtopBasis, J, or similar): represents set of linear combinations of symmetric-top
                functions, can be either a basis set, i.e., SymtopBasis(), or a result of an action
                of angular momentum or Cartesian-tensor operator(s) on a basis set, i.e., J() class.
                It must contain 'jk_table' attribute.
        Returns:
            res (np.ndarray): overlap matrix elements <self|arg> between self and arg sets of functions.
        """
        try:
            table_ket = arg.jk_table
        except AttributeError:
            raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jk_table'") from None
        table_bra = self.jk_table
        jk_ket = [tuple(x) for x in table_ket['jk']]
        jk_bra = [tuple(x) for x in table_bra['jk']]
        coefs_ket = table_ket['c']
        coefs_bra = table_bra['c'].conj().T
        # find overlapping jk quanta in both bra and ket lists
        both = set(jk_bra).intersection(jk_ket)
        ind_bra = [jk_bra.index(x) for x in both]
        ind_ket = [jk_ket.index(x) for x in both]
        if len(both)==0:
            warnings.warn(f"functions {retrieve_name(self)} and {retrieve_name(arg)} have no " \
                +f"overlapping J,k quanta, the overlap is zero!")
        # dot product across overlapping jk quanta
        res = np.dot(coefs_bra[:,ind_bra], coefs_ket[ind_ket,:])
        return res


    overlap = overlap_k


    def overlap_m(self, arg):
        """ Computes overlap matrix elements <self|arg> in the m-dependent space of quanta
        ('k' quanta are ignored)

        Args:
            arg (SymtopBasis, J, or similar): represents set of linear combinations of symmetric-top
                functions, can be either a basis set, i.e., SymtopBasis(), or a result of an action
                of angular momentum or Cartesian-tensor operator(s) on a basis set, i.e., J() class.
                It must contain 'jm_table' attribute.
        Returns:
            res (np.ndarray): overlap matrix elements <self|arg> between self and arg sets of functions.
        """
        try:
            table_ket = arg.jm_table
        except AttributeError:
            raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jm_table'") from None
        table_bra = self.jm_table
        jm_ket = [tuple(x) for x in table_ket['jm']]
        jm_bra = [tuple(x) for x in table_bra['jm']]
        coefs_ket = table_ket['c']
        coefs_bra = table_bra['c'].conj().T
        # find overlapping jm quanta in both bra and ket lists
        both = set(jm_bra).intersection(jm_ket)
        ind_bra = [jm_bra.index(x) for x in both]
        ind_ket = [jm_ket.index(x) for x in both]
        if len(both)==0:
            warnings.warn(f"functions {retrieve_name(self)} and {retrieve_name(arg)} have no " \
                +f"overlapping J,m quanta, the overlap is zero!")
        # dot product across overlapping jm quanta
        res = np.dot(coefs_bra[:,ind_bra], coefs_ket[ind_ket,:])
        return res


def symmetrize(arg, sym="D2"):
    """Returns dictionary of symmetry-adapted objects 'arg' for different irreps (as dict keys)
    of the symmetry group defined by 'sym'

    Args:
        arg (SymtopBasis): Basis of symmetric-top functions for selected J.
        sym (str): Point symmetry group, defaults to "D2".
    """
    try:
        sym_ = getattr(sys.modules[__name__], sym)
    except:
        raise NotImplementedError(f"symmetry '{sym}' is not implemented") from None

    if isinstance(arg, SymtopBasis):
        bas = arg
        J = bas.J
        nbas = len(bas.jk_table['jk'])
        symmetry = sym_(J)
        res = {sym_lab : copy.deepcopy(bas) for sym_lab in symmetry.sym_lab}

        nbas_sum = 0
        for irrep,sym_lab in enumerate(symmetry.sym_lab):
            jk_table = symmetry.proj(bas.jk_table, irrep)
            ind0 = [ifunc for ifunc in range(nbas) if all(abs(val)<small*max(1,J) for val in jk_table['c'][:,ifunc])]
            nbas_irrep = nbas - len(ind0)
            nbas_sum += nbas_irrep
            res[sym_lab].jk_table = np.zeros(nbas, dtype=[('jk', 'i4', (2)), ('c', np.complex128, [nbas_irrep])])
            res[sym_lab].jk_table['c'] = np.delete(jk_table['c'], ind0, 1)
            res[sym_lab].jk_table['jk'] = jk_table['jk']
            res[sym_lab].jkt = np.delete(bas.jkt, ind0, 0) #[jkt for ind,jkt in enumerate(bas.jkt) if ind not in ind0]
            res[sym_lab].dim = nbas_irrep
        assert (nbas==nbas_sum), f"nbas = {nbas} is not equal to nbas_sum = {nbas_sum}"
    else:
        raise TypeError(f"Unsupported type of argument: 'type(arg)'")

    return res



class SymtopSymmetry():

    def __init__(self, J):

        self.J = J

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


    def proj(self, jk_table, irrep):

        assert (irrep in range(self.nirrep)), f"irrep = {irrep} is not in range({self.nirrep})"
        assert(all(self.J==jj[0] for jj in jk_table['jk'])), f"J quanta in 'jk_table' are different " \
                +f"from self.J = {self.J}"

        nbas = len(jk_table['jk'])
        J = self.J
        Proj = np.zeros((nbas,nbas), dtype=np.complex128)

        for ioper in range(self.noper):
            Chi = float(self.characters[ioper,irrep]) # needs to be complex conjugate, fix if characters can be imaginary
            fac = Chi/self.noper
            for i,jk1 in enumerate(jk_table['jk']):
                for j,jk2 in enumerate(jk_table['jk']):
                    k1 = jk1[1]
                    k2 = jk2[1]
                    Proj[i,j] += fac * self.coefs[ioper,k1+J,k2+J,0]

        res = copy.deepcopy(jk_table)
        res['c'] = np.dot(Proj, jk_table['c'])
        return res



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



class J():
    """ Basic class for rotational angular momentum operators, acting on both k- and m-space quanta """

    def __init__(self, *args, **kwargs):
        for arg in args:
            try:
                x = arg.jk_table
            except AttributeError:
                raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jk_table'") from None
            else:
               self.jk_table = arg.jk_table.copy()
            try:
                x = arg.jm_table
            except AttributeError:
                raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jm_table'") from None
            else:
               self.jm_table = arg.jm_table.copy()
        if 'jk_table' in kwargs:
            self.jk_table = copy.deepcopy(kwargs['jk_table'])
            try:
                self.jm_table = copy.deepcopy(kwargs['jm_table'])
            except KeyError:
                raise KeyError(f"missing argument 'jm_table' must always be provided together with 'jk_table'") from None


    def __add__(self, arg):
        try:
            x = self.jk_table
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'jk_table'") from None
        try:
            x = self.jm_table
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'jm_table'") from None

        if isinstance(arg, J):
            # k-part
            try:
                x = arg.jk_table
            except AttributeError:
                raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jk_table'") from None
            if not (self.jk_table['jk']==arg.jk_table['jk']).all():
                raise ValueError(f"{self.__class__.__name__} and {arg.__class__.__name__} " \
                    +f"work on different basis sets ('jk_table' attributes do not match)") from None
            jk_table = self.jk_table.copy()
            for ielem,(j,k) in enumerate(jk_table['jk']):
                jelem = np.where((arg.jk_table['jk']==(j,k)).all(axis=1))[0][0]
                jk_table['c'][ielem,:] += arg.jk_table['c'][jelem,:]
            # m-part
            try:
                x = arg.jm_table
            except AttributeError:
                raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jm_table'") from None
            if not (self.jm_table['jm']==arg.jm_table['jm']).all():
                raise ValueError(f"{self.__class__.__name__} and {arg.__class__.__name__} " \
                    +f"work on different basis sets ('jm_table' attributes do not match)") from None
            jm_table = self.jm_table.copy()
            for ielem,(j,m) in enumerate(jm_table['jm']):
                jelem = np.where((arg.jm_table['jm']==(j,m)).all(axis=1))[0][0]
                jm_table['c'][ielem,:] += arg.jm_table['c'][jelem,:]
            res = J(jk_table=jk_table, jm_table=jm_table)

        else:
            raise TypeError(f"unsupported operand type(s) for '+': {self.__class__.__name__} and " \
                    +f"{arg.__class__.__name__}") from None
        return res


    def __sub__(self, arg):
        try:
            x = self.jk_table
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'jk_table'") from None
        try:
            x = self.jm_table
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'jm_table'") from None

        if isinstance(arg, J):
            # k-part
            try:
                x = arg.jk_table
            except AttributeError:
                raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jk_table'") from None
            if not (self.jk_table['jk']==arg.jk_table['jk']).all():
                raise ValueError(f"{self.__class__.__name__} and {arg.__class__.__name__} " \
                    +f"work on different basis sets ('jk_table' attributes do not match)") from None
            jk_table = self.jk_table.copy()
            for ielem,(j,k) in enumerate(jk_table['jk']):
                jelem = np.where((arg.jk_table['jk']==(j,k)).all(axis=1))[0][0]
                jk_table['c'][ielem,:] -= arg.jk_table['c'][jelem,:]
            # m-part
            try:
                x = arg.jm_table
            except AttributeError:
                raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jm_table'") from None
            if not (self.jm_table['jm']==arg.jm_table['jm']).all():
                raise ValueError(f"{self.__class__.__name__} and {arg.__class__.__name__} " \
                    +f"work on different basis sets ('jm_table' attributes do not match)") from None
            jm_table = self.jm_table.copy()
            for ielem,(j,m) in enumerate(jm_table['jm']):
                jelem = np.where((arg.jm_table['jm']==(j,m)).all(axis=1))[0][0]
                jm_table['c'][ielem,:] -= arg.jm_table['c'][jelem,:]
            res = J(jk_table=jk_table, jm_table=jm_table)

        else:
            raise TypeError(f"unsupported operand type(s) for '-': {self.__class__.__name__} and " \
                    +f"{arg.__class__.__name__}") from None
        return res


    def __mul__(self, arg):
        scalar = (int, float, complex, np.int, np.int8, np.int16, np.int32, 
                  np.int64, np.float, np.float16, np.float32, np.float64,
                  np.complex64, np.complex128)
        if isinstance(arg, J):
            try:
                x = arg.jk_table
            except AttributeError:
                raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jk_table'") from None
            try:
                x = arg.jm_table
            except AttributeError:
                raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jm_table'") from None
            res = self.__class__(arg)
        elif isinstance(arg, scalar):
            try:
                x = self.jk_table
            except AttributeError:
                raise AttributeError(f"{self.__class__.__name__} has no attribute 'jk_table'") from None
            try:
                x = self.jm_table
            except AttributeError:
                raise AttributeError(f"{self.__class__.__name__} has no attribute 'jm_table'") from None
            jk_table = self.jk_table.copy()
            jk_table['c'] *= arg
            jm_table = self.jm_table.copy()
            res = J(jk_table=jk_table, jm_table=jm_table)
        else:
            raise TypeError(f"unsupported operand type(s) for '*': {self.__class__.__name__} and " \
                    +f"{arg.__class__.__name__}") from None
        return res


    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__



class mol_Jp(J):
    """ Molecular-frame J+ = Jx + iJy """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            jk_table = self.jk_table.copy()
            jk_table['c'] = 0
            for ielem,(j,k) in enumerate(jk_table['jk']):
                if abs(k-1)<=j:
                    fac = math.sqrt( j*(j+1)-k*(k-1) )
                    k2 = k-1
                    jelem = np.where((jk_table['jk']==(j,k2)).all(axis=1))[0][0]
                    jk_table['c'][jelem,:] = self.jk_table['c'][ielem,:] * fac
            self.jk_table = jk_table
        except AttributeError:
            pass

Jp = mol_Jp


class mol_Jm(J):
    """ Molecular-frame J- = Jx - iJy """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            jk_table = self.jk_table.copy()
            jk_table['c'] = 0
            for ielem,(j,k) in enumerate(jk_table['jk']):
                if abs(k+1)<=j:
                    fac = math.sqrt( j*(j+1)-k*(k+1) )
                    k2 = k+1
                    jelem = np.where((jk_table['jk']==(j,k2)).all(axis=1))[0][0]
                    jk_table['c'][jelem,:] = self.jk_table['c'][ielem,:] * fac
            self.jk_table = jk_table
        except AttributeError:
            pass

Jm = mol_Jm


class mol_Jz(J):
    """ Molecular-frame Jz """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            for ielem,(j,k) in enumerate(self.jk_table['jk']):
                self.jk_table['c'][ielem,:] = self.jk_table['c'][ielem,:] * k
        except AttributeError:
            pass

Jz = mol_Jz


class mol_JJ(J):
    """ Molecular-frame J^2 """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            for ielem,(j,k) in enumerate(self.jk_table['jk']):
                self.jk_table['c'][ielem,:] = self.jk_table['c'][ielem,:] * j*(j+1)
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
    """ Basic class for laboratory-frame Cartesian tensor operators """

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
        """ Initialises laboratory-fixed Cartesian tensor given molecule-fixed tensor """

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


    def __call__(self, arg, tol=1e-14):
        """ Computes |psi'> = CartTensor|psi>

        Returns result in a split-form of K-tensor and M-tensor acting separately
        on J,k and J,m sets of quantum numbers, respectively.

        The output objects are K[omega] and M[(alpha,omega)], where 'omega' denotes the irreducible
        tensor rank and 'alpha' denotes the lab-fixed Cartesian component of tensor.
        The result CartTensor|psi> can be computed as sum_omega( K[omega] * M[(alpha,omega)] )
        for selected lab-fixed Cartesian component 'alpha'.

        Args:
            arg (SymtopBasis, J, or similar): set of linear combinations of symmetric-top functions,
                it must contain 'jk_table' and 'jm_table' attributes.
            tol (float): tolerance for neglecting projection coefficients, defaults to 1e-12.
        Returns:
            res_k (dict): K[omega]
            res_m (dict): M[(alpha,omega)]
        """
        irreps = set(omega for (omega,sigma) in self.os)
        dj_max = max(irreps)    # selection rules |j1-j2|<=omega
        os_ind = {omega : [ind for ind,(o,s) in enumerate(self.os) if o==omega] for omega in irreps}

        # input set of basis functions
        jk_table = arg.jk_table
        jm_table = arg.jm_table

        # generate quanta for new tensor-projected set of basis functions (jk_table2, jm_table2)

        jmin = min([min(j for (j,k) in jk_table['jk']), min(j for (j,m) in jm_table['jm'])])
        jmax = max([max(j for (j,k) in jk_table['jk']), max(j for (j,m) in jm_table['jm'])])
        nbas_k = jk_table['c'].shape[1]
        nbas_m = jm_table['c'].shape[1]

        jk = [(int(J),int(k)) for J in range(max([0,jmin-dj_max]),jmax+1+dj_max) for k in range(-J,J+1)]
        jm = [(int(J),int(m)) for J in range(max([0,jmin-dj_max]),jmax+1+dj_max) for m in range(-J,J+1)]
        nprim_k = len(jk)
        nprim_m = len(jm)

        jk_table2 = np.zeros(nprim_k, dtype=[('jk', 'i4', (2)), ('c', np.complex128, [nbas_k])])
        jk_table2['jk'] = jk
        jm_table2 = np.zeros(nprim_m, dtype=[('jm', 'i4', (2)), ('c', np.complex128, [nbas_m])])
        jm_table2['jm'] = jm

        # output objects
        res_k = {irrep : J(jk_table=jk_table2, jm_table=None) for irrep in irreps}
        res_m = {(cart,irrep) : J(jk_table=None, jm_table=jm_table2) for irrep in irreps for cart in self.cart}

        # some initializations in pywigxjpf module for computing 3j symbols
        wig_table_init((jmax+dj_max)*2, 3)
        wig_temp_init((jmax+dj_max)*2)

        # compute K|psi>
        for ind1,(j1,k1) in enumerate(jk_table['jk']):
            for ind2,(j2,k2) in enumerate(jk_table2['jk']):
                fac = (-1)**abs(k2)
                # compute <j2,k2|K-tensor|j1,k1>
                threeJ = np.array([wig3jj([j1*2, o*2, j2*2, k1*2, s*2, -k2*2]) for (o,s) in self.os])
                for irrep in irreps:
                    ind = os_ind[irrep]
                    me = np.dot(threeJ[ind], np.dot(self.Us[ind,:], self.tens_flat)) * fac
                    res_k[irrep].jk_table['c'][ind2,:] += me * jk_table['c'][ind1,:]

        # compute M|psi>
        for ind1,(j1,m1) in enumerate(jm_table['jm']):
            for ind2,(j2,m2) in enumerate(jm_table2['jm']):
                fac = np.sqrt((2*j1+1)*(2*j2+1)) * (-1)**abs(m2)
                # compute <j2,m2|M-tensor|j1,m1>
                threeJ = np.array([wig3jj([j1*2, o*2, j2*2, m1*2, s*2, -m2*2]) for (o,s) in self.os])
                for irrep in irreps:
                    ind = os_ind[irrep]
                    me = np.dot(self.Ux[:,ind], threeJ[ind]) * fac
                    for icart,cart in enumerate(self.cart):
                        res_m[(cart,irrep)].jm_table['c'][ind2,:] += me[icart] * jm_table['c'][ind1,:]

        # free memory in pywigxjpf module
        wig_temp_free()
        wig_table_free()

        # experimental: delete zeros in K-tensor
        for key in res_k.keys():
            tab = res_k[key].jk_table.copy()
            nbas = tab['c'].shape[1]
            nprim = tab['c'].shape[0]
            ind0 = [ifunc for ifunc in range(nprim) if all(abs(val)<tol for val in tab['c'][ifunc,:])]
            nprim_new = nprim - len(ind0)
            res_k[key].jk_table = np.zeros(nprim_new, dtype=[('jk', 'i4', (2)), ('c', np.complex128, [nbas])])
            res_k[key].jk_table['c'] = np.delete(tab['c'], ind0, 0)
            res_k[key].jk_table['jk'] = np.delete(tab['jk'], ind0, 0)

        # experimental: delete zeros in M-tensor
        for key in res_m.keys():
            tab = res_m[key].jm_table.copy()
            nbas = tab['c'].shape[1]
            nprim = tab['c'].shape[0]
            ind0 = [ifunc for ifunc in range(nprim) if all(abs(val)<tol for val in tab['c'][ifunc,:])]
            nprim_new = nprim - len(ind0)
            res_m[key].jm_table = np.zeros(nprim_new, dtype=[('jm', 'i4', (2)), ('c', np.complex128, [nbas])])
            res_m[key].jm_table['c'] = np.delete(tab['c'], ind0, 0)
            res_m[key].jm_table['jm'] = np.delete(tab['jm'], ind0, 0)

        return res_k, res_m


    def me(self, psi_bra, psi_ket):
        """ Computes matrix elements of Cartesian tensor operator <psi_bra|CartTensor|psi_ket> """
        try:
            x = psi_bra.jk_table
            y = psi_bra.jm_table
        except AttributeError:
            raise AttributeError(f"{psi_bra.__class__.__name__} has no attribute 'jk_table' " \
                    +f"or/and 'jm_table'") from None
        try:
            x = psi_ket.jk_table
            y = psi_ket.jm_table
        except AttributeError:
            raise AttributeError(f"{psi_ket.__class__.__name__} has no attribute 'jk_table' " \
                    +f"or/and 'jm_table'") from None

        # K[omega]|psi> and M[(cart,omega)]|psi>
        ktens, mtens = self(psi_ket)

        nirrep = len(ktens)
        nirrep_ = len(set(irrep for (cart,irrep) in mtens.keys()))
        assert (nirrep==nirrep_),f"number of irreps in K-tensor and M-tensor do not agree: " \
                +f"{nirrep} != {nirrep_}"

        ncart = len(set(cart for (cart,irrep) in mtens.keys()))
        ncart_ = len(self.cart)
        assert (ncart==ncart_), f"number of Cartesian components in M-tensor and CartTensor " \
                +f"do not agree: {ncart} != {ncart_}"

        dim_bra_k = psi_bra.jk_table['c'].shape[1]
        dim_ket_k = psi_ket.jk_table['c'].shape[1]
        dim_bra_m = psi_bra.jm_table['c'].shape[1]
        dim_ket_m = psi_ket.jm_table['c'].shape[1]

        res_k = np.zeros((nirrep, dim_bra_k, dim_ket_k), dtype=np.complex128)
        res_m = np.zeros((nirrep, ncart, dim_bra_m, dim_ket_m), dtype=np.complex128)
        irreps = list(set(omega for (omega,sigma) in self.os))

        # <psi'|K[omega]|psi>
        for irrep,kt in ktens.items():
            irrep_ = irreps.index(irrep)
            res_k[irrep_,:,:] = psi_bra.overlap_k(kt)

        # <psi'|M[(cart,omega)]|psi>
        for (cart,irrep),mt in mtens.items():
            irrep_ = irreps.index(irrep)
            icart = self.cart.index(cart)
            res_m[irrep_,icart,:,:] = psi_bra.overlap_m(mt)

        # <psi'|CartTensor|psi> = sum_omega( <psi'|M[(cart,omega)]|psi> * <psi'|K[omega]|psi> )
        return np.einsum('ijkl,abc->jkblc', res_m, res_k) # [icart,m2,k2,m1,k1]



def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

