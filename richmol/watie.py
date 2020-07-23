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


bohr_to_angstrom = 0.529177249    # converts distances from atomic units to Angstrom
planck = 6.62606896e-27           # Plank constant in erg a second
avogno = 6.0221415e+23            # Avogadro constant
vellgt = 2.99792458e+10           # Speed of light constant in centimetres per second
boltz = 1.380658e-16              # Boltzmann constant in erg per Kelvin
small = np.finfo(float).eps
large = np.finfo(float).max


# load Fortran library symtoplib
symtoplib_path = os.path.join(os.path.dirname(__file__), 'symtoplib')
fsymtop = np.ctypeslib.load_library('symtoplib', symtoplib_path)


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
            frame = "tens_name" will rotate to a principal axes system of a tensor with the name 
                "tens_name", this tensor must be initialized before, see 'tensor' property.
            frame = "zxy" will permute axes x-->z, y-->x, and y-->z.
            frame = "zxy,pas" will rotate to "pas" and permute x-->z, y-->x, and y-->z.
        """
        if isinstance(arg, str):

            rotmat0 = np.eye(3, dtype=np.float64)

            for fr in reversed([v.strip() for v in arg.split(',')]):

                assert (len(fr)>0), f"Illegal frame type specification: '{arg}'"

                if fr.lower()=="pas":
                    # principal axes system
                    try:
                        diag, rotmat = np.linalg.eigh(self.imom())
                    except np.linalg.LinAlgError:
                        raise RuntimeError("Eigenvalues did not converge") from None
                    # append rotation to a total rotation matrix
                    rotmat0 = np.dot(np.transpose(rotmat), rotmat0)
                    # evaluate and store rotational constants in units cm^-1
                    self.frame_diag = diag
                    convert_to_cm = planck * avogno * 1e+16 / (8.0 * np.pi * np.pi * vellgt) 
                    self.ABC = [convert_to_cm/val for val in diag]

                elif "".join(sorted(fr.lower()))=="xyz":
                    # axes permutation
                    ind = [("x","y","z").index(s) for s in list(fr.lower())]
                    rotmat = np.zeros((3,3), dtype=np.float64)
                    for i in range(3):
                        rotmat[i,ind[i]] = 1.0
                    # append rotation to a total rotation matrix
                    rotmat0 = np.dot(rotmat, rotmat0)

                else:
                    # axes system defined by to-diagonal rotation of arbitrary rank-2 (3x3) tensor
                    # the tensor must be initialized before, with the name matching fr
                    try:
                        tens = self.tensor[fr]
                    except KeyError:
                        raise KeyError(f"Tensor '{fr}' was not initialised") from None
                    if tens.ndim!=2:
                        raise ValueError(f"Tensor '{fr}' has inappropriate rank: {tens.ndim} != 2") from None
                    if np.any(np.abs(tens-tens.T)>small*10.0):
                        raise ValueError(f"Tensor '{fr}' is not symmetric") from None
                    try:
                        diag, rotmat = np.linalg.eigh(tens)
                    except np.linalg.LinAlgError:
                        raise RuntimeError("Eigenvalues did not converge") from None
                    # append rotation to a total rotation matrix
                    rotmat0 = np.dot(np.transpose(rotmat), rotmat0)
                    self.frame_diag = diag

        else:
            raise TypeError(f"Unsupported argument type '{type(arg)}' for frame specification, must be 'str'") from None

        # update global frame rotation matrix
        try:
            self.frame_rotation = np.dot(rotmat0, self.frame_rotation)
        except AttributeError:
            self.frame_rotation = rotmat0

        # update a string that keeps track of all frame rotations
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



class SymtopBasis():
    """Basis of symmetric top functions for selected J

    Args:
        J (int): Quantum number of the rotational angular momentum.
        linear (bool): True if molecule is linear, in this case quantum number k is set to zero.
    """

    def __init__(self, J, linear=False):

        self.J = J

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

        # generate Wang-type linear combinations
        for ibas,(J,k,tau) in enumerate(bas):
            coefs, kval = self.wang_coefs(J, k, tau)
            for kk,cc in zip(kval,coefs):
                iprim = np.where((self.jk_table['jk']==(J,kk)).all(axis=1))[0][0]
                self.jk_table['c'][iprim,ibas] = cc


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


    def overlap(self, arg):
        """ Computes overlap matrix elements <self|arg>

        Args:
            arg (SymtopBasis or J): represents set of linear combinations of symmetric-top functions,
                can be either a basis set, i.e., SymtopBas(), or a result of an action of angular
                momentum operator(s) on a basis set, i.e., J() class.
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



def symmetrize(arg, sym="D2"):
    """Returns dictionary of symmetry-adapted objects 'arg' for different irreps (as dict keys)
    of symmetry group defined by 'sym'

    Args:
        arg (SymtopBasis): Basis of symmetric-top functions for selected J.
            ( .... ): ...
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
            ind0 = [ifunc for ifunc in range(nbas) if all(abs(val)<small*1e3 for val in jk_table['c'][:,ifunc]) ]
            nbas_irrep = nbas - len(ind0)
            nbas_sum += nbas_irrep
            res[sym_lab].jk_table = np.zeros(nbas, dtype=[('jk', 'i4', (2)), ('c', np.complex128, [nbas_irrep])])
            res[sym_lab].jk_table['c'] = np.delete(jk_table['c'], ind0, 1)
            res[sym_lab].jk_table['jk'] = jk_table['jk']
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
    """ Basic class for rotational angular momentum operators """
    def __init__(self, arg=None):
        if isinstance(arg, np.ndarray):
            self.jk_table = arg.copy()
        elif arg==None:
            pass
        else:
            try:
                x = arg.jk_table
            except AttributeError:
                raise AttributeError(f"{arg.__class__.__name__} has no attribute 'jk_table'") from None
            else:
               self.jk_table = arg.jk_table.copy()


    def __add__(self, arg):
        try:
            x = self.jk_table
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'jk_table'") from None
        if isinstance(arg, J):
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
            res = J(jk_table)
        else:
            raise TypeError(f"unsupported operand type(s) for '+': {self.__class__.__name__} and " \
                    +f"{arg.__class__.__name__}") from None
        return res


    def __sub__(self, arg):
        try:
            x = self.jk_table
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'jk_table'") from None
        if isinstance(arg, J):
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
            res = J(jk_table)
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
            res = self.__class__(arg)
        elif isinstance(arg, scalar):
            try:
                x = self.jk_table
            except AttributeError:
                raise AttributeError(f"{self.__class__.__name__} has no attribute 'jk_table'") from None
            jk_table = self.jk_table.copy()
            jk_table['c'] *= arg
            res = J(jk_table)
        else:
            raise TypeError(f"unsupported operand type(s) for '*': {self.__class__.__name__} and " \
                    +f"{arg.__class__.__name__}") from None
        return res


    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__



class Jp(J):
    """ J+ = Jx + iJy """
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



class Jm(J):
    """ J- = Jx - iJy """
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



class Jz(J):
    """ Jz """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        if self.__dict__.get("jk_table") is None:
            pass
        else:
            for ielem,(j,k) in enumerate(self.jk_table['jk']):
                self.jk_table['c'][ielem,:] = self.jk_table['c'][ielem,:] * k



class JJ(J):
    """ J^2 """
    def __init__(self, arg=None):
        J.__init__(self, arg)
        if self.__dict__.get("jk_table") is None:
            pass
        else:
            for ielem,(j,k) in enumerate(self.jk_table['jk']):
                self.jk_table['c'][ielem,:] = self.jk_table['c'][ielem,:] * j*(j+1)



class Jxx(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = 0.25 * ( Jm(arg) * Jm(arg) +  Jm(arg) * Jp(arg) \
                +  Jp(arg) * Jm(arg) +  Jp(arg) * Jp(arg) )
            J.__init__(self, res)



class Jxy(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.25) * ( Jm(arg) * Jm(arg) -  Jm(arg) * Jp(arg) \
                +  Jp(arg) * Jm(arg) -  Jp(arg) * Jp(arg) )
            J.__init__(self, res)



class Jyx(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.25) * ( Jm(arg) * Jm(arg) +  Jm(arg) * Jp(arg) \
                -  Jp(arg) * Jm(arg) -  Jp(arg) * Jp(arg) )
            J.__init__(self, res)



class Jxz(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = 0.5 * ( Jm(arg) * Jz(arg) +  Jp(arg) * Jz(arg) )
            J.__init__(self, res)



class Jzx(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = 0.5 * ( Jz(arg) * Jm(arg) +  Jz(arg) * Jp(arg) )
            J.__init__(self, res)



class Jyy(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = -0.25 * ( Jm(arg) * Jm(arg) -  Jm(arg) * Jp(arg) \
                -  Jp(arg) * Jm(arg) +  Jp(arg) * Jp(arg) )
            J.__init__(self, res)



class Jyz(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.5) * ( Jm(arg) * Jz(arg) -  Jp(arg) * Jz(arg) )
            J.__init__(self, res)



class Jzy(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.5) * ( Jz(arg) * Jm(arg) -  Jz(arg) * Jp(arg) )
            J.__init__(self, res)



class Jzz(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = Jz(arg) * Jz(arg)
            J.__init__(self, res)



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
    t = np.random.rand(3,3)
    t = (t+t.T) * 0.5
    camphor.tensor = ("aa", t)
    print(camphor.tensor)
    print(camphor.imom())
    camphor.frame = "aa"
    print(camphor.tensor)
    sys.exit()
    #a,b = camphor.frame
    #print(a,b)
    #print(camphor.XYZ)
    print("\n")
    print(camphor.imom())
    print(camphor.ABC)
    bas = symmetrize(SymtopBasis(10), sym="D2")
    print(bas.keys())
    A,B,C = camphor.ABC
    for sym,sym_bas in bas.items():
        Jx2 = Jxx(sym_bas)
        Jy2 = Jyy(sym_bas)
        Jz2 = Jzz(sym_bas)
        h = B * Jx2 + A * Jy2 + C * Jz2 
        mat = sym_bas.overlap(h)
        enr, vec = np.linalg.eigh(mat)
        print(sym, enr)

    # A,B,C = camphor.ABC
    # bas = SymtopBasis(10)
    # Jx2 = Jxx(bas)
    # Jy2 = Jyy(bas)
    # Jz2 = Jzz(bas)
    # h = B * Jx2 + C * Jy2 + A * Jz2 
    # mat = bas.overlap(h)
    # enr, vec = np.linalg.eigh(mat)
    # print(enr)
