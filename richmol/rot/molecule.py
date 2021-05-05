from richmol.rot import mol_frames
from richmol.rot import mol_tens
from richmol.rot import atomdata
from scipy import constants
from richmol.rot import symmetry
import numpy as np
import string
import random
import inspect
import h5py
import datetime
import time
from richmol import json


_diag_tol = 1e-12 # for treating off-diagonal elements as zero
_sing_tol = 1e-12 # for treating singular value as zero
_xyz_tol = 1e-12 # for treating Cartesian coordinate as zero
_abc_tol_perc = 5 # maximal allowed difference (in percents in units of cm^-1) between the input
                  # (experimental) and calculated (from molecular geometry) rotational constants

_USER_TENSORS = dict()

class Molecule:
    """Describes rigid molecule
    """

    @property
    def XYZ(self):
        """Cartesian coordinates and masses of atoms

        Args:
            arg
                Atomic labels, Cartesian coordinates, and units in a tuple, for example,

                .. code-block:: python

                    water.XYZ = ("bohr",
                                 "O",  0.00000000,   0.00000000,   0.12395915,
                                 "H",  0.00000000,  -1.43102686,  -0.98366080,
                                 "H",  0.00000000,   1.43102686,  -0.98366080)

                To define a non-standard isotoplogue, add the corresponding number next
                to the atom label, e.g., "O18", "H2".
                Cartesian coordinates can also be read from an XYZ file, in this case `arg` needs
                to be a string with the name of the XYZ file.

        Returns:
            structured numpy array, len = <number of atoms>
                XYZ['xyz'] - Cartesian coordinates of atoms in Angstrom,
                XYZ['mass'] - atomic masses,
                XYZ['label'] - atomic labels
        """
        try:
            x = self.atoms
        except AttributeError:
            raise AttributeError(f"attribute 'XYZ' is not available") from None
        # rotate Cartesian coordinates to molecule-fixed frame
        res = self.atoms.copy()
        res['xyz'] = np.dot(res['xyz'], np.transpose(self.frame))
        return res


    @XYZ.setter
    def XYZ(self, arg):
        self.atoms = atomdata.read(arg)
        # register inertia tensor (inertia) and its eigenvector matrix (ipas)
        # we don't need to assign any values, they will be computed on the fly using self.imom()
        self.inertia = np.eye(3) # can be used for .frame="diag(inertia)"
        self.ipas = np.eye(3)    # can be used for .frame="ipas"


    def store_xyz(self, file_name, comment=""):
        """Stores Cartesian coordinates of atoms into XYZ file

        Args:
            file_name : str
                Name of XYZ file
            comment : str
                Optional comment line
        """
        xyz = self.XYZ['xyz']
        mass = self.XYZ['mass']
        lab = self.XYZ['label']
        with open(file_name, "w") as fl:
            fl.write(str(len(mass)) + "\n")
            fl.write(comment + "\n")
            for iatom in range(len(mass)):
                fl.write("%4s"%lab[iatom]+"      "+"  ".join("%16.12f"%x for x in xyz[iatom,:])+"\n")


    def __setattr__(self, name, value):
        # enable dynamic rotation to molecule-fixed frame if attribute is a Cartesian tensor
        if hasattr(value, "__name__") and value.__name__ == "user_mol_tens": # user-defined new Cartesian tensor, see mol_tensor function
            # replace random registration name with the name of the attribute
            random_name = value.name
            mol_tens._tensors[name] = mol_tens._tensors.pop(random_name)
            # call dynamic rotation of tensor
            mol_tens.dynamic_rotation(Molecule, name, value.val)
            #
            # check if attribute with name 'name' has been already assigned (registered)
            temp = dict([(val[0],key) for key,val in _USER_TENSORS.items()])
            if name in temp: # if yes, delete the entry
                existing_name = temp[name]
                del _USER_TENSORS[existing_name]
            # register tensor under new random name
            _USER_TENSORS[value.name] = [name, lambda: mol_tens._tensors[name][2]]
        elif hasattr(value, "__name__") and value.__name__ == "user_del_tens":
            del mol_tens._tensors[name]
            # object.__setattr__(self, name, None)
        elif name in mol_tens._tensors: # call dynamic rotation of tensor using predefined types in mol_tens module
            mol_tens.dynamic_rotation(Molecule, name, value)
        else:
            object.__setattr__(self, name, value)


    @property
    def frame(self):
        """Molecular frame embedding

        Args:
            arg
                String containing the name of the axes system (e.g., "ipas" for the
                inertial principal axes system), or the name of an attribute to be used as a rotation
                matrix, which can be combined with various linear operations (e.g., "diag" for diagonalization).

                Examples:

                .. code-block:: python

                    water.frame = "diag(inertia)" # rotates to a frame where the inertia tensor
                                                  # becomes diagonal
                    water.frame = "zxy" # permutes the x, y, and z axes
                    water.pol = [[9.1369, 0, 0], [0, 9.8701, 0], [0, 0, 9.4486]]
                    water.frame = "pol" # rotates frame with water.pol matrix
                    water.frame = "diag(pol)" # rotates to a frame where water.pol tensor
                                              # becomes diagonal
                    water.frame = "None" # or None, resets frame to the one defined
                                         # by the input  molecular geometry

        Returns:
            array (3,3)
                Frame rotation matrix
        """
        try:
            rotmat = self.frame_rotation
        except AttributeError:
            rotmat = mol_frames.rotmat('eye')
        return rotmat


    @frame.setter
    def frame(self, arg):
        if isinstance(arg, str):
            try:
                x = self.frame_rotation
            except AttributeError:
                self.frame_rotation = mol_frames.rotmat('eye')

            for fr in reversed([v.strip().lower() for v in arg.split(',')]):
                # if frame='None', reset frame to the one defined by the input molecular geometry
                if fr == "none":
                    fr = "null"
                rotmat = mol_frames.rotmat(fr, self)
                self.frame_rotation = np.dot(rotmat, self.frame_rotation)
        elif isinstance(arg, type(None)):
            # if frame=None, reset frame to the one defined by the input molecular geometry
            self.frame_rotation = mol_frames.rotmat("null", self)
        else:
            raise TypeError(f"bad argument type '{type(arg)}' for frame specification") from None


    def imom(self):
        """Computes inertia tensor"""
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
        """Computes rotational kinetic energy matrix (in units of :math:`cm^{-1}`)"""
        convert_to_cm = constants.value('Planck constant') * constants.value('Avogadro constant') \
                      * 1e+16 / (4.0 * np.pi**2 * constants.value('speed of light in vacuum')) * 1e5
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
            if sv[i] > _sing_tol:
                dmat[i,i] = 1.0/sv[i]
            else:
                no_sing += 1
                print(f"warning: rotational kinetic energy matrix is singular")
                if no_sing==1 and self.linear() is True:
                    print(f"this is fine for linear molecule: set singular element 1/{sv[i]}=0")
                    dmat[i,i] = 0
                else:
                    ifstop = True
        if ifstop is True:
            raise Exception(f"rotational kinetic energy matrix is singular, " + \
                f"please check your input geometry")

        gbig = np.dot(umat, np.dot(dmat, vmat))
        gbig *= convert_to_cm
        return gbig


    def linear(self):
        """Returns True/False if molecule is linear/non-linear"""
        try:
            xyz = self.XYZ['xyz']
        except AttributeError:
            if hasattr(self, 'B_exp') and not hasattr(self, 'ABC_exp'):
                return True
            elif hasattr(self, 'ABC_exp') and not hasattr(self, 'B_exp'):
                return False
            else:
                raise ValueError(f"cannot determine whether the molecule is linear " + \
                    f"on the basis of input molecular parameters") from None
        imom = self.imom()
        d, rotmat = np.linalg.eigh(imom)
        xyz2 = np.dot(xyz, rotmat)
        tol = _xyz_tol
        if (np.all(abs(xyz2[:,0])<tol) and np.all(abs(xyz2[:,1])<tol)) or \
            (np.all(abs(xyz2[:,0])<tol) and np.all(abs(xyz2[:,2])<tol)) or \
            (np.all(abs(xyz2[:,1])<tol) and np.all(abs(xyz2[:,2])<tol)):
            return True
        else:
            return False


    @property
    def kappa(self):
        """Returns rotational asymmetry parameter kappa = (2*B-A-C)/(A-C)"""
        A, B, C = self.ABC
        return (2*B-A-C)/(A-C)


    @kappa.setter
    def kappa(self, val):
        raise AttributeError(f"setting kappa is not permitted") from None


    @property
    def ABC_geom(self):
        """Returns A, B, and C rotational constants (in units of :math:`cm^{-1}`),
        calculated from molecular geometry input
        """
        itens = self.inertia
        if np.any(np.abs( np.diag(np.diag(itens)) - itens) > _diag_tol):
            raise ValueError(f"failed to compute rotational constants since inertia tensor is not diagonal, " + \
                f"max offdiag = {np.max(np.abs(np.diag(np.diag(itens))-itens)).round(16)}") from None
        convert_to_cm = constants.value('Planck constant') * constants.value('Avogadro constant') \
                      * 1e+16 / (8.0 * np.pi**2 * constants.value('speed of light in vacuum')) * 1e5
        abc = [convert_to_cm/val for val in np.diag(itens)]
        return [val for val in abc]


    @ABC_geom.setter
    def ABC_geom(self, val):
        raise AttributeError(f"setting calculated rotational constants is not permitted") from None


    @property
    def ABC(self):
        """Molecular rotational constants

        Args:
            arg
                Tuple with A, B, and C user-defined rotational constants (in :math:`cm^{-1}`).

        Returns:
            Tuple with A, B, and C user-defined rotational constants,
            if they have been initialized, otherwise returns the constants computed
            from the input molecular geometry.
        """
        try:
            self.check_ABC()
            return self.ABC_exp
        except AttributeError:
            return self.ABC_geom


    @ABC.setter
    def ABC(self, val):
        try:
            A, B, C = val
        except TypeError:
            raise TypeError(f"please specify rotational constants in a tuple (A, B, C)") from None
        self.ABC_exp = [val for val in reversed(sorted([A, B, C]))]
        self.check_ABC()


    @property
    def B_geom(self):
        """Returns B rotational constant (in units of :math:`cm^{-1}`), calculated from molecular geometry input"""
        if self.linear == False:
            raise ValueError(f"molecule is not linear, use ABC to compute rotational constants")
        with np.errstate(divide='ignore'): # ignore divide by zero warning
            b = self.ABC_geom[1]
        return b


    @B_geom.setter
    def B_geom(self, val):
        raise AttributeError(f"setting calculated rotational constants is not permitted") from None


    @property
    def B(self):
        """Molecular rotational constant B

        Args:
            arg
                User-defined rotational constant (in :math:`cm^{-1}`).

        Returns:
            User-defined constant, if it has been initialized, otherwise returns
            constant computed from the input molecular geometry.
        """
        try:
            self.check_B()
            return self.B_exp
        except AttributeError:
            return self.B_geom


    @B.setter
    def B(self, val):
        self.B_exp = val
        self.check_B()


    def check_ABC(self):
        try:
            ABC_geom = self.ABC_geom
        except AttributeError:
            return
        try:
            ABC_exp = self.ABC_exp
        except AttributeError:
            return
        if any(abs(e - c) > _abc_tol_perc/100.0*e for e,c in zip(ABC_exp, ABC_geom)):
            err = "\n".join( abc + " %12.6f"%e + " %12.6f"%c + " %12.6f"%(e-c) \
                             for abc,e,c in zip(("A","B","C"), ABC_exp, ABC_geom) )
            raise ValueError(f"input rotational constants disagree much with geometry\n" + \
                f"        exp          calc       exp-calc\n" + err) from None
        return


    def check_B(self):
        try:
            B_geom = self.B_geom
        except AttributeError:
            return
        try:
            B_exp = self.B_exp
        except AttributeError:
            return
        if any(abs(e - c) > _abc_tol_perc/100.0*e for e,c in zip(B_exp, B_geom)):
            err = "\n".join( abc + " %12.6f"%e + " %12.6f"%c + " %12.6f"%(e-c) \
                             for abc,e,c in zip(("B"), B_exp, B_geom) )
            raise ValueError(f"input rotational constants disagree much with geometry\n" + \
                f"        exp          calc       exp-calc\n" + err) from None
        return


    @property
    def abc(self):
        try:
            x = self.ABC_exp # works only if user-input (experimental) rotational constant were defined
            if self.kappa > 0:
                return 'xyz' # "IIIr representation for near oblate-top"
            elif self.kappa <= 0:
                return 'zyx' # "Ir representation for near prolate-top"
        except AttributeError:
            return 'xyz'

    @abc.setter
    def abc(self, val):
        raise AttributeError(f"setting abc->xyz mapping is not permitted") from None


    @property
    def sym(self):
        """Molecular point symmetry group

        Args:
            arg : str
                Molecular symmetry label, e.g., "C1", "D2", "C2v"

        Returns:
            :py:class:`richmol.rot.symmetry.SymtopSymmetry` class.
        """
        try:
            return self.symmetry
        except AttributeError:
            return symmetry.group("C1")


    @sym.setter
    def sym(self, val="C1"):
        # automatic symmetry works only for inertia principal axes system
        try:
            x = self.ABC
        except ValueError:
            print(f"warning: change molecular frame to inertia axes system to enable symmetry")
            val = "C1"
        except AttributeError:
            raise AttributeError(f"can't use symmetry if neither geometry nor rotational constants provided") from None
        self.symmetry = symmetry.group(val)


    def store(self, filename, name=None, comment=None, replace=False):
        """Stores object in HDF5 file
    
        Args:
            filename : str
                Name of HDF5 file
            name : str
                Name of the data group, by default the name of the variable is used
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
                doc= ""

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
                Name of the data group, if None, the first group with the matching "__class_name__" 
                attribute will be loaded
        """
        class_name = self.__module__ + "." + self.__class__.__name__

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
        """Generates '__class_name__' attribute for the tensor data group in HDF5 file"""
        return self.__module__ + "." + self.__class__.__name__



def mol_tensor(val):
    """Declares a new user-defined Cartesian tensor which, when assigned to an attribute
    of a Molecule object, will be dynamically rotated to a chosen molecular frame :py:func:`Molecule.frame`
    whenever the latter is changed

    Args:
        val : array
            Cartesian tensor

    Returns:
        object, needs to be assigned to an attribute of Molecule class
    """
    try:
        tens = np.array(val)
        ndim = tens.ndim
    except AttributeError:
        raise AttributeError(f"input argument is not a tensor") from None
    random_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    def general_rotate(obj):
        "Rotates tensor of arbitrary rank<=8"""
        # the value of tensor is kept in mol_tens._tensorsdictionary
        # it is accessed here by calling a function in _USER_TENSORS[
        x = _USER_TENSORS[random_name][1]()
        tens_name = _USER_TENSORS[random_name][0]
        sa = "abcdefgh"
        si = "ijklmnop"
        if x.ndim > len(sa):
            raise ValueError(f"tensor rank '{x.ndim}' exceeds the maximum {len(sa)} for tensor '{tens_name}'") from None
        key = "".join(sa[i]+si[i]+"," for i in range(x.ndim)) \
            + "".join(si[i] for i in range(x.ndim)) + "->" \
            + "".join(sa[i] for i in range(x.ndim))
        rot_mat = [obj.frame for i in range(x.ndim)]
        return np.einsum(key, *rot_mat, x)
    # register rotation function general_rotate in mol_tens module under random name
    general_rotate.__name__ = random_name
    setattr(mol_tens, random_name, mol_tens.register_tensor(ndim)(general_rotate))
    # return new type containing tensor values and random name of rotation function
    return type("user_mol_tens", (object,), {"val" : tens, "name" : random_name})


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

