from richmol.rot import mol_frames
from richmol.rot import mol_tens
from richmol.rot import atomdata
from richmol.rot import constants as const
import numpy as np
import string
import random


_diag_tol = 1e-12 # for treating off-diagonal elements as zero
_sing_tol = 1e-12 # for treating singular value as zero
_xyz_tol = 1e-12 # for treating Cartesian coordinate as zero
_abc_tol = 0.1 # maximal allowed difference (in cm^-1) between the input (experimental)
               # and calculated (from molecular geometry) rotational constants

_USER_TENSORS = dict()

class Molecule:

    @property
    def XYZ(self):
        """Masses and Cartesian coordinates of atoms"""
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


    def store_xyz(self, file_name: str, comment: str = ""):
        """Stores Cartesian coordinates of atoms into XYZ file"""
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
            mol_tens._TENSORS[name] = mol_tens._TENSORS.pop(random_name)
            # call dynamic rotation of tensor
            mol_tens.dynamic_rotation(Molecule, name, value.val)
            #
            # check if attribute with name 'name' has been already assigned (registered)
            temp = dict([(val[0],key) for key,val in _USER_TENSORS.items()])
            if name in temp: # if yes, delete the entry
                existing_name = temp[name]
                del _USER_TENSORS[existing_name]
            # register tensor under new random name
            _USER_TENSORS[value.name] = [name, lambda: mol_tens._TENSORS[name][2]]
        elif hasattr(value, "__name__") and value.__name__ == "user_del_tens":
            del mol_tens._TENSORS[name]
            # object.__setattr__(self, name, None)
        elif name in mol_tens._TENSORS: # call dynamic rotation of tensor using predefined types in mol_tens module
            mol_tens.dynamic_rotation(Molecule, name, value)
        else:
            object.__setattr__(self, name, value)


    @property
    def frame(self):
        """Defines molecule-fixed frame (embedding)"""
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
                rotmat = mol_frames.rotmat(fr, self)
                self.frame_rotation = np.dot(rotmat, self.frame_rotation)
        else:
            raise TypeError(f"bad argument type '{type(arg)}' for frame specification") from None


    def imom(self):
        """Returns inertia tensor"""
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
        """Computes rotational kinetic energy matrix"""
        convert_to_cm = const.planck * const.avogno * 1e+16 / (4.0 * np.pi * np.pi * const.vellgt)
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
        xyz = self.XYZ['xyz']
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
        """Returns asymmtery parameter kappa = (2*B-A-C)/(A-C)"""
        A, B, C = self.ABC
        return (2*B-A-C)/(A-C)


    @kappa.setter
    def kappa(self, val):
        raise AttributeError(f"setting kappa is not permitted") from None


    def ABC_calc(self):
        """Returns rotational constants in units of cm^-1 calculated from inertia tensor"""
        itens = self.inertia
        if np.any(np.abs( np.diag(np.diag(itens)) - itens) > _diag_tol):
            raise Exception(f"failed to compute rotational constants since inertia tensor is not diagonal, " + \
                f"max offdiag = {np.max(np.abs(np.diag(np.diag(itens))-itens)).round(16)}") from None
        convert_to_cm = const.planck * const.avogno * 1e+16 / (8.0 * np.pi * np.pi * const.vellgt) 
        abc = [convert_to_cm/val for val in np.diag(itens)]
        return [val for val in reversed(sorted(abc))]


    @property
    def ABC(self):
        """Returns experimental rotational constants if available, otherwise - calculated once"""
        try:
            return self.ABC_exp
        except AttributeError:
            return self.ABC_calc()


    @ABC.setter
    def ABC(self, val):
        """Set experimental rotational constants, in units of cm^-1"""
        try:
            A, B, C = val
        except valueError:
            raise ValueError(f"please specify rotational constants in a tuple (A, B, C)") from None
        self.ABC_exp = [val for val in reversed(sorted([A, B, C]))]
        # change coordinate system to inertia frame
        try:
            # nullify all previous frame rotations and add inertia frame
            self.frame = "ipas,null"
            # check for good match between calculated and experimental constants
            if any(abs(e - c) > _abc_tol for e,c in zip(self.ABC_exp, self.ABC_calc())):
                err = "\n".join( abc + " %12.6f"%e + " %12.6f"%c + " %12.6f"%(e-c) \
                                 for abc,e,c in zip(("A","B","C"), self.ABC_exp, self.ABC_calc()) )
                raise ValueError(f"input experimental rotational constants differ much from the calculated once\n" + \
                    f"        exp          calc       exp-calc\n" + err) from None
        except TypeError: # 'ipas' is not available
            # this means that .XYZ is not defined, so we assume that all input molecular tensors
            # are in the inertia frame, the frame rotation matrix is identity matrix
            print("remark: assume all input molecular tensors defined with respect to the inertia frame")
        # block .frame property
        self.block_frame = [True, "user-defined rotational constants"]


    @property
    def B(self):
        try:
            return self.B_exp
        except AttributeError:
            return self.ABC_calc()[1]


    @B.setter
    def B(self, val):
        self.B_exp = val
        # change coordinate system to inertia frame
        try:
            # nullify all previous frame rotations and add inertia frame
            self.frame = "ipas,null"
            # check for good match between calculated and experimental constants
            if abs(self.B_exp - self.ABC_calc()[1]) > _abc_tol:
                raise ValueError(f"input experimental rotational constant B = {self.B_exp} differ much " + \
                    f"from the calculated B = {self.ABC_calc()[1]}") from None
        except TypeError: # 'ipas' is not available
            # this means that .XYZ is not defined, so we assume that all input molecular tensors
            # are in the inertia frame, the frame rotation matrix is identity matrix
            print("remark: assume all input molecular tensors defined with respect to the inertia frame")
        # block .frame property
        self.block_frame = [True, "user-defined rotational constants"]


def mol_tensor(val):
    """Adds user-defined new Cartesian tensor which is not declared in mol_tens module.
    This will be also dynamically rotated to molecule-fixed frame
    """
    try:
        tens = np.array(val)
        ndim = tens.ndim
    except AttributeError:
        raise AttributeError(f"input argument is not a tensor") from None
    random_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    def general_rotate(obj):
        "Rotates tensor of arbitrary rank<=8"""
        # the value of tensor is kept in mol_tens._TENSORS dictionary
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



if __name__ == '__main__':
    import sys

    ocs = Molecule()
    ocs.XYZ = ("O", 0,1,0, "C", 0,-1,0, "S", 0,0,0)
    print(ocs.ABC)
    ocs.B = 0.7
    print(ocs.B)
    print(ocs.kappa)

    camphor = Molecule()
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
    camphor.store_xyz("camphor.xyz", "some random comment")
    sys.exit()

    camphor.dip = [1.21615, -0.30746, 0.01140]
    camphor.pol = [[115.80434, -0.58739, 0.03276], \
                   [-0.58739, 112.28245, 1.36146], \
                   [0.03276, 1.36146, 108.47809]]

    camphor.vec = cart_tens([1.21615, -0.30746, 0.01140])
    camphor.mat = cart_tens([[115.80434, -0.58739, 0.03276], \
                   [-0.58739, 112.28245, 1.36146], \
                   [0.03276, 1.36146, 108.47809]])

    print(camphor.dip)
    print(camphor.vec)
    print(camphor.pol)
    print(camphor.mat)

    camphor.frame = "diag(inertia)"

    print(camphor.dip)
    print(camphor.vec)
    print(camphor.pol)
    print(camphor.mat)
    print(camphor.vec)
    # # print(camphor.XYZ['xyz'])
    # # print(camphor.itens)
    # # print(camphor.XYZ['xyz'])
    # print(camphor.itens)
    # camphor.frame = "pas"
    # print(camphor.frame)
    # print(camphor.itens)
    # print(camphor.B)
    print(camphor.kappa)

