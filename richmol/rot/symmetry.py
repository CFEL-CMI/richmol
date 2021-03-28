import numpy as np
from richmol.wigner import symtop
import copy


_symmetries = dict()

def register_sym(func):
    _symmetries[func.__name__] = func
    return func


def group(sym, *args, **kwargs):
    """Returns symmetry class specified by 'sym'"""
    if sym in _symmetries:
        return _symmetries[sym]
    else:
        raise TypeError(f"symmetry '{sym}' is not available") from None


def symmetrize(arg, sym="D2", thresh=1e-12):
    """Generates symmetry-adapted set of wave functions in symmetric-top basis

    Args:
        arg : PsiTableMK
            Wave functions in symmetric-top basis.
        sym : str
            Symmetry group.
        thresh : float
            Threshold for treating symmetrization and wave function superposition
            coefficients as zero.

    Returns:
        res : dict
            Dictionary of symmetry-adapted wave functions for different symmetries,
            i.e., res[sym] -> PsiTableMK
    """
    try:
        x = arg.k
    except AttributeError:
        raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
    try:
        x = arg.m
    except AttributeError:
        raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None

    # list of J quanta spanned by arg
    Jlist = list(set(j for (j,k) in arg.k.table['prim']))

    # create copies of arg for different irreps
    symmetry = sym(Jlist[0])
    res = {sym_lab : copy.deepcopy(arg) for sym_lab in symmetry.sym_lab}
    for elem in res.values():
        elem.k.table['c'] = 0

    nstat = arg.k.table['c'].shape[1]
    prim = [tuple(x) for x in arg.k.table['prim']]

    for J in Jlist:
        symmetry = sym(J)
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
                    raise ValueError(f"input set of function is missing " + \
                        f"some primitive symmetric-top functions that are necessary " + \
                        f"for symmetrization, for example, (J,k) = {(J,k)} is missing") from None

        for irrep,sym_lab in enumerate(symmetry.sym_lab):
            pmat = np.dot(proj[irrep,:,ind_k], arg.k.table['c'][ind_p,:])
            res[sym_lab].k.table['c'][ind_p,:] = pmat[ind_k,:]

    # remove states with zero coefficients
    remove = []
    for sym_lab,elem in res.items():
        elem.k = elem.k.del_zero_stat(thresh=thresh)
        if elem.k is None:
            remove.append(sym_lab)
    for sym_lab in remove: del res[sym_lab]

    # check if the total number of states remains the same
    nstat_sym = sum(elem.k.table['c'].shape[1] for elem in res.values()) 
    if nstat_sym != nstat:
        raise RuntimeError(f"number of states before symmetrization = {nstat} " + \
            f"is different from the total number of states across all irreps = {nstat_sym}")

    return res



class SymtopSymmetry():
    """Generates symmetry-adapting projection operators for symmetric-top functions"""

    def __init__(self, J):

        try:
            self.J = int(round(J))
        except TypeError:
            raise TypeError(f"J = '{J}' is not a number") from None
        assert (self.J>=0), f"J = {J} < 0"

        # compute symmetrization coefficients for symmetric-top functions
        jmin = J
        jmax = J
        npoints = self.noper
        self.coefs = symtop.threed_grid(J, J, self.euler_rotation, int(True), npoints)


    def proj(self):
        """Builds projection operators"""
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


@register_sym
class C1(SymtopSymmetry):
    def __init__(self, J):

        self.noper = 1
        self.nirrep = 1
        self.ndeg = [1]

        self.characters = np.zeros((self.nirrep,self.noper), dtype=np.float64)
        self.euler_rotation = np.zeros((3,self.noper), dtype=np.float64)

        self.characters[0,0] = 1 # A

        self.sym_lab = ['A']

        pi = np.pi
        self.euler_rotation[:,0] = [0,0,0] # E

        SymtopSymmetry.__init__(self, J)


@register_sym
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

        self.sym_lab = ['A','B1','B2','B3']

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


@register_sym
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

        self.sym_lab = ['Ag','Au','B1g','B1u','B2g','B2u','B3g','B3u']

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


@register_sym
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

        self.sym_lab = ['A1','A2','B1','B2']

        pi = np.pi
        # order of angles in euler_rotation[0:3,:] is [phi, theta, chi]
        self.euler_rotation[:,0] = [0,0,0]             # E
        self.euler_rotation[:,1] = [pi,0,0]            # C2(z)
        self.euler_rotation[:,2] = [0,pi,0]            # C2(y)
        self.euler_rotation[:,3] = [0.5*pi,pi,1.5*pi]  # C2(x)

        SymtopSymmetry.__init__(self, J)

