import numpy as np
import warnings
import math
import copy
from richmol import json

_small = abs(np.finfo(float).eps)*10
_large = abs(np.finfo(float).max)

_assign_nprim = 10  # max number of primitive functions printed in the state assignment
_assign_ndig_c2 = 6 # number of significant digits printed for the assignment coefficient |c|^2

# allow for repetitions of warning for the same source location
# warnings.simplefilter('always', UserWarning)


class PsiTable():
    """Basic class to handle operations on rotational wave functions.

    The wave functions are represented by a table of superposition
    coefficients, where different rows correspond to different primitive basis
    functions (symmetric-top functions), and different columns correspond
    to different eigenstates.

    Args:
        prim : list
            List of primitive quanta, can be list of tuples of many quanta.
        stat : list
            List of eigenstate quanta (assignments), can be list of tuples of many quanta.
        coefs : numpy ndarray (len(prim), len(stat))
            Wave function superposition coefficients.

    Attributes:
        table : structured numpy array
            table['c'][:,:] is a numpy.complex128 matrix containing wave function
                superposition coefficients, with its rows corresponding to different
                primitive basis functions (symmetric-top functions)
                and columns - to different eigenstates.
            table['prim'][:] is an array of tuples of integer quantum numbers
                (q1, q2, ...) labelling different primitive basis functions.
            table['stat'][:] is an array of tuples of U10 quantum numbers
                (s1, s2, ...) labelling different eigenstates.
        enr : numpy array, enr.dtype = float, enr.shape = table['c'].shape[1]
            Energies of eigenstates. Added dynamically at a stage of wave
            function unitary transformation using PsiTable.rotate.
        sym : numpy array, sym.dtype = str, sym.shape = table['c'].shape[1]
            Molecular symmetry labels of eigenstates.
    """

    def __init__(self, prim, stat, coefs=None):

        if not isinstance(prim, (list, tuple, np.ndarray)):
            raise TypeError(f"bad argument type '{type(prim)}' for 'prim'") from None
        if not isinstance(stat, (list, tuple, np.ndarray)):
            raise TypeError(f"bad argument type '{type(stat)}' for 'stat'") from None
        try:
            x = [int(val) for elem in prim for val in elem]
        except ValueError:
            raise ValueError(f"failed to convert elements in 'prim' into integers") from None

        nprim = len(prim)
        nstat = len(stat)
        assert (nprim>0), f"len(prim) = 0"
        assert (nstat>0), f"len(stat) = 0"
        assert (nprim>=nstat), f"len(prim) < len(stat): {nprim} < {nstat}"

        nelem_stat = list(set(len(elem) for elem in stat))
        if len(nelem_stat)>1:
            raise ValueError(f"inconsistent lengths across elements of 'stat'") from None
        nelem_prim = list(set(len(elem) for elem in prim))
        if len(nelem_prim)>1:
            raise ValueError(f"inconsistent lengths across elements of 'prim'") from None

        # check for duplicates in prim
        if len(list(set(tuple(p) for p in prim))) != len(prim):
            raise ValueError(f"duplicate elements in 'prim'")

        # check for duplicates in stat
        # if len(list(set(tuple(s) for s in stat))) != len(stat):
        #     raise ValueError(f"Duplicate elements in 'stat'")

        dt = [('prim', 'i4', (nelem_prim)), ('stat', 'U10', (nelem_stat)), ('c', np.complex128, [nstat])]
        self.table = np.zeros(nprim, dtype=dt)
        self.table['prim'] = prim
        self.table['stat'][:nstat] = stat

        if coefs is not None:
            try:
                shape = coefs.shape
            except AttributeError:
                raise TypeError(f"bad argument type '{type(coefs)}' for 'coefs'") from None
            if any(x!=y for x,y in zip(shape,[nprim,nstat])):
                raise ValueError(f"shapes of 'coefs' = {shape}, 'prim' = {nprim} " + \
                    f"and 'stat' = {nstat} are not aligned: {shape} != ({nprim}, {nstat})") from None
            self.table['c'][:,:] = coefs


    @classmethod
    def fromPsiTable(cls, arg):
        """Initializes PsiTable from an argument of PsiTable type using deepcopy"""
        if not isinstance(arg, PsiTable):
            raise TypeError(f"bad argument type '{type(arg)}'") from None
        cls = copy.deepcopy(arg)
        return cls


    def __add__(self, arg):
        try:
            x = arg.table
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'table'") from None

        if not np.array_equal(self.table['prim'], arg.table['prim']):
            raise ValueError(f"'{type(self)}' objects under sum have different sets of primitive quanta " + \
                f"(table['prim'] attributes do not match)") from None

        if not np.array_equal(self.table['stat'], arg.table['stat']):
            raise ValueError(f"'{type(self)}' objects under sum have different sets of state quanta " + \
                f"(table['stat'] attributes do not match)") from None

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
            raise ValueError(f"'{type(self)}' objects under sub have different sets of primitive quanta " + \
                f"(table['prim'] attributes do not match)") from None

        if not np.array_equal(self.table['stat'], arg.table['stat']):
            raise ValueError(f"'{type(self)}' objects under sub have different sets of state quanta " + \
                f"(table['stat'] attributes do not match)") from None

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
            raise TypeError(f"unsupported operand type(s) for '*': '{self.__class__.__name__}' and " + \
                f"'{arg.__class__.__name__}'") from None
        return PsiTable(prim, stat, coefs)


    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__


    def append(self, arg, check_duplicate_stat=False, del_duplicate_stat=False, del_zero_stat=False, \
               del_zero_prim=False, thresh=1e-12):
        """Appends two wave function sets together: self + arg.

        If requested:
            'check_duplicate_stat' = True: checks for duplicate states after append.
            'del_duplicate_stat' = True: deletes duplicate states.
            'del_zero_stat' = True: deletes states with all coefficients below 'thresh'.
            'del_zero_prim' = True: deletes primitive functions that have negligible
                contribution (below 'thresh') to all states.
        """
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

        if len(stat1[0]) != len(stat2[0]) or len(prim1[0]) != len(prim2[0]):
            raise ValueError(f"shapes of wave function sets are not aligned: " +\
                +f"{(len(prim1[0]), len(stat1[0]))} != {(len(prim2[0]), len(stat2[0]))}") from None

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
        if check_duplicate_stat == True:
            if len(list(set(tuple(s) for s in stat))) != len(stat):
                raise ValueError(f"found duplicate eigenstates after adding two wave function sets") from None

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
        """Applies unitary transformation.

        Args:
            arg : numpy 2D array or tuple (numpy 2D array, 1D array)
                Unitary transformation matrix (2D array) and, if provided,
                a set of eigenstate energies (1D array)
            stat : list
                Custom eigenstate assignments for unitary-transformed wavefunctions.

        Returns:
            PsiTable
                Unitary-transformed wavefunction set. If energies are provided,
                these are stored in PsiTable.enr.
        """
        try:
            rotmat, enr = arg
        except ValueError:
            rotmat = arg
            enr = None

        try:
            shape = rotmat.shape
        except AttributeError:
            raise AttributeError(f"bad argument type '{type(rotmat)}' for rotation matrix") from None

        if enr is None:
            pass
        elif isinstance(enr, (list, tuple, np.ndarray)):
            if shape[0] != len(enr):
                raise ValueError(f"shapes of rotation matrix = {shape} and energy array " + \
                    f"= {len(enr)} are not aligned: {shape[0]} != {len(enr)}") from None
        else:
            raise ValueError(f"bad argument type '{type(enr)}' for energy array") from None

        nstat = self.table['c'].shape[1]
        if shape[1] != nstat:
            raise ValueError(f"shapes of rotation matrix = {shape} and eigenstate vector " + \
                f"= {self.table['c'].shape} are no aligned, {shape[1]} != {nstat}") from None

        if np.all(np.abs(rotmat) < _small):
            raise ValueError(f"all elements of rotation matrix are zero") from None
        if np.any(np.abs(rotmat) > _large):
            raise ValueError(f"some of rotation matrix elements are too large") from None
        if np.any(np.isnan(rotmat)):
            raise ValueError(f"some of rotation matrix elements are NaN") from None

        coefs = np.dot(self.table['c'][:,:], rotmat.T)

        # state assignments
        if stat is None:
            stat = []
            ndig = _assign_ndig_c2 # number of digits in |c|^2 to be kept for assignment
            c2_form = "%"+str(ndig+3)+"."+str(ndig)+"f"
            for v in rotmat:
                n = _assign_nprim # max number of primitive states to be used for assignment
                ind = (-abs(v)**2).argsort()[:n]
                elem_stat = self.table['stat'][ind]
                c2 = [c2_form%abs(v[i])**2 for i in ind]
                ll = [ elem for i in range(len(ind)) for elem in list(elem_stat[i])+[c2[i]] ]
                stat.append(ll)
        elif len(stat) != rotmat.shape[0]:
            raise ValueError(f"shapes of rotation matrix = {rotmat.shape} and 'stat' = {len(stat)} " + \
                f"are not aligned: {rotmat.shape[0]} != {len(stat)}") from None
        prim = [elem for elem in self.table['prim']]
        res = PsiTable(prim, stat, coefs)

        if enr is not None:
            res.enr = np.array(enr, dtype=np.float64)

        return res


    def del_zero_stat(self, prim=None, stat=None, coefs=None, thresh=1e-12):
        """Deletes states with zero coefficients"""
        if all(x is None for x in (prim,stat,coefs)):
            prim = self.table['prim'].copy()
            stat = self.table['stat'].copy()
            coefs = self.table['c'].copy()
            freturn = lambda prim, stat, coefs: PsiTable(prim, stat, coefs)
        elif all(x is not None for x in (prim,stat,coefs)):
            freturn = lambda prim, stat, coefs: (prim, stat, coefs)
        else:
            raise ValueError(f"expecting none or at least three arguments 'prim', 'stat', and 'coefs'") from None
        nstat = coefs.shape[1]
        ind = [istat for istat in range(nstat) if all(abs(val) < thresh for val in coefs[:,istat])]
        coefs2 = np.delete(coefs, ind, 1)
        stat2 = np.delete(stat[:nstat], ind, 0)
        prim2 = [elem for elem in prim]
        if len(stat2)==0:
            return None # somehow deleted all states
        return freturn(prim2, stat2, coefs2)


    def del_zero_prim(self, prim=None, stat=None, coefs=None, thresh=1e-12):
        """Deletes primitives that are not coupled by states"""
        if all(x is None for x in (prim,stat,coefs)):
            prim = self.table['prim'].copy()
            stat = self.table['stat'].copy()
            coefs = self.table['c'].copy()
            freturn = lambda prim, stat, coefs: PsiTable(prim, stat, coefs)
        elif all(x is not None for x in (prim,stat,coefs)):
            freturn = lambda prim, stat, coefs: (prim, stat, coefs)
        else:
            raise ValueError(f"expecting none or at least three arguments 'prim', 'stat', and 'coefs'") from None
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
        """Deletes duplicate states"""
        if all(x is None for x in (prim,stat,coefs)):
            prim = self.table['prim'].copy()
            stat = self.table['stat'].copy()
            coefs = self.table['c'].copy()
            freturn = lambda prim, stat, coefs: PsiTable(prim, stat, coefs)
        elif all(x is not None for x in (prim,stat,coefs)):
            freturn = lambda prim, stat, coefs: (prim, stat, coefs)
        else:
            raise ValueError(f"expecting none or at least three arguments 'prim', 'stat', and 'coefs'") from None
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
    """Basic class to handle operations on rotational wave functions, which are
    represented by two tables of superposition coefficients (PsiTable class),
    one for molecule-fixed quanta K (K-subspace) and another for laboratory-frame
    quanta M (M-subspace).

    Attributes:
        k : PsiTable
            Table of superposition coefficients for the K-subspace.
        m : PsiTable
            Table of superposition coefficients for the M-subspace.
    """

    def __init__(self, psik, psim):
        if not isinstance(psik, PsiTable):
            raise TypeError(f"bad argument type '{type(psik)}'")
        if not isinstance(psim, PsiTable):
            raise TypeError(f"bad argument type '{type(psim)}'")

        # initialize using PsiTable.__init__
        # this way some of the attributes (added dynamically) in psik and psim will be lost
        '''
        nstat = psim.table['c'].shape[1]
        self.m = PsiTable(psim.table['prim'], psim.table['stat'][:nstat], psim.table['c'])
        nstat = psik.table['c'].shape[1]
        self.k = PsiTable(psik.table['prim'], psik.table['stat'][:nstat], psik.table['c'])
        '''

        # initialize using PsiTable.fromPsiTable
        # this way psik and psim will be deep-copied keeping all dynamically added attributes
        self.k = PsiTable.fromPsiTable(psik)
        self.m = PsiTable.fromPsiTable(psim)


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


    def append(self, arg, check_duplicate_stat=False, del_duplicate_stat=False, del_zero_stat=False, \
               del_zero_prim=False, thresh=1e-12):
        """Appends two wave function sets together: self + arg.

        If requested:
            'check_duplicate_stat' = True: checks for duplicate states.
            'del_duplicate_stat' = True: deletes duplicate states.
            'del_zero_stat' = True: deletes states with all coefficients below 'thresh'.
            'del_zero_prim' = True: deletes primitive functions that have negligible
                contribution (below 'thresh') to all states.

        Args:
            arg : PsiTableMK
                Appended wave function set.
        """
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        res_k = self.k.append(arg.k, check_duplicate_stat, del_duplicate_stat, del_zero_stat, del_zero_prim, thresh)
        res_m = self.m.append(arg.m, check_duplicate_stat, del_duplicate_stat, del_zero_stat, del_zero_prim, thresh)
        return PsiTableMK(res_k, res_m)


    def overlap(self, arg):
        """Computes overlap < self | arg >.

        Args:
            arg : PsiTableMK
                Wave function set for overlap.

        Returns:
            ovlp_k : numpy 2D array
                Overlap for K-subspace.
            ovlp_m : numpy 2D array
                Overlap for M-subspace.
        """
        try:
            x = arg.m
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
        try:
            x = arg.k
        except AttributeError:
            raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
        ovlp_k = self.k.overlap(arg.k)
        ovlp_m = self.m.overlap(arg.m)
        return ovlp_k, ovlp_m


    def overlap_k(self, arg):
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
        ovlp_m = self.m.overlap(arg.m)
        return ovlp_m


    def rotate(self, krot=None, mrot=None, kstat=None, mstat=None):
        """Applies unitary transformation.

        Args:
            krot : numpy 2D array or tuple (numpy 2D array, 1D array)
                Unitary transformation matrix (2D array) and, if provided,
                a set of eigenstate energies (1D array) for the K-subspace.
            mrot : numpy 2D array or tuple (numpy 2D array, 1D array)
                Unitary transformation matrix (2D array) and, if provided,
                a set of eigenstate energies (1D array) for the M-subspace.
            kstat : list
                Custom eigenstate assignments for unitary-transformed wave functions
                in the K-subspace.
            mstat : list
                Custom eigenstate assignments for unitary-transformed wave functions
                in the M-subspace.

        Returns:
            PsiTableMK
                Unitary-transformed wave function set. If energies are provided,
                these are stored in PsiTableMK.k.enr and PsiTableMK.m.enr
                for K-, and M-subspaces, respectively.
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
        """Returns number of primitive basis functions"""
        return self.k.table['c'].shape[1]

    @nstates.setter
    def nstates(self, val):
        raise AttributeError(f"setting number of states is not permitted") from None


    @property
    def enr(self):
        """Returns energies of eigenstates"""
        nstat = self.k.table['c'].shape[1]
        try:
            enr = self.k.enr
        except AttributeError:
            raise AttributeError(f"attribute 'enr' not found, use PsiTableMK.rotate to assign energies") from None
        return enr

    @enr.setter
    def enr(self, val):
        raise AttributeError(f"setting energies is not permitted, use PsiTableMK.rotate to assign energies") from None


    @property
    def assign(self):
        """Returns assignment of eigenstates.

        In order to control the maximal number of primitive functions used
        for assignment, change _assign_nprim, to control the number of printed
        significant digits for coefficients of primitive functions,
        change _assign_ndig_c2
        """
        nstat = self.k.table['c'].shape[1]
        assign = []
        for a in self.k.table['stat'][:nstat]:
            assign.append(a[:self.assign_nprim*4])
        return assign

    @assign.setter
    def assign(self, val):
        raise AttributeError(f"setting assignment is not permitted") from None


    @property
    def assign_nprim(self):
        """Number of primitive functions printed in the assignment"""
        try:
            return self.assign_no_prim
        except AttributeError:
            return 1

    @assign_nprim.setter
    def assign_nprim(self, val):
        self.assign_no_prim = int(val)


    @property
    def sym(self):
        """Returns symmetry of eigenstates"""
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
        elif isinstance(val, (tuple, list)):
            nstat = self.k.table['c'].shape[1]
            if len(val) != nstat:
                raise ValueError(f"length of symmetry list = {len(val)} " + \
                    f"is not consistent with the number of states = {nstat}") from None
            self.k.sym = np.array([elem for elem in val])
        else:
            raise TypeError(f"bad argument type '{type(val)}' for symmetry") from None


    @json.register_encoder("PsiTableMK")
    def json_encode(self):
        res = {}
        res["__PsiTableMK__"] = True
        res["k.table"] = self.k.table
        res["m.table"] = self.m.table
        try:
            res["abc"] = self.abc
        except AttributeError:
            pass
        try:
            res["sym"] = self.sym.tolist()
        except AttributeError:
            pass
        try:
            res["enr"] = self.enr.tolist()
        except AttributeError:
            pass
        return res

    @json.register_decoder("__PsiTableMK__")
    def json_decode(dct):
        k_table = dct["k.table"]
        nprim = k_table['c'].shape[0]
        nstat = k_table['c'].shape[1]
        prim = k_table['prim'][:nprim]
        stat = k_table['stat'][:nstat]
        coefs = k_table['c']
        PsiTableK = PsiTable(prim, stat, coefs)

        m_table = dct["m.table"]
        nprim = m_table['c'].shape[0]
        nstat = m_table['c'].shape[1]
        prim = m_table['prim'][:nprim]
        stat = m_table['stat'][:nstat]
        coefs = m_table['c']
        PsiTableM = PsiTable(prim, stat, coefs)

        bas = PsiTableMK(PsiTableK, PsiTableM)

        try:
            bas.abc = dct["abc"]
        except KeyError:
            pass
        try:
            bas.sym = dct["sym"]
        except KeyError:
            pass
        try:
            enr = np.array(dct["enr"])
            rotmat = np.eye(bas.nstates, dtype=np.float64)
            bas.rotate(krot=(rotmat, enr))
        except KeyError:
            pass
        return bas



class SymtopBasis(PsiTableMK):
    """Basis of symmetric-top functions for selected J.

    Args:
        J : int
            Quantum number of the rotational angular momentum.
        linear : bool
            Set to True if molecule is linear, in this case quantum number K = 0.
        m_list : list
            List of m quanta spanned by basis, by default m=-J..J
    """

    def __init__(self, J, linear=False, m_list=[]):

        try:
            self.J = int(round(J))
        except TypeError:
            raise TypeError(f"J = '{J}' is not a number") from None
        assert (self.J>=0), f"J = {J} < 0"

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
        if len(m_list) > 0:
            if any([abs(m) > J for m in m_list]):
                raise ValueError(f"some of the absolute values of m quanta in " + \
                    f"'m_list' = {m_list} are larger than the value of J = {J}") from None
            prim_m = [(int(J),int(m)) for m in m_list]
        else:
            prim_m = [(int(J),int(m)) for m in range(-J,J+1)]
        coefs_m = np.eye(len(prim_m), dtype=np.complex128)

        # initialize
        PsiTableMK.__init__(self, PsiTable(prim, bas, coefs), PsiTable(prim_m, prim_m, coefs_m))


    def wang_coefs(self, j, k, tau):
        """Wang's symmetrization coefficients c1 and c2 for symmetric-top function
        in the form |J,k,tau> = c1|J,k> + c2|J,-k>.

        Args:
            j, k, tau : int 
                Rotational quantum numbers, where k = 0.. J and tau = 0 or 1
                defines parity as (-1)^tau.

        Returns:
            coefs : list
                Wang's symmetrization coefficients, coefs=[c1,c2] for k>0
                and coefs=[c1] for k=0.
            kval : list
                List of k-values, kval=[k,-k] for k>0 and kval=[k] for k=0.
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

