import numpy as np
import math
from richmol.pywigxjpf import wig_table_init, wig_temp_init, wig3jj, wig_temp_free, wig_table_free
from richmol.rot.basis import PsiTableMK, PsiTable
from scipy.sparse import csr_matrix
from richmol.field import CarTens
from richmol.rot.molecule import Molecule
from richmol.rot.solution import hamiltonian
from collections import defaultdict
import inspect
import os


_sym_tol = 1e-12 # max difference between off-diag elements of symmetric matrix


class LabTensor(CarTens):
    """Laboratory-frame Cartesian tensor operator

    Attrs:
        rank : int
            Rank of tensor operator.
        Us : numpy.complex128 2D array
            Cartesian-to-spherical tensor transformation matrix.
        Ux : numpy.complex128 2D array
            Spherical-to-Cartesian tensor transformation matrix (Ux = (Us^T)^*).
        cart : list of str
            Contains string labels of tensor Cartesian components
            (e.g, 'x', 'y', 'z', 'xx', 'xy', 'xz', ...) in the order corresponding
            to the order of Cartesian components in rows of 'Ux' (columns of 'Us').
        os : [(omega,sigma) for omega in range(nirrep) for sigma in range(-omega,omega+1)]
            List of spherical-tensor indices (omega,sigma) in the order corresponding
            to the spherical-tensor components in columns of 'Ux' (rows of 'Us'),
            nirrep here is the number of tensor irreducible representations.
        tens_flat : 1D array
            Contains elements of molecular-frame Cartesian tensor, flattened
            in the order corresponding to the order of Cartesian components in cart.
        molecule : rot.molecule.Molecule
            Molecular parameters.
        Jlist1 : list
            List of J quanta spanned by the basis.
        Jlist2 : list
            Equal to Jlist1.
        symlist1 : dict
            List of symmetries, for each J (as keys), spanned by the basis.
        symlist2 : dict
            Equal to symlist1.
        dim_m1, dim_k1, dim1 : nested dict
            Dimensions of M, K, and MxK tensors (or respective subspaces of basis set)
            for different values of J quanta and different symmetries,
            i.e., dim_m1[J][sym] -> int, dim_k1[J][sym] -> int.
        dim_m2, dim_k2, dim2 : nested dict
            Equal to respective dim_m1, dim_k1, dim1.
        assign_m1, assign_k1 : nested dict
            Assignments of states in M and K subspaces of basis set,
            for different values of J quanta and different symmetries,
            i.e., m_assign1[J][sym] -> list (len = dim_m1[J][sym]),
            and k_assign1[J][sym] -> list (len = dim_k1[J][sym]).
        assign_m2, assign_k2 : nested dict
            Equal to respective assign_m1, assign_k1.
        kmat : nested dict
            K-tensor matrix elements (in CSR format) for different pairs of bra
            and ket J quanta, different pairs of bra and ket symmetries,
            and different irreducible components of tensor,
            i.e., kmat[(J1, J2)][(sym1, sym2)][irrep].
        mmat : nested dict
            M-tensor matrix elements (in CSR format) for different pairs of bra
            and ket J quanta, different pairs of bra and ket symmetries,
            and different Cartesian and irreducible components of tensor,
            i.e., mmat[(J1, J2)][(sym1, sym2)][cart][irrep].

    Args:
        arg : rot.molecule.Molecule or numpy.ndarray, list or tuple
            Molecule object or molecular-frame Cartesian tensor.
            For arg = rot.molecule.Molecule, the resulting tensor represents
            field-free Hamiltonian.
        basis : nested dict
            Wave functions in symmetric-top basis (SymtopBasis class)
            for different values of J quantum number and different symmetries,
            i.e., basis[J][sym] -> SymtopBasis.
        thresh : float
            Threshold for neglecting matrix elements.
    """
    # transformation matrix from Cartesian to spherical-tensor representation
    # for tensors of different ranks (dict keys)

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

    # Cartesian components and irreducible representations for tensors of different ranks

    cart_ind = {1 : ["x","y","z"],
                2 : ["xx","xy","xz","yx","yy","yz","zx","zy","zz"]}

    irrep_ind = {1 : [(1,-1),(1,0),(1,1)],
                 2 : [(o,s) for o in range(3) for s in range(-o,o+1)] }

    def __init__(self, arg, basis, thresh=1e-14, **kwargs):

        tens = None

        # if arg is Molecule
        if isinstance(arg, Molecule):
            self.rank = 0
            self.cart = ["0"]
            self.os = [(0,0)]
            self.molecule = arg
        # if arg is tensor
        elif isinstance(arg, (tuple, list)):
            tens = np.array(arg)
        elif isinstance(arg, (np.ndarray,np.generic)):
            tens = arg
        else:
            raise TypeError(f"bad argument type for 'arg': '{type(arg)}'") from None

        # initialize tensor attributes

        if tens is not None:

            if not all(dim==3 for dim in tens.shape):
                raise ValueError(f"input tensor has inappropriate shape: '{tens.shape}' != {[3]*tens.ndim}") from None
            if np.any(np.isnan(tens)):
                raise ValueError(f"some of elements of input tensor are NaN") from None

            rank = tens.ndim
            self.rank = rank
            try:
                self.Us = self.tmat_s[rank]
                self.Ux = self.tmat_x[rank]
                self.os = self.irrep_ind[rank]
                self.cart = self.cart_ind[rank]
            except KeyError:
                raise NotImplementedError(f"tensor of rank = {rank} is not implemented") from None

            # save molecular-frame tensor in flatted form with the elements following the order in self.cart
            self.tens_flat = np.zeros(len(self.cart), dtype=tens.dtype)
            for ix,sx in enumerate(self.cart):
                s = [ss for ss in sx]                  # e.g. split "xy" into ["x","y"]
                ind = ["xyz".index(ss) for ss in s]    # e.g. convert ["x","y"] into [0,1]
                self.tens_flat[ix] = tens.item(tuple(ind))

            # special cases if tensor is symmetric and/or traceless
            if self.rank==2:
                symmetric = lambda tens, tol=_sym_tol: np.all(np.abs(tens-tens.T) < tol)
                traceless = lambda tens, tol=_sym_tol: abs(np.sum(np.diag(tens))) < tol
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

        # matrix elements

        self.kmat, self.mmat = self.matelem(basis, thresh)

        # for tensor representation of field-free Hamiltonian, make basis an attribute

        if isinstance(arg, Molecule):
            self.basis = basis

        # for pure Cartesian tensor, keep in the attributes only basis set quanta

        Jlist = [J for J in basis.keys()]
        symlist = {J : [sym for sym in basis[J].keys()] for J in basis.keys()}
        dim_k = { J : { sym : bas_sym.k.table['c'].shape[1] for sym, bas_sym in bas_J.items() }
                  for J, bas_J in basis.items() }
        dim_m = { J : { sym : bas_sym.m.table['c'].shape[1] for sym, bas_sym in bas_J.items() }
                  for J, bas_J in basis.items() }
        quanta_m = { J : { sym : [ " ".join(q for q in elem)
                                   for elem in bas_sym.m.table['stat'][:dim_m[J][sym]] ]
                           for sym, bas_sym in bas_J.items() }
                     for J, bas_J in basis.items() }
        quanta_k = { J : { sym : [ " ".join(q for q in elem)
                                   for elem in bas_sym.k.table['stat'][:dim_k[J][sym]] ]
                           for sym, bas_sym in bas_J.items() }
                     for J, bas_J in basis.items() }

        loc = locals()
        for name in ("Jlist", "symlist", "dim_m", "dim_k", "quanta_m", "quanta_k"):
            setattr(self, name, loc[name])

        # init object as CarTens (to apply state filters)

        # write tensor into temp file and reload in CarTens()
        filename = "tmp.h5"
        self.store(filename=filename, name=retrieve_name(self), thresh=thresh)
        CarTens.__init__(self, filename=filename,  name=retrieve_name(self), **kwargs)
        if os.path.exists("tmp.h5"):
            os.remove("tmp.h5")


    def ham_proj(self, basis):
        """Computes action of field-free Hamiltonian operator onto a wave function

        Args:
            basis : SymtopBasis or PsiTableMK
                Wave functions in symmetric-top basis
    
        Returns:
            res : nested dict
                Hamiltonian-projected wave functions as dictionary of SymtopBasis
                objects, for single Cartesian and single irreducible component,
                i.e., res['0'][0].
        """
        H = hamiltonian(self.molecule, basis)
        irreps = set(omega for (omega,sigma) in self.os)
        res = { cart : { irrep : H for irrep in irreps } for cart in self.cart }
        return res


    def tens_proj(self, basis):
        """Computes action of tensor operator onto a wave function

        Args:
            basis : SymtopBasis or PsiTableMK
                Wave functions in symmetric-top basis.

        Returns:
            res : nested dict
                Tensor-projected wave functions as dictionary of SymtopBasis objects,
                for different Cartesian and irreducible components of tensor,
                i.e., res[cart][irep].
                The K-subspace projections are independent of Cartesian component
                cart and computed only for a single component cart=self.cart[0].
        """
        try:
            jm_table = basis.m.table
        except AttributeError:
            raise AttributeError(f"'{basis.__class__.__name__}' has no attribute 'm'") from None
        try:
            jk_table = basis.k.table
        except AttributeError:
            raise AttributeError(f"'{basis.__class__.__name__}' has no attribute 'k'") from None

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

        # output dictionaries
        res = { cart : { irrep : PsiTableMK(PsiTable(prim_k, stat_k), PsiTable(prim_m, stat_m))
                for irrep in irreps } for cart in self.cart }

        # some initializations in pywigxjpf module for computing 3j symbols
        wig_table_init((jmax+dj_max)*2, 3)
        wig_temp_init((jmax+dj_max)*2)

        # compute K|psi>
        cart0 = self.cart[0]
        for ind1,(j1,k1) in enumerate(jk_table['prim']):
            for ind2,(j2,k2) in enumerate(prim_k):
                fac = (-1)**abs(k2)
                # compute <j2,k2|K-tensor|j1,k1>
                threeJ = np.asarray([wig3jj([j1*2, o*2, j2*2, k1*2, s*2, -k2*2]) for (o,s) in self.os], dtype=np.float64)
                for irrep in irreps:
                    ind = os_ind[irrep]
                    me = np.dot(threeJ[ind], np.dot(self.Us[ind,:], self.tens_flat)) * fac
                    res[cart0][irrep].k.table['c'][ind2,:] += me * jk_table['c'][ind1,:]

        # compute M|psi>
        for ind1,(j1,m1) in enumerate(jm_table['prim']):
            for ind2,(j2,m2) in enumerate(prim_m):
                fac = np.sqrt((2*j1+1)*(2*j2+1)) * (-1)**abs(m2)
                # compute <j2,m2|M-tensor|j1,m1>
                threeJ = np.asarray([wig3jj([j1*2, o*2, j2*2, m1*2, s*2, -m2*2]) for (o,s) in self.os], dtype=np.float64)
                for irrep in irreps:
                    ind = os_ind[irrep]
                    me = np.dot(self.Ux[:,ind], threeJ[ind]) * fac
                    for icart,cart in enumerate(self.cart):
                        res[cart][irrep].m.table['c'][ind2,:] += me[icart] * jm_table['c'][ind1,:]

        # free memory in pywigxjpf module
        wig_temp_free()
        wig_table_free()

        return res


    def matelem(self, basis, thresh=1e-14):
        """Computes matrix elements of tensor operator

        Args:
            basis : nested dict
                Wave functions in symmetric-top basis (SymtopBasis class)
                for different values of J quantum number and different symmetries,
                i.e., basis[J][sym] -> SymtopBasis.
            thresh : float
                Threshold for neglecting matrix elements.

        Returns:
            kmat : nested dict
                See LabTensor.kmat
            mmat : nested dict
                See LabTensor.mmat
        """
        # dimensions of M and K tensors

        dim_m = { J: { sym : bas.m.table['c'].shape[1] for sym, bas in symbas.items() }
                  for J, symbas in basis.items() }
        dim_k = { J: { sym : bas.k.table['c'].shape[1] for sym, bas in symbas.items() }
                  for J, symbas in basis.items() }

        # selection rules |J-J'| <= omega
        dJ_max = max(set(omega for (omega,sigma) in self.os))

        # compute matrix elements for different pairs of J quanta and different symmetries

        mydict = lambda: defaultdict(mydict)
        mmat = mydict()
        kmat = mydict()

        for J2, symbas2 in basis.items():
            for sym2, bas2 in symbas2.items():

                try:
                    # compute H0|psi>
                    x = self.molecule
                    proj = self.ham_proj(bas2)
                except AttributeError:
                    # compute LabTensor|psi>
                    proj = self.tens_proj(bas2)

                for J1, symbas1 in basis.items():

                    if abs(J1-J2) > dJ_max: # rigorous selection rules
                        continue

                    for sym1, bas1 in symbas1.items():

                        for cart, bas_cart in proj.items():
                            for irrep, bas_irrep in bas_cart.items():

                                # K-tensor

                                if cart == self.cart[0]:

                                    # compute matrix elements
                                    me = bas1.overlap_k(bas_irrep)

                                    # set to zero elements with absolute values smaller than a threshold
                                    me[np.abs(me) < thresh] = 0
                                    me_csr = csr_matrix(me)

                                    # check matrix dimensions
                                    assert (np.all(me_csr.shape == (dim_k[J1][sym1], dim_k[J2][sym2]))), \
                                        f"shape of K-matrix = {me_csr.shape} is not aligned with shape " + \
                                        f"of basis set = {(dim_k[J1][sym1], dim_k[J2, sym2])} " + \
                                        f"for (J1, J2) = {(J1, J2)} and (sym1, sym2) = {(sym1, sym2)}"

                                    # add matrix element to K-tensor
                                    if me_csr.nnz > 0:
                                        kmat[(J1, J2)][(sym1, sym2)][irrep] = me_csr

                                # M-tensor

                                # compute matrix elements
                                me = bas1.overlap_m(bas_irrep)

                                # set to zero elements with absolute values smaller than a threshold
                                me[np.abs(me) < thresh] = 0
                                me_csr = csr_matrix(me)

                                # check matrix dimensions
                                assert (np.all(me_csr.shape == (dim_m[J1][sym1], dim_m[J2][sym2]))), \
                                    f"shape of M-matrix = {me_csr.shape} is not aligned with shape " + \
                                    f"of basis set = {(dim_m[J1][sym1], dim_m[J2, sym2])} " + \
                                    f"for (J1, J2) = {(J1, J2)} and (sym1, sym2) = {(sym1, sym2)}"

                                # add matrix elements to M-tensor
                                if me_csr.nnz > 0:
                                    mmat[(J1, J2)][(sym1, sym2)][cart][irrep] = me_csr

        return kmat, mmat


    def class_name(self):
        """Generates '__class_name__' attribute for the tensor data group in HDF5 file"""
        base = list(self.__class__.__bases__)[0]
        return base.__module__ + "." + base.__name__



def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
