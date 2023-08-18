import numpy as np
import math
from richmol.rot.basis import PsiTableMK, PsiTable
from scipy.sparse import csr_matrix
from richmol.field import CarTens
from richmol.rot.molecule import Molecule, mol_frames
from richmol.rot.solution import hamiltonian
import collections
from collections.abc import Callable
import inspect
import py3nj
# import quadpy
from richmol_quad import lebedev, lebedev_npoints
import quaternionic
import spherical


_sym_tol = 1e-12 # max difference between off-diagonal elements of symmetric matrix
_use_pywigxjpf = False # use pywigxjpf module for computing Wigner symbols


def config(use_pywigxjpf=False):
    if use_pywigxjpf is True:
        print("... use `pywigxjpf` module for computing Wigner symbols in rot.labtens")
        global _use_pywigxjpf, wig_table_init, wig_temp_init, wig3jj, wig_temp_free, wig_table_free
        from pywigxjpf import wig_table_init, wig_temp_init, wig3jj, wig_temp_free, wig_table_free
        _use_pywigxjpf = True


class LabTensor(CarTens):
    """Represents rotational matrix elements of laboratory-frame Cartesian tensor operator

    This is a subclass of :py:class:`richmol.field.CarTens` class.

    Attrs:
        Us : numpy.complex128 2D array
            Cartesian-to-spherical tensor transformation matrix.
        Ux : numpy.complex128 2D array
            Spherical-to-Cartesian tensor transformation matrix, Ux = np.conj(Us.T)
        tens_flat : 1D array
            Flattened molecular-frame Cartesian tensor.
        molecule : :py:class:`richmol.rot.Molecule`
            Molecular parameters.

    Args:
        arg : numpy.ndarray, list or tuple, :py:class:`richmol.rot.Molecule`, function, or str
            Depending on the type of argument, initializes lab-frame molecular tensor operator
            (`arg` is numpy.ndarray, list, or tuple), field-free Hamiltonian operator
            (`arg` is :py:class:`richmol.rot.Molecule`), arbitrary function of spherical
            coordinates theta and phi (`arg` is function(theta, phi)), or matrix elements of some
            named operators (`arg` is str).
            Named operators: "cos2theta" for cos²(theta), "costheta" for cos(theta).
        basis : :py:class:`richmol.rot.Solution`
            Rotational wave functions.
        thresh : float
            Threshold for neglecting matrix elements.

    Kwargs:
        bra, ket : function(**kw)
            State filters for bra and ket basis sets, take as arguments state quantum numbers, symmetry,
            and energy, i.e., `J`, `m`, `k`, `sym`, and `enr`, and return True or False depending if
            the corresponding state needs to be included (True) or excluded (False) form the basis.
            By default, all states spanned by basis are included.
            The following keyword arguments are passed into the bra and ket functions:
                J : float (round to first decimal)
                    Value of J (or F) quantum number
                sym : str
                    State symmetry
                enr : float
                    State energy
                m : str
                    State assignment in the M subspace, usually just a value of the m quantum number
                k : str
                    State assignment in the K subspace, which are state's rotational or ro-vibrational
                    quanta and energy combined in a string
        wig_jmax : int
            Maximal value of J quantum number used in the expansion of input user-defined function
            of spherical coordinates in terms of Wigner D-functions (used when `arg` is function).
        leb_deg : int (deprecated, use `leb_ind`)
            Degree of angular Lebedev quadrature used for computing the Wigner expansion coefficients
            (used when `arg` is function)
        leb_ind : int
            Index (0..32) of the angular Lebedev quadrature used for computing the
            Wigner expansion coefficients (used when `arg` is a function)
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

    def __init__(self, arg, basis, thresh=None, **kwargs):

        tens = None
        func = None
        tens_str = None

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
        # if arg is function of spherical coordinates
        elif isinstance(arg, Callable):
            func = arg
        elif isinstance(arg, str):
            tens_str = arg
        else:
            raise TypeError(f"bad argument type for 'arg': '{type(arg)}'") from None

        # initialize tensor

        if tens is not None:
            self.init_tens_from_rank(tens, thresh=thresh)
        elif func is not None:
            self.init_tens_from_func(func, thresh=thresh, **kwargs)
        elif tens_str is not None:
            self.init_tens_from_name(tens_str, thresh=thresh)

        # matrix elements

        self.kmat, self.mmat = self.matelem(basis, thresh=thresh)

        # for tensor representation of field-free Hamiltonian, make basis an attribute

        if isinstance(arg, Molecule):

            # self.basis = basis # can be deprecated, replaced by `symtop_basis`

            csr_mat = lambda m, thr: csr_matrix(np.where(np.abs(m) > thr, m, 0))
            self.symtop_basis = {
                round(float(J), 1): {
                    sym: {
                        'm': {
                            'prim': basis[J][sym].m.table['prim'],
                            'stat': basis[J][sym].m.table['stat'],
                            'c': csr_mat(basis[J][sym].m.table['c'], thresh)
                        },
                        'k': {
                            'prim': basis[J][sym].k.table['prim'],
                            'stat': basis[J][sym].k.table['stat'],
                            'c': csr_mat(basis[J][sym].k.table['c'], thresh)
                        }
                    }
                    for sym in basis[J].keys()
                }
                for J in basis.keys()
            }

        # for pure Cartesian tensor, keep in the attributes only basis set dimensions and quanta

        Jlist = [round(float(J),1) for J in basis.keys()]
        symlist = {J : [sym for sym in basis[J].keys()] for J in basis.keys()}
        dim_k = { J : { sym : bas_sym.k.table['c'].shape[1]
                  for sym, bas_sym in bas_J.items() }
                  for J, bas_J in basis.items() }
        dim_m = { J : { sym : bas_sym.m.table['c'].shape[1]
                  for sym, bas_sym in bas_J.items() }
                  for J, bas_J in basis.items() }
        dim = {J : {sym : dim_m[J][sym] * dim_k[J][sym] for sym in symlist[J]} for J in Jlist}

        # use int(m) as m-assignment
        quanta_m = { J : { sym : [ int(elem[1]) for elem in bas_sym.m.table['stat'][:dim_m[J][sym]] ]
                     for sym, bas_sym in bas_J.items() }
                     for J, bas_J in basis.items() }
        try:
            # use (k, energy) as k-assignment
            # quanta_k = { J : { sym : [ (" ".join(q for q in elem), e)
            #              for elem,e in zip(bas_sym.k.table['stat'][:dim_k[J][sym]], bas_sym.k.enr) ]
            #              for sym, bas_sym in bas_J.items() }
            #              for J, bas_J in basis.items() }
            quanta_k = { J : { sym : [ (" ".join(q for q in elem), e)
                         for elem,e in zip(bas_sym.assign, bas_sym.k.enr) ]
                         for sym, bas_sym in bas_J.items() }
                         for J, bas_J in basis.items() }
        except AttributeError:
            # use (k, None) as k-assignment
            # quanta_k = { J : { sym : [ (" ".join(q for q in elem), None)
            #              for elem in bas_sym.k.table['stat'][:dim_k[J][sym]] ]
            #              for sym, bas_sym in bas_J.items() }
            #              for J, bas_J in basis.items() }
            quanta_k = { J : { sym : [ (" ".join(q for q in elem), None)
                         for elem in bas_sym.assign ]
                         for sym, bas_sym in bas_J.items() }
                         for J, bas_J in basis.items() }

        self.Jlist1 = Jlist
        self.symlist1 = symlist
        self.quanta_k1 = quanta_k
        self.quanta_m1 = quanta_m
        self.dim_k1 = dim_k
        self.dim_m1 = dim_m
        self.dim1 = dim

        self.Jlist2 = Jlist
        self.symlist2 = symlist
        self.quanta_k2 = quanta_k
        self.quanta_m2 = quanta_m
        self.dim_k2 = dim_k
        self.dim_m2 = dim_m
        self.dim2 = dim

        self.filter(**kwargs)


    def init_tens_from_rank(self, tens, thresh=None):
        """Initializes components of the lab-frame tensor from the molecular-frame one
        """
        if thresh is None:
            thresh_ = np.finfo(np.complex128).eps
        else:
            thresh_ = thresh

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

        self.Us[np.abs(self.Us) < thresh_] = 0
        self.Ux[np.abs(self.Ux) < thresh_] = 0


    def init_tens_from_func(self, func, wig_jmax=100, leb_deg=131, leb_ind=32,
                            thresh=None, **kwargs):
        """Initializes components of lab-frame tensor from arbitrary function `func`(theta, phi)
        of spherical angles

        Expands function in terms of spherical-top functions |J,k=0,m> with J=0..`wig_jmax` and m=-J..J,
        computes expansion coefficients <func|J,k=0,m> using Lebedev quadrature
        defined by its index `leb_ind`=0..32 (formerly by degree `leb_deg`),
        expansion coefficients smaller than `thresh` are neglected
        """
        if thresh is None:
            thresh_ = np.finfo(np.complex128).eps
        else:
            thresh_ = thresh

        self.rank = 0
        self.cart = ["0"]
        self.tens_flat = np.array([1], dtype=np.float64)

        # init Lebedev quadrature

        ##### Using quadpy (which sadly is not free anymore)
        # quad_name = "lebedev_" + str(leb_deg).zfill(3)
        # try:
        #     leb = quadpy.u3.schemes[quad_name]
        # except KeyError:
        #     raise KeyError(f"quadrature '{quad_name}' not found, available schemes: " + \
        #         f"{[key for key in quadpy.u3.schemes.keys() if key.startswith('lebedev')]}") from None
        # theta, phi = leb().theta_phi
        # weights = leb().weights

        ##### Using fortran implementation
        leb_npoints = lebedev_npoints(leb_ind)
        (theta, phi), weights = lebedev(leb_npoints)

        # compute function on quadrature grid, multiply by quadrature weights
        f = func(theta, phi) * weights

        # compute overlap integrals <func|J,k=0,m>
        wigner = spherical.Wigner(wig_jmax)
        R = quaternionic.array.from_spherical_coordinates(theta, phi)
        ovlp = {J : np.zeros(2*J+1, dtype=np.complex128) for J in range(wig_jmax+1)}
        for rr, ff in zip(R, f):
            D = wigner.D(rr)
            for J in range(wig_jmax+1):
                ovlp[J] += np.array([np.conj(D[wigner.Dindex(J, m, 0)]) for m in range(-J, J+1)], dtype=np.complex128) * ff

        for J in range(wig_jmax+1):
            ovlp[J] = ovlp[J] * 2*np.pi * np.sqrt((2*J+1)/(8*np.pi**2))
            ovlp[J][np.abs(ovlp[J]) < thresh_] = 0
            if np.all(np.abs(ovlp[J] < thresh_)):
                del ovlp[J]

        # initialize tensor Cartesian-to-spherical transformation
        self.os = [(o, s) for o in ovlp.keys() for s in range(-o, o)]
        self.Ux = np.zeros((1, len(self.os)), dtype=np.complex128)
        self.Us = np.zeros((len(self.os), 1), dtype=np.complex128)

        ind_sigma = {(omega, sigma) : [s for s in range(-omega, omega+1)].index(sigma) for (omega, sigma) in self.os}
        for i,(omega, sigma) in enumerate(self.os):
            isigma = ind_sigma[(omega, sigma)]
            self.Ux[0, i] = ovlp[omega][isigma]
            if sigma == 0:
                self.Us[i, 0] = 1


    def init_tens_from_name(self, tens_name, thresh=None):
        """Initializes Cartesian tensor operator from given name.
        Used to initialize simple "observable" quantities such as cos(theta), cos^2(theta)
        """
        if tens_name.lower() == "costheta":
            self.rank = 0
            self.cart = ["0"]
            self.os = [(1,0)]
            self.tens_flat = np.array([1], dtype=np.float64)
            self.Ux = np.zeros((1, len(self.os)), dtype=np.complex128)
            self.Us = np.zeros((len(self.os), 1), dtype=np.complex128)
            self.Us[0,:] = 1
            self.Ux[:,0] = 1
        elif tens_name.lower() == "cos2theta":
            self.rank = 0
            self.cart = ["0"]
            self.os = [(2,0)]
            self.tens_flat = np.array([1], dtype=np.float64)
            self.Ux = np.zeros((1, len(self.os)), dtype=np.complex128)
            self.Us = np.zeros((len(self.os), 1), dtype=np.complex128)
            self.Us[0,:] = 1
            self.Ux[:,0] = 2.0/3.0
        else:
            raise ValueError(f"unknown name for tensor operator: '{tens_name}'") from None


    def ham_proj(self, basis):
        """Computes action of field-free Hamiltonian operator onto a wave function

        Args:
            basis : :py:class:`richmol.rot.SymtopBasis` or :py:class:`richmol.rot.basis.PsiTableMK`
                Wave functions in symmetric-top basis
    
        Returns:
            res : nested dict
                Hamiltonian-projected wave functions as dictionary of :py:class:`richmol.rot.SymtopBasis`
                for single Cartesian and single irreducible component, i.e., res[0]['0']
        """
        H = hamiltonian(self.molecule, basis)

        # Hamiltonian does not act on the m-part of the basis, the arithmetic operations
        # in H, however, de-normalize the m-matrix, so we reset it to the m-part of the basis
        H.m = basis.m

        irreps = set(omega for (omega,sigma) in self.os)
        res = { irrep : { cart : H for cart in self.cart } for irrep in irreps }
        return res


    def tens_proj(self, basis, maxJ=None, minJ=0, thresh=None):
        """Computes action of tensor operator onto wave function

        Args:
            basis : :py:class:`richmol.rot.SymtopBasis` or :py:class:`richmol.rot.basis.PsiTableMK`
                Wave functions in symmetric-top basis.
            maxJ : int
                Maximal value of J quantum number allowed in tensor-projected wave functions.
                If set to None (default), all projections will be computed.
            minJ : int
                Minimal value of J quantum number allowed in tensor-projected wave functions.
            thresh : float
                Threshold for considering elements of :py:attr:`Ux` and :py:attr:`Us` tensor
                transformation matrices as zero. If set to None, numpy-determined machine limit
                for numpy.complex128 floating point type is used.

        Returns:
            res : nested dict
                Tensor-projected wave functions as dictionary of :py:class:`richmol.rot.SymtopBasis`
                objects, for different Cartesian and irreducible components of tensor, i.e., res[irrep][cart].
                The K-subspace projections are independent of Cartesian component cart and computed
                only for a single component cart=self.cart[0].
        """
        try:
            jm_table = basis.m.table
        except AttributeError:
            raise AttributeError(f"'{basis.__class__.__name__}' has no attribute 'm'") from None
        try:
            jk_table = basis.k.table
        except AttributeError:
            raise AttributeError(f"'{basis.__class__.__name__}' has no attribute 'k'") from None

        if thresh is None:
            thresh_ = np.finfo(np.complex128).eps
        else:
            thresh_ = thresh

        irreps = set(omega for (omega,sigma) in self.os)
        dj_max = max(irreps)    # selection rules |j1-j2|<=omega
        sigma_ind = {omega : np.array([ind for ind,(o,s) in enumerate(self.os) if o==omega]) for omega in irreps}
        sigma = {omega : np.array([s for (o,s) in self.os if o==omega]) for omega in irreps}

        # generate tables for tensor-projected set of basis functions

        jmin = min([min(j for (j,k) in jk_table['prim']), min(j for (j,m) in jm_table['prim'])])
        jmax = max([max(j for (j,k) in jk_table['prim']), max(j for (j,m) in jm_table['prim'])])

        if maxJ is None:
            # generate all J, m, and k quanta projected by tensor
            prim_k = [(int(J),int(k)) for J in range(max([minJ, jmin-dj_max]), jmax+1+dj_max) for k in range(-J,J+1)]
            prim_m = [(int(J),int(m)) for J in range(max([minJ, jmin-dj_max]), jmax+1+dj_max) for m in range(-J,J+1)]
        else:
            # restrict J quanta projected by tensor to range [minJ, maxJ]
            prim_k = [(int(J),int(k)) for J in range(max([minJ, jmin-dj_max]), min([maxJ+1, jmax+1+dj_max])) for k in range(-J,J+1)]
            prim_m = [(int(J),int(m)) for J in range(max([minJ, jmin-dj_max]), min([maxJ+1, jmax+1+dj_max])) for m in range(-J,J+1)]

        nstat_k = jk_table['c'].shape[1]
        nstat_m = jm_table['c'].shape[1]

        stat_m = jm_table['stat'][:nstat_m]
        stat_k = jk_table['stat'][:nstat_k]

        # output dictionaries
        res = { irrep : { cart : PsiTableMK(PsiTable(prim_k, stat_k), PsiTable(prim_m, stat_m))
                for cart in self.cart } for irrep in irreps }

        # some initializations in pywigxjpf module for computing 3j symbols
        if _use_pywigxjpf:
            wig_table_init((jmax+dj_max)*2, 3)
            wig_temp_init((jmax+dj_max)*2)

        # compute K|psi>

        UsT = {irrep : np.dot(self.Us[sigma_ind[irrep], :], self.tens_flat) for irrep in irreps}
        ind = {irrep : np.where(np.abs(UsT[irrep]) > thresh_)[0] for irrep in irreps}
        nos = {irrep : len(ind[irrep]) for irrep in irreps if len(ind[irrep]) > 0}
        sig = {irrep : sigma[irrep][ind[irrep]] for irrep in nos.keys()}
        UsT = {irrep : UsT[irrep][ind[irrep]] for irrep in nos.keys()}

        cart0 = self.cart[0]
        for ind1,(j1,k1) in enumerate(jk_table['prim']):
            for ind2,(j2,k2) in enumerate(prim_k):

                # compute <j2,k2|K-tensor|j1,k1>
                fac = (-1)**abs(k2)
                for irrep, n in nos.items():
                    if _use_pywigxjpf:
                        threeJ = np.asarray([wig3jj(j1*2, irrep*2, j2*2, k1*2, s*2, -k2*2) for s in sig[irrep]], dtype=np.float64)
                    else:
                        threeJ = py3nj.wigner3j([j1*2]*n, [irrep*2]*n, [j2*2]*n, [k1*2]*n, sig[irrep]*2, [-k2*2]*n)
                    me = np.dot(threeJ, UsT[irrep]) * fac
                    res[irrep][cart0].k.table['c'][ind2,:] += me * jk_table['c'][ind1,:]

        # compute M|psi>

        Ux = {irrep : self.Ux[:, sigma_ind[irrep]] for irrep in irreps}
        ind = {irrep : list(set(i for i in np.where(np.abs(Ux[irrep]) > thresh_)[1])) for irrep in irreps}
        nos = {irrep : len(ind[irrep]) for irrep in irreps if len(ind[irrep]) > 0}
        sig = {irrep : sigma[irrep][ind[irrep]] for irrep in nos.keys()}

        for ind1,(j1,m1) in enumerate(jm_table['prim']):
            for ind2,(j2,m2) in enumerate(prim_m):

                # compute <j2,m2|M-tensor|j1,m1>
                fac = np.sqrt((2*j1+1)*(2*j2+1)) * (-1)**abs(m2)
                for irrep, n in nos.items():
                    if _use_pywigxjpf:
                        threeJ = np.asarray([wig3jj(j1*2, irrep*2, j2*2, m1*2, s*2, -m2*2) for s in sig[irrep]], dtype=np.float64)
                    else:
                        threeJ = py3nj.wigner3j([j1*2]*n, [irrep*2]*n, [j2*2]*n, [m1*2]*n, sig[irrep]*2, [-m2*2]*n)
                    me = np.dot(np.take(Ux[irrep], ind[irrep], axis=1), threeJ) * fac
                    for icart,cart in enumerate(self.cart):
                        res[irrep][cart].m.table['c'][ind2,:] += me[icart] * jm_table['c'][ind1,:]

        # free memory in pywigxjpf module
        if _use_pywigxjpf:
            wig_temp_free()
            wig_table_free()

        return res


    def matelem(self, basis, thresh=None):
        """Computes matrix elements of tensor operator

        Args:
            basis : nested dict
                Wave functions in symmetric-top basis (:py:class:`richmol.rot.SymtopBasis class`)
                for different values of J quantum number and different symmetries,
                i.e., basis[J][sym] -> SymtopBasis.
            thresh : float
                Threshold for considering matrix elements as zero.
                If set to None, numpy-determined machine limit for numpy.complex128 floating point
                type is used.

        Returns:
            kmat : nested dict
                See :py:attr:`kmat`.
            mmat : nested dict
                See :py:attr:`mmat`.
        """
        if thresh is None:
            thresh_ = np.finfo(np.complex128).eps
        else:
            thresh_ = thresh

        # dimensions of M and K tensors

        dim_m = { J: { sym : bas.m.table['c'].shape[1] for sym, bas in symbas.items() }
                  for J, symbas in basis.items() }
        dim_k = { J: { sym : bas.k.table['c'].shape[1] for sym, bas in symbas.items() }
                  for J, symbas in basis.items() }

        # selection rules |J-J'| <= omega
        dJ_max = max(set(omega for (omega,sigma) in self.os))

        # max and min value of J spanned by basis
        maxJ = max([J for J in basis.keys()])
        minJ = min([J for J in basis.keys()])

        # compute matrix elements for different pairs of J quanta and different symmetries

        mydict = lambda: collections.defaultdict(mydict)
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
                    proj = self.tens_proj(bas2, maxJ=maxJ, thresh=thresh_)

                for J1, symbas1 in basis.items():

                    if abs(J1-J2) > dJ_max: # rigorous selection rules
                        continue

                    for sym1, bas1 in symbas1.items():

                        for irrep, bas_irrep in proj.items():
                            for cart, bas_cart in bas_irrep.items():

                                # K-tensor

                                if cart == self.cart[0]:

                                    # compute matrix elements
                                    me = bas1.overlap_k(bas_cart)

                                    # set to zero elements with absolute values smaller than a threshold
                                    me[np.abs(me) < thresh_] = 0
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
                                me = bas1.overlap_m(bas_cart)

                                # set to zero elements with absolute values smaller than a threshold
                                me[np.abs(me) < thresh_] = 0
                                me_csr = csr_matrix(me)

                                # check matrix dimensions
                                assert (np.all(me_csr.shape == (dim_m[J1][sym1], dim_m[J2][sym2]))), \
                                    f"shape of M-matrix = {me_csr.shape} is not aligned with shape " + \
                                    f"of basis set = {(dim_m[J1][sym1], dim_m[J2, sym2])} " + \
                                    f"for (J1, J2) = {(J1, J2)} and (sym1, sym2) = {(sym1, sym2)}"

                                # add matrix elements to M-tensor
                                if me_csr.nnz > 0:
                                    mmat[(J1, J2)][(sym1, sym2)][irrep][cart] = me_csr

        return kmat, mmat


    def class_name(self):
        """Generates string containing name of the parent class"""
        base = list(self.__class__.__bases__)[0]
        return base.__module__ + "." + base.__name__


def cos2theta(theta, phi):
    return np.cos(theta)**2


def cos2theta2d(theta, phi):
    tol = 1.0e-12
    if abs(theta - np.pi/2.0) <= tol or abs(theta - 3.0*np.pi/2.0) <= tol:
        if abs(phi) <= tol or abs(phi - np.pi) <= tol or abs(phi - 2.0*np.pi) <= tol:
            return 1.0
    return np.cos(theta)**2/(np.sin(theta)**2*np.sin(phi)**2 + np.cos(theta)**2)


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
