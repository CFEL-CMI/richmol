import numpy as np
import scipy.sparse
from scipy.sparse import kron, csr_matrix, issparse
import scipy.constants as const
import itertools
from itertools import chain
import copy
import h5py
import inspect
import datetime
import time
import regex as re
from collections import defaultdict
from collections.abc import Mapping
from richmol import json_ext as json
import os
from numba import njit, prange, complex128

from jax.lib import xla_bridge
if xla_bridge.get_backend().platform == "gpu":
    import cupy as cp
    import cupyx


class CarTens():
    """ General class for laboratory-frame Cartesian tensor operator

    Args:
        filename : str
            Name of the HDF5 file from which tensor data is loaded.
        name : str
            Name of the data group in HDF5 file.

    Kwargs:
        thresh : float
            Threshold for neglecting matrix elements when reading from file
        bra, ket : function(**kw)
            State filters for bra and ket basis sets, take as arguments state
            quantum numbers, symmetry, and energy, i.e., `J`, `m`, `k`, `sym`,
            and `enr`, and return True or False depending on if the
            corresponding state needs to be included or excluded form the
            basis. By default, all states stored in the file are included. The
            following keyword arguments are passed into the bra and ket
            functions:
                J : float (round to first decimal)
                    Value of J (or F) quantum number
                sym : str
                    State symmetry
                enr : float
                    State energy
                m : str
                    State assignment in the M subspace, usually just a value of
                    the m quantum number
                k : str
                    State assignment in the K subspace, which are the
                    rotational or ro-vibrational quanta joined in a string

    Attrs:
        rank : int
            Rank of tensor operator.
        cart : list of str
            Contains string labels of tensor Cartesian components (e.g, 'x',
            'y', 'z', 'xx', 'xy', 'xz', ...)
        os : [ (omega, sigma) for omega in range(nirrep)
               for sigma in range(-omega, omega + 1) ]
            List of spherical-tensor indices (omega, sigma), here `nirrep` is
            the number of tensor irreducible representations.
        Jlist1, Jlist2 : list
            Lists of J quanta, spanned by the bra and ket basis sets,
            respectively.
        symlist1, symlist2 : dict
            List of symmetries, for each J, spanned by the bra and ket basis
            sets, respectively.
            Example:

            .. code-block:: python

                [sym for J in self.Jlist1 for sym in symlist1[J]]

        dim1, dim2 : nested dict
            Matrix dimensions of tensor for different J and symmetry, spanned
            by the bra and ket basis sets, respectively.
            Example:

            .. code-block:: python

                [ dim for J in self.Jlist1 for sym in symlist1[J]
                  for dim in dim1[J][sym] ]

        dim_m1, dim_k1 : nested dict
            Dimensions of M and K tensors for different J and symmetry, spanned
            by the bra basis.
            Example:

            .. code-block:: python

                [ dim for J in self.Jlist1 for sym in symlist1[J]
                  for dim in dim_m1[J][sym] ]

        dim_m2, dim_k2 : nested dict
            Same as `dim_m1` and `dim_m2` but for the ket basis.
        quanta_m1, quanta_k1 : nested dict
            M and ro-vibrational quantum numbers, respectively, for different J
            and symmetry, spanned by the bra basis set.
            The elements of `quanta_m1[J][sym]` list represent the m quantum
            number, while the elements of `quanta_k1[J][sym]` list are tuples
            (q, enr), where q is the string of ro-vibrational quantum numbers
            and enr is the ro-vibrational energy (or None).
            Example:

            .. code-block:: python

                [ int(m) for J in self.Jlist1 for sym in symlist1[J]
                  for m in quanta_m1[J][sym] ]
                [ (k, energy) for J in self.Jlist1 for sym in symlist1[J]
                  for m in quanta_k1[J][sym] ]

        quanta_m2, quanta_k2 : nested dict
            Same as quanta_m1 and `quanta_k1` but for the ket basis.
        kmat : nested dict
            K-tensor matrix elements (in CSR format) for different pairs of bra
            and ket J quanta, symmetries, and irreducible spherical-tensor
            components.
            Example:

            .. code-block:: python

                for (J1, J2), kmat_J in kmat.items():
                    for (sym1, sym2), kmat_sym in kmat_J.items():
                        for irrep, kmat_irrep in kmat_sym.items():
                            # K-subspace matrix elements
                            print(type(kmat_irrep)) # scipy.sparse.spmatrix

        mmat : nested dict
            M-tensor matrix elements (in CSR format) for different pairs of bra
            and ket J quanta, symmetries, irreducible spherical-tensor and
            Cartesian components.
            Example:

            .. code-block:: python

                for (J1, J2), mmat_J in mmat.items()
                    for (sym1, sym2), mmat_sym in mmat_J.items():
                        for irrep, mmat_irrep in mmat_sym.items():
                            for cart, mmat_cart in mmat_irrep.items():
                                # M-subspace matrix elements
                                print(type(mmat_cart)) # scipy.sparse.spmatrix

        mfmat : nested dict
            M-tensor matrix elements contracted with field. Produced after
            multiplication of tensor with a vector of X, Y, and Z field values
            (see :py:func:`field`). Has the same structure as :py:attr:`kmat`.
        eigvec : nested dict
            Eigenvectors (in dense matrix format) for different J quanta
            and different symmetries.

            NOTE: state selection filters (:py:func:`filter`) are not implemented,
            and thus have no effect, for `eigvec` attribute.

            Example:

            .. code-block:: python

                for J in eigvec.items():
                    for sym in eigvec[J].items():
                        print(type(eigvec[J][sym]))
    """

    def __init__(self, filename=None, name=None, **kwargs):

        if filename is None:
            pass
        else:
            if h5py.is_hdf5(filename):
                # read tensor from HDF5 file
                if name is None:
                    raise ValueError(
                        f"please specify the name of the data group in HDF5 " \
                            + f"file '{filename}'"
                    ) from None
                self.read(filename, name=name, **kwargs)


    def filter(self, thresh=None, bra=lambda **kw: True, ket=lambda **kw: True):
        """ Applies state selection filters to tensor matrix elements

        Args:
            thresh : float
                Threshold for neglecting matrix elements
            bra : function(**kw)
                State filter function for bra basis sets (see `bra` in kwargs
                of :py:class:`CarTens`).
            ket : function(**kw)
                State filter function for ket basis sets (see `ket` in kwargs
                of :py:class:`CarTens`).
        """
        # truncate quantum numbers

        self.ind_k1 = {
            J : { sym : [ i for i,(q,e) in enumerate(self.quanta_k1[J][sym])
                          if bra(J=J, sym=sym, k=q, enr=e) ]
                  for sym in self.symlist1[J] }
            for J in self.Jlist1
        }

        self.ind_k2 = {
            J : { sym : [ i for i,(q,e) in enumerate(self.quanta_k2[J][sym])
                          if ket(J=J, sym=sym, k=q, enr=e) ]
                  for sym in self.symlist2[J] }
            for J in self.Jlist2
        }

        self.ind_m1 = {
            J : { sym : [ i for i,q in enumerate(self.quanta_m1[J][sym])
                          if bra(J=J, sym=sym, m=q) ]
                  for sym in self.symlist1[J] }
            for J in self.Jlist1
        }
        self.ind_m2 = {
            J : { sym : [ i for i,q in enumerate(self.quanta_m2[J][sym])
                          if ket(J=J, sym=sym, m=q) ]
                  for sym in self.symlist2[J] }
            for J in self.Jlist2
        }

        self.quanta_k1 = {
            J : { sym : [ (q, e) for (q, e) in self.quanta_k1[J][sym]
                          if bra(J=J, sym=sym, k=q, enr=e) ]
                  for sym in self.symlist1[J] }
            for J in self.Jlist1
        }
        self.quanta_k2 = {
            J : { sym : [ (q, e) for (q, e) in self.quanta_k2[J][sym]
                          if ket(J=J, sym=sym, k=q, enr=e) ]
                  for sym in self.symlist2[J] }
            for J in self.Jlist2
        }

        self.quanta_m1 = {
            J : { sym : [ q for q in self.quanta_m1[J][sym]
                          if bra(J=J, sym=sym, m=q) ]
                  for sym in self.symlist1[J] }
            for J in self.Jlist1
        }

        self.quanta_m2 = {
            J : { sym : [ q for q in self.quanta_m2[J][sym]
                          if ket(J=J, sym=sym, m=q) ]
                  for sym in self.symlist2[J] }
            for J in self.Jlist2
        }

        # remove empty elements form dictionaries
        attrs = ( "ind_k1", "ind_k2", "ind_m1", "ind_m2", "quanta_k1",
                  "quanta_k2", "quanta_m1", "quanta_m2" )
        for attr in attrs:
            val = getattr(self, attr)
            val2 = {
                key : { key2 : val2 for key2, val2 in val.items()
                        if len(val2) > 0 }
                for key, val in val.items()
            }
            val3 = {key : val for key, val in val2.items() if len(val) > 0}
            setattr(self, attr, val3)

        # update lists of J quanta
        self.Jlist1 = sorted(
            list(set(self.ind_k1.keys()) & set(self.ind_m1.keys()))
        )
        self.Jlist2 = sorted(
            list(set(self.ind_k2.keys()) & set(self.ind_m2.keys()))
        )

        # update lists of symmetries
        self.symlist1 = {
            J : sorted(list(
                    set(self.ind_k1[J].keys()) & set(self.ind_m1[J].keys())
                ))
            for J in self.Jlist1
        }
        self.symlist2 = {
            J : sorted(list(
                    set(self.ind_k2[J].keys()) & set(self.ind_m2[J].keys())
                ))
            for J in self.Jlist2
        }

        # update state inds and assignments with selected J quanta and syms
        for attr in attrs:
            ind = int(attr[-1])
            val = getattr(self, attr)
            if ind==1:
                setattr(
                    self,
                    attr,
                    { J : {sym : val[J][sym] for sym in self.symlist1[J]}
                      for J in self.Jlist1 }
                )
            elif ind==2:
                setattr(
                    self,
                    attr,
                    { J : {sym : val[J][sym] for sym in self.symlist2[J]}
                      for J in self.Jlist2 }
                )
            else:
                raise ValueError(f"bad value for `ind` = {ind}") from None

        # dimensions

        self.dim_k1 = {
            J : {sym : len(self.ind_k1[J][sym]) for sym in self.symlist1[J]}
            for J in self.Jlist1
        }
        self.dim_k2 = {
            J : {sym : len(self.ind_k2[J][sym]) for sym in self.symlist2[J]}
            for J in self.Jlist2
        }
        self.dim_m1 = {
            J : {sym : len(self.ind_m1[J][sym]) for sym in self.symlist1[J]}
            for J in self.Jlist1
        }
        self.dim_m2 = {
            J : {sym : len(self.ind_m2[J][sym]) for sym in self.symlist2[J]}
            for J in self.Jlist2
        }

        self.dim1 = {
            J : { sym : self.dim_k1[J][sym] * self.dim_m1[J][sym]
                  for sym in self.symlist1[J] }
            for J in self.Jlist1
        }
        self.dim2 = {
            J : { sym : self.dim_k2[J][sym] * self.dim_m2[J][sym]
                  for sym in self.symlist2[J] }
            for J in self.Jlist2
        }

        # truncate M and K tensors

        Jpairs = list(set(self.mmat.keys()) & set(self.kmat.keys()))

        # delete J pairs for which M and K tensors have no overlap
        for J in set(self.mmat.keys()) - set(Jpairs):
            del self.mmat[J]
        for J in set(self.kmat.keys()) - set(Jpairs):
            del self.kmat[J]

        for (J1, J2) in Jpairs:

            if J1 not in self.Jlist1 or J2 not in self.Jlist2:
                del self.mmat[(J1, J2)]
                del self.kmat[(J1, J2)]
                continue

            mmat_J = self.mmat[(J1, J2)]
            kmat_J = self.kmat[(J1, J2)]

            sympairs = list(set(mmat_J.keys()) & set(kmat_J.keys()))

            # delete symmetry pairs for which M and K tensors have no overlap
            for sym in set(mmat_J.keys()) - set(sympairs):
                del self.mmat[(J1, J2)][sym]
            for sym in set(kmat_J.keys()) - set(sympairs):
                del self.kmat[(J1, J2)][sym]

            for (sym1, sym2) in sympairs:

                if sym1 not in self.symlist1[J1] \
                    or sym2 not in self.symlist2[J2]:
                    del self.mmat[(J1, J2)][(sym1, sym2)]
                    del self.kmat[(J1, J2)][(sym1, sym2)]
                    continue

                mmat_sym = mmat_J[(sym1, sym2)]
                kmat_sym = kmat_J[(sym1, sym2)]

                im1 = self.ind_m1[J1][sym1]
                im2 = self.ind_m2[J2][sym2]
                ik1 = self.ind_k1[J1][sym1]
                ik2 = self.ind_k2[J2][sym2]

                mmat = {
                    irrep : { cart : mat[im1, :].tocsc()[:, im2].tocsr()
                              for cart, mat in mmat_irrep.items() }
                    for irrep, mmat_irrep in mmat_sym.items()
                }

                kmat = {
                    irrep : mat[ik1, :].tocsc()[:, ik2].tocsr()
                    for irrep, mat in kmat_sym.items()
                }

                irreps = list(set(kmat.keys()) & set(mmat.keys()))
                for irrep in irreps:
                    k = kmat[irrep]
                    m = mmat[irrep]
                    if thresh is not None:
                        mask = abs(k.data) < thresh
                        k.data[mask] = 0
                        k.eliminate_zeros()
                        m_ = dict()
                        for cart, mm in m.items():
                            mask = abs(mm.data) < thresh
                            mm.data[mask] = 0
                            mm.eliminate_zeros()
                            if mm.nnz > 0:
                                m_[cart] = mm
                        m = m_
                    if k.nnz > 0 and len(m) > 0:
                        mmat[irrep] = m
                        kmat[irrep] = k
                    else:
                        del mmat[irrep]
                        del kmat[irrep]

                if len(mmat) > 0 and len(kmat) > 0:
                    self.mmat[(J1, J2)][(sym1, sym2)] = mmat
                    self.kmat[(J1, J2)][(sym1, sym2)] = kmat
                else:
                    del self.mmat[(J1, J2)][(sym1, sym2)]
                    del self.kmat[(J1, J2)][(sym1, sym2)]


    def tomat(self, form='block', repres='csr_matrix', thresh=None, cart=None):
        """ Returns full matrix representation of tensor

        Args:
            form : str
                For `form` = 'block', the matrix representation is build as
                dictionary containing matrix blocks for different pairs of J
                and symmetries. For `form` = 'full', full 2D matrix is
                constructed.
            repres : str
                Defines representation for matrix blocks or full 2D matrix. Can
                be set to the name of one of :py:class:`scipy.sparse` matrix 
                classes, e.g., `repres` = 'coo_matrix'. Alternatively it can be
                set to 'dense'.
            thresh : float
                Threshold for neglecting matrix elements when converting into 
                the sparse form.
            cart : str
                Desired Cartesian component of tensor, e.g., `cart` = 'z' or
                `cart` = 'xx'. If set to None (default), the function will
                attempt to return a matrix representation of the corresponding
                potential (i.e., product of tensor and field)

        Returns:
            nested dict or 2D array
                For `form` = 'block', returns dictionary containing matrix
                blocks for different pairs of J and symmetries. For `form` =
                'full', returns 2D matrix.
        """
        assert (form in ('block', 'full')), \
            f"`form` unknown: '{form}' (use 'block', 'full')"

        if cart is None:
            try:
                x = self.mfmat
            except AttributeError:
                raise AttributeError(
                    f"specify Cartesian component `cart` of tensor or " \
                        + f"multiply tensor with field before computing its " \
                        + f"its matrix representation"
                ) from None
        else:
            if cart not in self.cart:
                raise ValueError(
                    f"specified Cartesian component '{cart}' is not " \
                        + f"contained in tensor components '{self.cart}'"
                ) from None

        mydict = lambda: defaultdict(mydict)
        mat = mydict()

        if cart is None:

            for Jpair in list(set(self.mfmat.keys()) & set(self.kmat.keys())):

                mmat_J = self.mfmat[Jpair]
                kmat_J = self.kmat[Jpair]

                for sympair in list(set(mmat_J.keys()) & set(kmat_J.keys())):

                    mmat = mmat_J[sympair]
                    kmat = kmat_J[sympair]

                    # do M \otimes K
                    me = sum(
                        kron(mmat[irrep], kmat[irrep])
                        for irrep in list(set(mmat.keys()) & set(kmat.keys()))
                    )

                    if not np.isscalar(me):
                        mat[Jpair][sympair] = me

        else:

            for Jpair in list(set(self.mmat.keys()) & set(self.kmat.keys())):

                mmat_J = self.mmat[Jpair]
                kmat_J = self.kmat[Jpair]

                for sympair in list(set(mmat_J.keys()) & set(kmat_J.keys())):

                    mmat = {
                        irrep : val[cart]
                        for irrep,val in mmat_J[sympair].items() if cart in val
                    }
                    kmat = kmat_J[sympair]

                    # do M \otimes K
                    me = sum(
                        kron(mmat[irrep], kmat[irrep])
                        for irrep in list(set(mmat.keys()) & set(kmat.keys()))
                    )

                    if not np.isscalar(me):
                        mat[Jpair][sympair] = me

        if thresh is not None and thresh > 0:
            for Jpair, mat_J in mat.items():
                for sympair, mat_sym in mat_J.items():
                    mat_sym = mat_sym.tocoo()
                    mask = np.argwhere(abs(mat_sym.data) > thresh).flatten()
                    mat_sym = csr_matrix(
                        ( mat_sym.data[mask],
                            (mat_sym.row[mask], mat_sym.col[mask])),
                        shape = mat_sym.shape
                    )
                    mat[Jpair][sympair] = mat_sym

        if form == 'block':
            for Jpair, mat_J in mat.items():
                for sympair, mat_sym in mat_J.items():
                    if repres == 'dense':
                        mat_sym = mat_sym.toarray()
                    else:
                        mat_sym = getattr(scipy.sparse, repres)(mat_sym)
                    mat[Jpair][sympair] = mat_sym

        elif form == 'full':
            mat = self.full_form(mat, repres, thresh)

        return mat


    def assign(self, form='block'):
        """ Returns assignments of bra and ket basis states

        Args:
            form : str
                Form of the assignment output (see `form` argument of
                :py:func:`CarTens.tomat` function)

        Returns:
            assign1, assign2 : dict
                Assignments of bra and ket states, respectively. For `form` =
                'block', `assign[J][sym]['m']` and `assign[J][sym]['k']`
                contain list of m and ro-vibrational quantum numbers,
                respectively, for states with given values of J quantum number
                and symmetry sym. For `form` = 'full', `assign['m']`,
                `assign['k']`, `assign['sym']`, and `assign['J']` contain list
                of m quanta, ro-vibrational quanta, symmetries, and J values
                for all states in the basis. The ordering of elements in
                `assign1` and `assign2` lists corresponds to the ordering of
                rows and columns in a matrix returned by
                :py:func:`CarTens.tomat` function.
        """
        assert (form in ('block', 'full')), \
            f"bad value of argument 'form' = '{form}' (use 'block' or 'full')"

        m1 = {
            J : {sym : self.quanta_m1[J][sym] for sym in self.symlist1[J]}
            for J in self.Jlist1
        }
        m2 = {
            J : {sym : self.quanta_m2[J][sym] for sym in self.symlist2[J]}
            for J in self.Jlist2
        }
        k1 = {
            J : {sym : self.quanta_k1[J][sym] for sym in self.symlist1[J]}
            for J in self.Jlist1
        }
        k2 = {
            J : {sym : self.quanta_k2[J][sym] for sym in self.symlist2[J]}
            for J in self.Jlist2
        }

        mydict = lambda: defaultdict(mydict)

        if form == 'block':

            assign1 = mydict()
            for J in self.Jlist1:
                for sym in self.symlist1[J]:
                    mk = [ elem for elem \
                           in itertools.product(m1[J][sym], k1[J][sym]) ]
                    assign1[J][sym]['m'] = [elem[0] for elem in mk]
                    assign1[J][sym]['k'] = [elem[1] for elem in mk]

            assign2 = mydict()
            for J in self.Jlist2:
                for sym in self.symlist2[J]:
                    mk = [ elem for elem \
                           in itertools.product(m2[J][sym], k2[J][sym]) ]
                    assign2[J][sym]['m'] = [elem[0] for elem in mk]
                    assign2[J][sym]['k'] = [elem[1] for elem in mk]

        else:

            assign1 = {'m':[], 'k':[], 'sym':[], 'J':[]}
            for J in self.Jlist1:
                for sym in self.symlist1[J]:
                    mk = [ elem for elem \
                           in itertools.product(m1[J][sym], k1[J][sym]) ]
                    assign1['m'] += [elem[0] for elem in mk]
                    assign1['k'] += [elem[1] for elem in mk]
                    assign1['sym'] += [sym for elem in mk]
                    assign1['J'] += [J for elem in mk]

            assign2 = {'m':[], 'k':[], 'sym':[], 'J':[]}
            for J in self.Jlist2:
                for sym in self.symlist2[J]:
                    mk = [ elem for elem \
                           in itertools.product(m2[J][sym], k2[J][sym]) ]
                    assign2['m'] += [elem[0] for elem in mk]
                    assign2['k'] += [elem[1] for elem in mk]
                    assign2['sym'] += [sym for elem in mk]
                    assign2['J'] += [J for elem in mk]

        return assign1, assign2


    def full_form(self, mat, repres='csr_matrix', thresh=None):
        """ Converts block representation of tensor matrix into 2D matrix form

        Args:
            mat : nested dict
                Block representation of matrix for different values of bra and
                ket J quanta and different symmetries
            repres : str
                Set to the name of one of :py:class:`scipy.sparse` matrix
                classes to output the matrix in sparse format, e.g., `sparse` =
                'coo_matrix' or 'csr_matrix'. Alternatively set to 'dense'.
            thresh : float
                Threshold for neglecting matrix elements when converting into
                sparse form.

        Returns:
            array
                2D matrix representation.
        """
        res = scipy.sparse.bmat(
            [ [ mat[(J1, J2)][(sym1, sym2)]
                if (J1, J2) in mat.keys() \
                    and (sym1, sym2) in mat[(J1, J2)].keys()
                else csr_matrix(
                    np.zeros((self.dim1[J1][sym1], self.dim2[J2][sym2]))
                )
                for J2 in self.Jlist2 for sym2 in self.symlist2[J2] ]
                for J1 in self.Jlist1 for sym1 in self.symlist1[J1] ]
        )
        if repres == 'dense':
            res = res.toarray()
        else:
            res = getattr(scipy.sparse, repres)(res)
        return res


    def block_form(self, mat):
        """ Converts 2D tensor matrix into a block form

        Args:
            mat : array (numpy.ndarray or scipy.sparse.spmatrix)
                2D tensor matrix.

        Returns:
            nested dict
                Block form of tensor matrix, split for different values of bra
                and ket J quanta and different symmetries.
        """
        ind0 = np.cumsum(
            [ self.dim1[J][sym] for J in self.Jlist1
              for sym in self.symlist1[J] ]
        )
        ind1 = np.cumsum(
            [ self.dim2[J][sym] for J in self.Jlist2
              for sym in self.symlist2[J] ]
        )
        try:
            # input matrix is scipy sparse matrix
            mat_ = [ np.split(mat2, ind1, axis=1)[:-1]
                     for mat2 in np.split(mat.toarray(), ind0, axis=0)[:-1] ]
        except AttributeError:
            # input matrix is ndarray
            mat_ = [ np.split(mat2, ind1, axis=1)[:-1]
                     for mat2 in np.split(mat, ind0, axis=0)[:-1] ]

        Jsym1 = [(J, sym) for J in self.Jlist1 for sym in self.symlist1[J]]
        Jsym2 = [(J, sym) for J in self.Jlist2 for sym in self.symlist2[J]]

        res = dict()
        for i,(J1,sym1) in enumerate(Jsym1):
            for j,(J2,sym2) in enumerate(Jsym2):
                m = mat_[i][j]
                if issparse(mat):
                    m = getattr(scipy.sparse, mat.getformat()+"_matrix")(m)
                    m.eliminate_zeros()
                    nnz = m.nnz
                else:
                    nnz = sum(np.abs(m) > 0)
                if nnz == 0:
                    continue
                try:
                    res[(J1, J2)][(sym1, sym2)] = m
                except KeyError:
                    res[(J1, J2)] = {(sym1, sym2) : m}
        return res


    def map(self, tens, vbra=None, vket=None):
        """ Maps elements of bra and ket vectors `vbra` and `vket`, defined
                with respect to bra and ket basis sets of tensor operator
                `tens`, onto a larger basis set of `self` operator.

        The bra and ket basis sets of `tens` operator must be fully contained
            in `self`.

        Args:
            tens : :py:class:`CarTens`
                Tensor operator.
            vbra : nested dict, numpy.ndarray, or scipy.sparse.spmatrix
                Vector, defined with respect to the bra basis of `tens`,
                represented by a numpy.ndarray or scipy.sparse.spmatrix array,
                or in block form as dictionary for different J quanta and
                different symmetries.
            vket : nested dict, numpy.ndarray, or scipy.sparse.spmatrix
                Vector, defined with respect to the ket basis of `tens`.

        Returns:
            (vec1, vec2) : (type(vbra), type(vket)
                Input vectors mapped onto bra and ket basis sets of `self`.
        """
        def vec_map(vec, quanta_m, quanta_k, quanta_m_, quanta_k_):
            """Maps elements of vector `vec`, defined wrt to quanta `quanta_m`
                   x `quanta_k`, into a new vector wrt larger space of quanta
                   `quanta_m_` x `quanta_k_`. Input vector must be in block
                   form.
            """
            vec_ = dict()
            for J in vec.keys():
                for sym in vec[J].keys():
                    m = quanta_m[J][sym]
                    k = quanta_k[J][sym]
                    mk = [elem for elem in itertools.product(m, k)]
                    if len(list(set(mk))) != len(mk):
                        raise ValueError(
                            f"found states with same assignment for J, sym " \
                                + f" = {J, sym} in `tens`"
                        ) from None
                    try:
                        m_ = quanta_m_[J][sym]
                        k_ = quanta_k_[J][sym]
                        mk_ = [elem for elem in itertools.product(m_, k_)]
                        if len(list(set(mk_))) != len(mk_):
                            raise ValueError(
                                f"found states with same assignment for J, " \
                                    + f"sym = {J, sym} in `self`"
                            ) from None
                    except KeyError:
                        raise KeyError(
                            f"`tens` is not fully contained in `self`"
                        ) from None
                    # slow and simple search
                    ind = []
                    for elem in mk:
                        try:
                            ind.append(mk_.index(elem))
                        except ValueError:
                            raise ValueError(
                                f"`tens` basis state J, sym, m, k = " \
                                    + f"{J, sym, elem[0], elem[1]} is not " \
                                    + f"found in 'self'"
                            ) from None
                    # update vector
                    try:
                        vec_[J][sym] = np.zeros(len(mk_), dtype=np.float64)
                    except KeyError:
                        vec_[J] = {sym : np.zeros(len(mk_), dtype=np.float64)}
                    vec_[J][sym][ind] = vec[J][sym]


        def vec_to_block(vec, vec_name, Jlist, symlist, dim):
            """ Converts vector into block form """
            if isinstance(vec, (list, tuple, np.ndarray, np.generic)) \
                or issparse(vec):
                # convert `vec` into block form
                ind = np.cumsum(
                    [dim[J][sym] for J in Jlist for sym in symlist[J]]
                )
                try:
                    # scipy sparse
                    vec_ = [np.split(vec.toarray(), ind)[:-1]]
                except AttributeError:
                    # ndarray
                    vec_ = [np.split(vec, ind)[:-1]]
                Jsym = [(J, sym) for J in Jlist for sym in symlist[J]]
                res = dict()
                for i, (J, sym) in enumerate(Jsym):
                    try:
                        res[J][sym] = vec_[i]
                    except KeyError:
                        res[J] = {sym : vec_[i]}
            elif isinstance(vec, Mapping):
                # check if `vec` is fully contained in `tens`
                if not all([J in Jlist for J in vec.keys()]):
                    raise ValueError(
                        f"some of J quanta in the input vector '{vec_name}' " \
                            + f"are not contained in the input tensor `tens`"
                    ) from None
                if not all(
                    [ sym in symlist[J] for J in vec.keys() \
                      for sym in vec[J].keys ]
                ):
                    raise ValueError(
                        f"some of symmetries in the input vector " \
                            + f"'{vec_name}' are not contained in the input " \
                            + f"tensor `tens`"
                    ) from None
                res = vec
            else:
                raise ValueError(
                    f"bad argument type for '{vec_name}': '{type(vec)}'"
                ) from None
            return res


        def vec_to_inpform(vec_inp, vec_name, vec_out, Jlist, symlist, dim):
            """ Converts output vector to the same form as input vector """
            if isinstance(vec_inp, (list, tuple, np.ndarray, np.generic)) \
                or issparse(vec_inp):
                res = scipy.sparse.bmat(
                    [ vec_out[J][sym]
                      if J in vec_out.keys() and sym in vec_out[J].keys()
                      else csr_matrix(np.zeros(dim[J][sym]))
                      for J in Jlist for sym in symlist[J] ]
                )
                try:
                    # scipy sparse
                    res = getattr(
                        scipy.sparse, vec_inp.getformat()+"_matrix"
                    )(res)
                except AttributeError:
                    # ndarray
                    res = res.toarray()
            elif isinstance(vec_inp, Mapping):
                res = vec_out
            else:
                raise ValueError(
                    f"bad argument type for '{vec_name}': '{type(vec)}'"
                ) from None
            return res


        if not isinstance(arg, CarTens):
            raise TypeError(
                f"bad argument type for 'tens': '{type(tens)}'"
            ) from None

        res = []

        if vbra is not None:
            vec = vec_to_block(
                vbra, 'vbra', tens.Jlist1, tens.symlist1, tens.dim1
            )
            vec = vec_map(
                vec,
                tens.quanta_m1,
                tens.quanta_k1,
                self.quanta_m1,
                self.quanta_k1
            )
            vec = vec_to_inpform(
                vbra, 'vbra', vec, self.Jlist1, self.symlist1, self.dim1
            )
            res.append(vec)

        if vket is not None:
            vec = vec_to_block(
                vket, 'vket', tens.Jlist2, tens.symlist2, tens.dim2
            )
            vec = vec_map(
                vec,
                tens.quanta_m2,
                tens.quanta_k2,
                self.quanta_m2,
                self.quanta_k2
            )
            vec = vec_to_inpform(
                vket, 'vket', vec, self.Jlist2, self.symlist2, self.dim2
            )
            res.append(vec)

        return tuple(res)


    def mul(self, arg):
        """ In-place multiplication of tensor with a scalar `arg` """
        scalar = ( int, float, complex, np.int, np.int8, np.int16, np.int32,
                   np.int64, np.float, np.float16, np.float32, np.float64,
                   np.complex64, np.complex128 )
        if isinstance(arg, scalar):
            # multiply K-tensor with a scalar
            for Jpair in self.kmat.keys():
                for sympair in self.kmat[Jpair].keys():
                    self.kmat[Jpair][sympair] = { 
                        key : val * arg
                        for key, val in self.kmat[Jpair][sympair].items()
                    }
        else:
            raise TypeError(
                f"bad argument type for `arg` : '{type(arg)}'"
            ) from None


    def add_cartens(self, arg):
        """ Adds two tensors together

        Args:
            arg : :py:class:`CarTens`
                Tensor operator, must be defined with respect to the same basis
                as `self`.

        Returns:
            :py:class:`CarTens`
                Sum of `self` and `arg` tensor operators. Please note that
                functionality of the returned class is limited to class's
                matrix-vector operations. Some of the attributes are missing,
                such as, for example, `mmat`, `cart`, `os`, `rank`.
        """
        if not isinstance(arg, CarTens):
            raise TypeError(
                f"bad argument type for `arg`: '{type(arg)}'"
            ) from None

        # check  if two tensors are defined with respect to the same basis set

        # check_dict = ( "Jlist1", "Jlist2", "symlist1", "symlist2", "dim1",
        #                "dim2", "dim_k1", "dim_k2", "dim_m1", "dim_m2",
        #                "quanta_k1", "quanta_k2", "quanta_m1", "quanta_m2" )
        check_dict = ( "Jlist1", "Jlist2", "symlist1", "symlist2", "dim1",
                       "dim2", "dim_k1", "dim_k2", "dim_m1", "dim_m2" )
        nelem = 0
        for elem in check_dict:
            attr1 = getattr(self, elem)
            attr2 = getattr(arg, elem)
            if attr1 != attr2:
                print(
                    f"\ncan't add two tensors: {retrieve_name(self)}.{elem} " \
                        + f"!= {retrieve_name(arg)}.{elem}"
                )
                nelem+=1
        if nelem>0:
            raise ValueError(
                f"tensors `{retrieve_name(self)}` + `{retrieve_name(arg)}` " \
                    + f"defined with respect to different basis sets"
            ) from None

        # generate mfmat for tensors without Cartesian components, e.g. H0
        try:
            if self.cart[0] == "0":
                self.field([0, 0, 1])
        except AttributeError:
            pass
        try:
            if arg.cart[0] == "0":
                arg.field([0, 0, 1]) 
        except AttributeError:
            pass

        # initialize output object
        res = CarTens()
        res.__name__ = retrieve_name(self) + "+" + retrieve_name(arg)
        res.__dict__.update(self.__dict__)

        # delete attributes that may lead to erroneous use/behaviour
        del_attrs = ("os", "rank", "cart", "mmat")
        for attr in del_attrs:
            if hasattr(res, attr):
                delattr(res, attr)

        # sum M and K tensors

        irreps1 = [ list(self.kmat[J][sym].keys()) for J in self.kmat.keys()
                    for sym in self.kmat[J].keys()] + \
                  [ list(self.mfmat[J][sym].keys()) for J in self.mfmat.keys()
                    for sym in self.mfmat[J].keys() ]

        irreps2 = [ list(arg.kmat[J][sym].keys()) for J in arg.kmat.keys()
                    for sym in arg.kmat[J].keys() ] + \
                  [ list(arg.mfmat[J][sym].keys()) for J in arg.mfmat.keys()
                    for sym in arg.mfmat[J].keys() ]

        irreps1 = {
            elem : str(elem)+"_1" for elem in set(itertools.chain(*irreps1))
        }
        irreps2 = {
            elem : str(elem)+"_2" for elem in set(itertools.chain(*irreps2))
        }

        mydict = lambda: defaultdict(mydict)
        res.kmat = mydict()
        res.mfmat = mydict()

        for Jpair in self.kmat.keys():
            for sympair in self.kmat[Jpair].keys():
                res.kmat[Jpair][sympair] = {
                    irreps1[irrep] : val
                    for irrep, val in self.kmat[Jpair][sympair].items()
                }

        for Jpair in self.mfmat.keys():
            for sympair in self.mfmat[Jpair].keys():
                res.mfmat[Jpair][sympair] = {
                    irreps1[irrep] : val
                    for irrep, val in self.mfmat[Jpair][sympair].items()
                }

        for Jpair in arg.kmat.keys():
            for sympair in arg.kmat[Jpair].keys():
                dct = {
                    irreps2[irrep] : val
                    for irrep, val in arg.kmat[Jpair][sympair].items()
                }
                res.kmat[Jpair][sympair].update(dct)

        for Jpair in arg.mfmat.keys():
            for sympair in arg.mfmat[Jpair].keys():
                dct = {
                    irreps2[irrep] : val
                    for irrep, val in arg.mfmat[Jpair][sympair].items()
                }
                res.mfmat[Jpair][sympair].update(dct)

        return res


    def field(self, field, thresh=None):
        """ In-place multiplication of tensor with field

        The result is stored in the attribute :py:attr:`mfmat`.

        Args:
            field : array (3)
                Contains field's X, Y, Z components
             thresh : float
                Threshold for neglecting field's components and their products,
                as well as the elements of the resulting product tensor.
        """
        try:
            fx, fy, fz = field[:3]
            fxyz = np.array([fx, fy, fz])
        except (TypeError, IndexError):
            raise IndexError(
                f"field variable must be an iterable with three items " \
                    + f"which represent field's X, Y, and Z components"
            ) from None

        field_prod = {
            "".join("xyz"[c] for c in comb) : np.prod(fxyz[list(comb)])
            for comb in itertools.product((0,1,2), repeat=self.rank)
        }

        field_prod["0"] = 1 # mfmat for product of field with H0

        if thresh is not None:
            field_prod = {
                key : val
                for key, val in field_prod.items() if abs(val) >= thresh
            }

        mydict = lambda: defaultdict(mydict)
        self.mfmat = mydict()

        # return if all field product elements below threshold
        if len(field_prod) == 0:
            return

        # compute MF-tensor
        for (J1, J2), mmat_J in self.mmat.items():
            for (sym1, sym2), mmat_sym in mmat_J.items():

                res = dict()
                for irrep, mmat in mmat_sym.items():

                    # contract M-tensor with field
                    mat = []
                    for cart in list(set(mmat.keys()) & set(field_prod.keys())):
                        # if not field_prod[cart] == 0: # this is already prescreened in field_prod
                        mat.append(
                            field_prod[cart] * mmat[cart].toarray()
                        )
                    if len(mat) == 1:
                        mat = csr_matrix(mat[0])
                    else:
                        mat = csr_matrix(sum(mat))

                    # threshold
                    if thresh is not None and thresh > 0:
                        mask = abs(mat.data) < thresh
                        mat.data[mask] = 0
                        mat.eliminate_zeros()
                    if mat.nnz > 0:
                        res[irrep] = mat

                if len(res) > 0:
                    self.mfmat[(J1, J2)][(sym1, sym2)] = res


    def vec(self, vec, matvec_lib='scipy'):
        """ Computes product of tensor with vector

        Args:
            vec : nested dict
                Dictionary containing vector elements for different J and
                symmetries.

                .. code-block:: python

                    for J in vec.keys():
                        for sym in vec[J].keys():
                            print( vec[J][sym].shape == self.dim2[J][sym] ) # must be True
            matvec_lib : str
                The python library to use for SpMV. By default, this is set to
                'scipy'. Alternatively this can be set to 'numba' or 'cupy'.

        Returns:
            nested dict
                Resulting vector, has same structure as input `vec`.
        """
        try:
            x = self.mfmat
        except AttributeError:
            raise AttributeError(
                f"you need to multiply tensor with field before applying it " \
                    + f"to a vector"
            ) from None

        if not isinstance(vec, Mapping):
            raise TypeError(
                f"bad argument type for `vec`: '{type(vec)}'"
            ) from None

        assert (matvec_lib in ['scipy', 'numba', 'cupy']), \
            f"bad argument for `matvec_lib`: '{matvec_lib}' " \
                + f"(must be 'scipy', 'numba', 'cupy')"

        def matvec_func():

            if matvec_lib == 'scipy':
                return lambda spm, v : spm.dot(v)

            elif matvec_lib == 'numba':
                @njit(parallel=True)
                def matvec_numba(data, indices, indptr, dim1, v):
                    u = np.zeros((dim1, v.shape[1]), dtype=complex128)
                    for i in prange(v.shape[1]):
                        for j in prange(dim1):
                            for k in range(indptr[j], indptr[j + 1]):
                                u[j, i] += data[k] * v[indices[k], i]
                    return u
                return lambda spm, v : matvec_numba(
                    spm.data, spm.indices, spm.indptr, spm.shape[0], v
                )

            elif matvec_lib == 'cupy':
                def matvec_cupy(spm, v):
                    spm_gpu = cupyx.scipy.sparse.csr_matrix(spm)
                    u_gpu = spm_gpu.dot(cp.array(v))
                    return cp.asnumpy(u_gpu)
                return lambda spm, v : matvec_cupy(spm, v)

        matvec = matvec_func()

        vec2 = dict()

        for (J1, J2) in list(set(self.mfmat.keys()) & set(self.kmat.keys())):

            mfmat_J = self.mfmat[(J1, J2)]
            kmat_J = self.kmat[(J1, J2)]
            if not J1 in list(vec2.keys()):
                vec2[J1] = dict()

            for (sym1, sym2) in list(set(mfmat_J.keys()) & set(kmat_J.keys())):

                mfmat = mfmat_J[(sym1, sym2)]
                kmat = kmat_J[(sym1, sym2)]

                dim1 = self.dim_m2[J2][sym2]
                dim2 = self.dim_k2[J2][sym2]
                dim = self.dim1[J1][sym1]

                try:
                    vecT = np.transpose(vec[J2][sym2].reshape(dim1, dim2))
                except KeyError:
                    continue

                res = []
                for irrep in list(set(mfmat.keys()) & set(kmat.keys())):
                    tmat = matvec(kmat[irrep], vecT)
                    res.append(
                        matvec(mfmat[irrep], np.transpose(tmat)).reshape(dim)
                    )

                try:
                    vec2[J1][sym1] += sum(res)
                except KeyError:
                    vec2[J1][sym1] = sum(res)

        return vec2


    def __mul__(self, arg):
        """ Multiplication with `scalar` (:py:func:`mul`), `field`
                (:py:func:`field`) and `vector` (:py:func:`vec`)
        """
        scalar = ( int, float, complex, np.int, np.int8, np.int16, np.int32,
                   np.int64, np.float, np.float16, np.float32, np.float64,
                   np.complex64, np.complex128 )
        if isinstance(arg, scalar):
            # multiply with a scalar
            res = copy.deepcopy(self)
            res.mul(arg)
        elif isinstance(arg, (np.ndarray, list, tuple)):
            # multiply with field
            res = copy.deepcopy(self)
            res.field(arg)
        elif isinstance(arg, dict):
            # multiply with wavepacket coefficient vector
            res = copy.deepcopy(self)
            res.vec(arg)
        else:
            raise TypeError(
                f"unsupported operand type(s) for '*': " \
                    + f"'{self.__class__.__name__}' and '{type(arg)}'"
            ) from None
        return res


    def __add__(self, arg):
        """Sum with another tensor :py:class:`CarTens`"""
        if isinstance(arg, CarTens):
            res = self.add_cartens(arg)
        else:
            raise TypeError(
                f"unsupported operand type(s) for '+': " \
                    + f"'{self.__class__.__name__}' and '{type(arg)}'"
            ) from None
        return res


    def __sub__(self, arg):
        """Subtract another tensor :py:class:`CarTens`"""
        if isinstance(arg, CarTens):
            res = self.add_cartens(arg * (-1))
        else:
            raise TypeError(
                f"unsupported operand type(s) for '-': " \
                    + f"'{self.__class__.__name__}' and '{type(arg)}'"
            ) from None
        return res


    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__


    def store(self, filename, name=None, comment=None, replace=False, replace_k=False,
              replace_m=False, store_k=True, store_m=True, thresh=None):
        """ Stores object into HDF5 file
    
        Args:
            filename : str
                Name of HDF5 file.
            name : str
                Name of the data group, by default, name of the variable will be used.
            comment : str
                User comment.
            replace : bool
                If True, the existing in file complete tensor data group will
                be replaced.
            replace_k : bool
                If True, the existing in file K-tensor data sets will be 
                replaced.
            replace_m : bool
                If True, the existing in file M-tensor data sets will be
                replaced.
            store_k : bool
                If False, writing K-tensor into file will be skipped.
            store_m : bool
                If False, writing M-tensor into file will be skipped.
            thresh : float
                Threshold for neglecting matrix elements (M and K tensors) when
                writing into file.
        """
        if name is None:
            name = retrieve_name(self)

        with h5py.File(filename, 'a') as fl:
            if name in fl:
                if replace is True:
                    del fl[name]
                    group = fl.create_group(name)
                else:
                    group = fl[name]
                    #print(
                    #    f"found existing dataset '{name}' in file " \
                    #        + f"'{filename}', will append it, potentially " \
                    #        + f"replacing some data"
                    #)
            else:
                group = fl.create_group(name)

            class_name = self.class_name()
            group.attrs["__class_name__"] = class_name

            # description of object

            try:
                doc = group.attrs["__doc__"]
            except KeyError:
                doc = "Cartesian tensor operator"

            date = datetime.datetime.fromtimestamp(
                time.time()
            ).strftime('%Y-%m-%d %H:%M:%S')
            doc += ", store date: " + date.replace('\n','')
            if comment is not None:
                doc += ", comment: " \
                    + " ".join(elem for elem in comment.split())

            group.attrs["__doc__"] = doc

            # store attributes

            exclude = ["mmat", "kmat", "molecule", "basis", "eigvec",
                       "rotdens", "rotdens_kv", "rotdens_", "rotdens_kv_"]
            try:
                exclude = exclude + [elem for elem in self.store_exclude]
            except AttributeError:
                pass

            attrs = list(set(vars(self).keys()) - set(exclude))
            for attr in attrs:
                val = getattr(self, attr)
                try:
                    group.attrs[attr] = val
                except TypeError:
                    jd = json.dumps(val)
                    group.attrs[attr + "__json"] = jd

            # store basis functions (dict(Solution()))

            if hasattr(self, "basis"):
                group.attrs["basis__json"] = json.dumps(dict(self.basis))

            # store eigenvectors

            if hasattr(self, "eigvec"):
                for J1 in self.eigvec.keys():
                    for sym1 in self.eigvec[J1].keys():
                        v = self.eigvec[J1][sym1]
                        try:
                            group_j = group[J_group_key(J1, J1)]
                        except:
                            group_j = group.create_group(J_group_key(J1, J1))
                        try:
                            group_sym = group_j[sym_group_key(sym1, sym1)]
                        except:
                            group_sym = group_j.create_group(sym_group_key(sym1, sym1))
                        if "eigvec" in group_sym:
                            del group_sym["eigvec"]
                        group_sym.create_dataset("eigvec", data=v)

            # store coefficients for computing rotational density

            if hasattr(self, "rotdens"):
                for J1 in self.rotdens.keys():
                    for sym1 in self.rotdens[J1].keys():
                        try:
                            group_j = group[J_group_key(J1, J1)]
                        except:
                            group_j = group.create_group(J_group_key(J1, J1))
                        try:
                            group_sym = group_j[sym_group_key(sym1, sym1)]
                        except:
                            group_sym = group_j.create_group(sym_group_key(sym1, sym1))
                        for key in ("rotdens_data", "rotdens_indices", "rotdens_indptr",
                                    "rotdens_shape", "rotdens_kv"):
                            try:
                                del group_sym[key]
                            except KeyError:
                                pass
                        group_sym.create_dataset("rotdens_data", data=self.rotdens[J1][sym1].data)
                        group_sym.create_dataset("rotdens_indices", data=self.rotdens[J1][sym1].indices)
                        group_sym.create_dataset("rotdens_indptr", data=self.rotdens[J1][sym1].indptr)
                        group_sym.create_dataset("rotdens_shape", data=self.rotdens[J1][sym1].shape)
                        group_sym.create_dataset("rotdens_kv", data=self.rotdens_kv[J1][sym1])

            # store M and K tensors

            # loop over pairs of coupled J quanta
            for (J1, J2) in list(set(self.mmat.keys()) & set(self.kmat.keys())):

                mmat_sym = self.mmat[(J1, J2)]
                kmat_sym = self.kmat[(J1, J2)]

                # loop over pairs of coupled symmetries
                for (sym1, sym2) in list(set(mmat_sym.keys()) & set(kmat_sym.keys())):

                    # store K-matrix

                    if store_k is True:

                        kmat = kmat_sym[(sym1, sym2)]

                        # remove elements smaller than 'thresh'
                        if thresh is not None:
                            kmat_ = dict()
                            for irrep, mat in kmat.items():
                                mask = np.abs(mat.data) < thresh
                                mat.data[mask] = 0
                                mat.eliminate_zeros()
                                if mat.nnz > 0:
                                    kmat_[irrep] = mat
                            kmat = kmat_

                        data = [k.data for k in kmat.values() if k.nnz > 0]
                        indices = [k.indices for k in kmat.values() if k.nnz > 0]
                        indptr = [k.indptr for k in kmat.values() if k.nnz > 0]
                        shape = [k.shape for k in kmat.values() if k.nnz > 0]
                        irreps = [key for key,k in kmat.items() if k.nnz > 0]
                        if len(data) > 0:
                            try:
                                group_j = group[J_group_key(J1, J2)]
                            except:
                                group_j = group.create_group(J_group_key(J1, J2))
                            try:
                                group_sym = group_j[sym_group_key(sym1, sym2)]
                            except:
                                group_sym = group_j.create_group(sym_group_key(sym1, sym2))
                            if replace_k:
                                for key in ("kmat_data", "kmat_indices", "kmat_indptr"):
                                    try:
                                        del group_sym[key]
                                    except KeyError:
                                        pass
                            try:
                                group_sym.create_dataset("kmat_data", data=np.concatenate(data))
                                group_sym.create_dataset("kmat_indices", data=np.concatenate(indices))
                                group_sym.create_dataset("kmat_indptr", data=np.concatenate(indptr))
                            except:
                                raise RuntimeError(
                                    f"found existing K-tensor dataset for " \
                                        + f"`(J1, J2)` = '{(J1, J2)}' and " \
                                        + f"`(sym1, sym2)` = '{(sym1, sym2)}' " \
                                        + f" in file '{filename}', use " \
                                        + f"`replace_k` = True to replace " \
                                        + f"K-tensor datasets"
                                    ) from None
                            group_sym.attrs["kmat_nnz"] = [len(dat) for dat in data]
                            group_sym.attrs["kmat_nind"] = [len(ind) for ind in indices]
                            group_sym.attrs["kmat_nptr"] = [len(ind) for ind in indptr]
                            group_sym.attrs["kmat_irreps"] = irreps
                            group_sym.attrs["kmat_shape"] = shape

                    # store M-matrix

                    if store_m is True:

                        mmat = {
                            (irrep, cart) : m
                            for irrep, mat in mmat_sym[(sym1, sym2)].items()
                            for cart, m in mat.items()
                        }

                        # remove elements smaller than 'thresh'
                        if thresh is not None:
                            mmat_ = dict()
                            for key, mat in mmat.items():
                                mask = np.abs(mat.data) < thresh
                                mat.data[mask] = 0
                                mat.eliminate_zeros()
                                if mat.nnz > 0:
                                    mmat_[key] = mat
                            mmat = mmat_

                        data = [m.data for m in mmat.values() if m.nnz > 0]
                        indices = [m.indices for m in mmat.values() if m.nnz > 0]
                        indptr = [m.indptr for m in mmat.values() if m.nnz > 0]
                        shape = [m.shape for m in mmat.values() if m.nnz > 0]
                        irrep_cart = [key for key,m in mmat.items() if m.nnz > 0]
                        if len(data) > 0:
                            try:
                                group_j = group[J_group_key(J1, J2)]
                            except:
                                group_j = group.create_group(J_group_key(J1, J2))
                            try:
                                group_sym = group_j[sym_group_key(sym1, sym2)]
                            except:
                                group_sym = group_j.create_group(
                                    sym_group_key(sym1, sym2)
                                )
                            if replace_m:
                                for key in ("mmat_data", "mmat_indices", "mmat_indptr"):
                                    try:
                                        del group_sym[key]
                                    except KeyError:
                                        pass
                            try:
                                group_sym.create_dataset("mmat_data", data=np.concatenate(data))
                                group_sym.create_dataset("mmat_indices", data=np.concatenate(indices))
                                group_sym.create_dataset("mmat_indptr", data=np.concatenate(indptr))
                            except:
                                raise RuntimeError(
                                    f"found existing M-tensor dataset for " \
                                        + f"`(J1, J2)` = '{(J1, J2)}' and " \
                                        + f"`(sym1, sym2)` = '{(sym1, sym2)}' " \
                                        + f" in file '{filename}', use " \
                                        + f"`replace_m` = True to replace " \
                                        + f"M-tensor datasets"
                                    ) from None
                            group_sym.attrs["mmat_nnz"] = [len(dat) for dat in data]
                            group_sym.attrs["mmat_nind"] = [len(ind) for ind in indices]
                            group_sym.attrs["mmat_nptr"] = [len(ind) for ind in indptr]
                            group_sym.attrs["mmat_irrep_cart"] = np.array(irrep_cart, dtype='S')
                            group_sym.attrs["mmat_shape"] = shape


    def read(self, filename, name=None, thresh=None, **kwargs):
        """ Reads object from HDF5 file

        Args:
            filename : str
                Name of HDF5 file.
            name : str
                Name of the data group, by default the name of the variable
                will be used.
            thresh : float
                Threshold for neglecting matrix elements when reading from
                file.

        Kwargs:
            bra : function(**kw)
                State filter for bra basis sets (see `bra` in kwargs of
                :py:class:`CarTens`).
            ket : function(**kw)
                State filter for ket basis sets (see `ket` in kwargs of
                :py:class:`CarTens`).
        """
        J_key_re = re.sub(r'1.0', '\\\d+\.\\\d+', J_group_key(1, 1))
        sym_key_re = re.sub(r'A', '\\\w+', sym_group_key('A', 'A'))

        # name of HDF5 data group
        if name is None:
            name = retrieve_name(self)

        with h5py.File(filename, 'r') as fl:

            try:
                group = fl[name]
            except KeyError:
                raise KeyError(
                    f"file '{filename}' has no dataset with the name '{name}'"
                ) from None

            class_name = group.attrs["__class_name__"]
            if class_name != self.class_name():
                raise TypeError(
                    f"dataset with the name '{name}' in file '{filename}' " \
                        + f"has different type: '{class_name}'"
                ) from None

            # read attributes

            attrs = {}
            for key, val in group.attrs.items():
                if key.find('__json') == -1:
                    attrs[key] = val
                else:
                    jl = json.loads(val)
                    key = key.replace('__json', '')
                    attrs[key] = jl
            self.__dict__.update(attrs)

            # read M and K tensors
            # read eigenvectors

            mydict = lambda: defaultdict(mydict)
            self.kmat = mydict()
            self.mmat = mydict()
            self.eigvec = mydict()
            self.rotdens = mydict()
            self.rotdens_kv = mydict()

            # apply state selection filters
            self.filter(**kwargs)

            # search for J groups
            for key in group.keys():

                # pair of coupled J quanta
                if re.match(J_key_re, key):
                    Jpair = re.findall(f'\d+.\d+', key)
                    J1, J2 = (round(float(elem), 1) for elem in Jpair)
                    group_j = group[key]

                    # selected J quanta
                    if J1 not in self.Jlist1 or J2 not in self.Jlist2:
                        continue

                    # search for symmetry groups
                    for key2 in group_j.keys():

                        # pair of coupled symmetries
                        if re.match(sym_key_re, key2):
                            sympair = re.findall(f'\w+', key2)
                            _, sym1, sym2 = sympair
                            group_sym = group_j[key2]

                            # selected symmetries
                            if sym1 not in self.symlist1[J1] \
                                or sym2 not in self.symlist2[J2]:
                                continue

                            # indices of selected quanta in K and M tensors
                            ik1 = self.ind_k1[J1][sym1]
                            ik2 = self.ind_k2[J2][sym2]
                            im1 = self.ind_m1[J1][sym1]
                            im2 = self.ind_m2[J2][sym2]

                            # read eigenvectors

                            if J1 == J2 and sym1 == sym2:
                                try:
                                    data = group_sym['eigvec']
                                    self.eigvec[J1][sym1] = np.take(np.take(data, ik1, axis=0), ik2, axis=1)
                                except KeyError:
                                    pass

                            # read coefficients for rotational density estimations

                            if J1 == J2 and sym1 == sym2:
                                try:
                                    data = group_sym['rotdens_data']
                                    inds = group_sym['rotdens_indices']
                                    ptrs = group_sym['rotdens_indptr']
                                    sh = group_sym['rotdens_shape']
                                    kv = group_sym['rotdens_kv'][:, :]
                                    dens = csr_matrix((data, inds, ptrs), shape=sh)
                                    if dens.shape[-1] != len(ik1):
                                        print(f"Warning: read rotational density from file {filename} " + \
                                              f"for J = {J1} and sym = {sym1}, " + \
                                              f"the size of the basis for density = {dens.shape[-1]}, " + \
                                              f"which is different from that for the tensor = {len(ik1)}. " + \
                                              f"This is OK if tensor corresponds to hypefine solutions.")
                                        self.rotdens[J1][sym1] = dens.tocsr()
                                    else:
                                        self.rotdens[J1][sym1] = dens[:, ik1].tocsr()
                                    self.rotdens_kv[J1][sym1] = kv
                                except KeyError:
                                    pass

                            # read K-matrix

                            kmat = None
                            try:
                                nnz = group_sym.attrs['kmat_nnz']
                                nind = group_sym.attrs['kmat_nind']
                                nptr = group_sym.attrs['kmat_nptr']
                                shape = group_sym.attrs['kmat_shape']
                                irreps = group_sym.attrs['kmat_irreps']
                                data = np.split(
                                    group_sym['kmat_data'], np.cumsum(nnz)
                                )[:-1]
                                indices = np.split(
                                    group_sym['kmat_indices'], np.cumsum(nind)
                                )[:-1]
                                indptr = np.split(
                                    group_sym['kmat_indptr'], np.cumsum(nptr)
                                )[:-1]
                                kmat = {
                                    irrep : csr_matrix(
                                        (dat, ind, ptr), shape=sh
                                    )[ik1, :].tocsc()[:, ik2].tocsr()
                                    for irrep,dat,ind,ptr,sh \
                                    in zip(irreps,data,indices,indptr,shape)
                                }

                            except KeyError:
                                pass

                            # add K-matrix to tensor object
                            if kmat is not None:
                                # remove elements smaller than 'thresh'
                                if thresh is not None:
                                    kmat_ = dict()
                                    for key, mat in kmat.items():
                                        mask = np.abs(mat.data) < thresh
                                        mat.data[mask] = 0
                                        mat.eliminate_zeros()
                                        if mat.nnz > 0:
                                            kmat_[key] = mat
                                    kmat = kmat_
                                if len(kmat) > 0:
                                    self.kmat[(J1, J2)][(sym1, sym2)] = kmat

                            # read M-matrix

                            mmat = None
                            try:
                                nnz = group_sym.attrs['mmat_nnz']
                                nind = group_sym.attrs['mmat_nind']
                                nptr = group_sym.attrs['mmat_nptr']
                                shape = group_sym.attrs['mmat_shape']
                                irrep_cart = [(int(el1), el2.decode('utf-8'))
                                              for (el1, el2) in group_sym.attrs['mmat_irrep_cart']]
                                data = np.split(
                                    group_sym['mmat_data'], np.cumsum(nnz)
                                )[:-1]
                                indices = np.split(
                                    group_sym['mmat_indices'], np.cumsum(nind)
                                )[:-1]
                                indptr = np.split(
                                    group_sym['mmat_indptr'], np.cumsum(nptr)
                                )[:-1]
                                mmat = mydict()
                                for ielem, (irrep, cart) \
                                    in enumerate(irrep_cart):
                                    dat = data[ielem]
                                    ind = indices[ielem]
                                    ptr = indptr[ielem]
                                    sh = shape[ielem]
                                    mat = csr_matrix(
                                        (dat, ind, ptr), shape=sh
                                    )[im1, :].tocsc()[:, im2].tocsr()
                                    mmat[int(irrep)][cart] = mat
                            except KeyError:
                                pass

                            # add M-matrix to tensor object
                            if mmat is not None:
                                # remove elements smaller that 'thresh'
                                if thresh is not None:
                                    mmat_ = mydict()
                                    for key1, val1 in mmat.items():
                                        for key2, mat in val1.items():
                                            mask = np.abs(mat.data) < thresh
                                            mat.data[mask] = 0
                                            mat.eliminate_zeros()
                                            if mat.nnz > 0:
                                                mmat_[key1][key2] = mat
                                    mmat = mmat_
                                if len(mmat) > 0:
                                    self.mmat[(J1, J2)][(sym1, sym2)] = mmat

            # apply state selection filters again to reset the basis set inds
            self.filter(thresh=thresh, **kwargs)


    def class_name(self):
        """ Generates '__class_name__' attribute for the tensor data group in
               HDF5 file
        """
        return self.__module__ + '.' + self.__class__.__name__



def filter(obj, bra, ket, thresh=None):
    """ Applies state selection filters to input Cartesian tensor
            :py:class:`CarTens`

    Args:
        obj : :py:class:`CarTens`
            Cartesian tensor.
        bra : function(**kw)
            State filter for bra basis sets (see `bra` in kwargs of
            :py:class:`CarTens`).
        ket : function(**kw)
            State filter for ket basis sets (see `ket` in kwargs of
            :py:class:`CarTens`).

    Returns:
        :py:class:`CarTens`
            New Cartesian tensor with applied state filters.
    """
    if isinstance(obj, CarTens):
        res = copy.deepcopy(obj)
        res.filter(bra=bra, ket=ket, thresh=thresh)
    else:
        raise TypeError("bad argument type for 'obj': '{type(obj)}'") from None
    return res


def J_group_key(J1, J2):
    """ Generates HDF5 data group name for matrix elements between states with
            bra and ket J quanta equal to J1 and J2, respectively
    """
    return 'J:' + str(round(float(J1), 1)) + ',' + str(round(float(J2), 1))


def sym_group_key(sym1, sym2):
    """ Generates HDF5 data group name for matrix elements between states with
            bra and ket state symmetries equal to sym1 and sym2, respectively
    """
    return 'sym:' + str(sym1) + ',' + str(sym2)


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [ var_name for var_name, var_val in fi.frame.f_locals.items() \
                  if var_val is var ]
        if len(names) > 0:
            return names[0]
