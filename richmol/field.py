import numpy as np
import scipy.sparse
from scipy.sparse import kron, csr_matrix
import itertools
from itertools import chain
import copy
import h5py
import inspect
import datetime
import time
import regex as re
from collections import defaultdict
from richmol import json
import os


class CarTens():
    """General class for laboratory-frame Cartesian tensor operators

    Args:
        filename : str
            Name of the HDF5 file from which tensor data is loaded.
            Alternatively, one can load tensor from the old-format ascii files,
            by providing in 'filename' the name of richmol states file
            and in 'matelem' the template for generating the names of richmol
            matrix elements files.
        matelem : str
            In old-format style, matrix elements of Cartesian tensor are stored
            in separate files for different values of bra and ket J (or F) quanta.
            The argument 'filename' provides a template for generating the names
            of these files. For example, if filename = "matelem_alpha_j<j1>_j<j2>.rchm",
            the following files will be searched: "matelem_alpha_j0_j0.rchm",
            "matelem_alpha_j0_j1.rchm", "matelem_alpha_j0_j2.rchm",
            "matelem_alpha_j1_j1.rchm", etc., i.e., <j1> and <j2> are replaced
            by the integer values of bra and ket J quanta spanned by the basis
            of stationary states.
            For half-integer values of the F quantum number, replace <j1> and <j2>
            in the template by <f1> and <f2>, these will be substituted
            by the floating point values of F quanta rounded to the first decimal.

    Kwargs:
        bra, ket : function(**kw)
            State filters for bra and ket basis sets, take as arguments
            state quantum numbers and energies J, sym, m, k, and enr
            and return True or False depending on if the corresponding state
            needs to be included or excluded form the basis.
            By default, all states stored in the file are included.
            The following keyword arguments are passed into the bra and ket functions:
                J : float (round to first decimal)
                    Value of J quantum number
                sym : str
                    State symmetry
                enr : float
                    State energy
                m : str
                    State assignment in the M subspace, usually just a value
                    of the m quantum number as a string
                k : str
                    State assignment in the K subspace, which are the rotational
                    or ro-vibrational quanta joined in a string
    """

    def __init__(self, filename=None, matelem=None, **kwargs):

        if filename is None:
            pass
        else:
            if h5py.is_hdf5(filename):
                # read tensor from HDF5 file
                self.read(filename, **kwargs)
            else:
                # read old-format richmol files
                self.read_states(filename, **kwargs)
                if matelem is not None:
                    self.read_trans(matelem, **kwargs)


    def tomat(self, form='block', sparse=None, thresh=None, cart=None):
        """Returns full MxK matrix representation of tensor

        Args:
            form : str
                For form='block', the matrix representation is split into blocks
                for different values of J quanta and different symmetries,
                i.e., mat[(J1, J2)][(sym1, sym2)] -> array.
                For form='full', the matrix representation is build as a 2D matrix.
            sparse : str
                Defines sparse matrix representation, for different blocks (if form='block')
                or for full matrix (if form='full').
                Set to one of scipy.sparse sparse matrix classes, e.g.,
                sparse="coo_matrix" or "csr_matrix"
            thresh : float
                Threshold for neglecting matrix elements when converting into
                sparse form
            cart : str
                Desired Cartesian component of tensor, e.g., cart='xx'.
                If None, the function will attempt to return matrix representation
                of the corresponding potential (i.e. tensor times field)

        Returns:
            mat : nested dict or 2D ndarray
                For form='block', returns dictionary containing matrix elements
                of tensor operator for different pairs of J quanta and different
                pairs of symmetries, i.e., mat[(J1, J2)][(sym1, sym2)] -> ndarray.
                For form='full', returns 2D dense matrix, the order of blocks in full
                matrix corresponds to [(J, sym) for J in self.Jlist for sym in self.symlist[J]].
        """
        assert (form in ('block', 'full')), f"bad value of argument 'form' = '{form}' (use 'block' or 'full')"

        if cart is None:
            try:
                x = self.mfmat
            except AttributeError:
                raise AttributeError(f"you need to specify Cartesian component of tensor " + \
                    f"or multiply tensor with field before computing its matrix representation") from None
        else:
            if cart not in self.cart:
                raise ValueError(f"input Cartesian component '{cart}' " + \
                    f"is not contained in tensor components {self.cart}") from None

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
                    me = np.sum( kron(mmat[irrep], kmat[irrep]).todense()
                                 for irrep in list(set(mmat.keys()) & set(kmat.keys())) )

                    mat[Jpair][sympair] = me

        else:

            for Jpair in list(set(self.mmat.keys()) & set(self.kmat.keys())):

                mmat_J = self.mmat[Jpair]
                kmat_J = self.kmat[Jpair]

                for sympair in list(set(mmat_J.keys()) & set(kmat_J.keys())):

                    try:
                        mmat = mmat_J[sympair][cart]
                    except KeyError:
                        continue   # all MEs are zero for current J and symmetry pairs

                    # K tensor
                    kmat = kmat_J[sympair]

                    # do M \otimes K
                    me = np.sum( kron(mmat[irrep], kmat[irrep]).todense()
                                 for irrep in list(set(mmat.keys()) & set(kmat.keys())) )

                    # add to matrix if non zero
                    # if not np.isscalar(me):
                    mat[Jpair][sympair] = me

        if form == 'block':
            if sparse is not None:
                for Jpair, mat_J in mat.items():
                    for sympair, mat_sym in mat_J.items():
                        mat_sym = getattr(scipy.sparse, sparse)(mat_sym)
                        if thresh is not None:
                            mask = np.abs(mat_sym.data) < thresh
                            mat_sym[mask] = 0
                            mat_sym.eliminate_zeros()
                        mat[Jpair][sympair] = mat_sym
        elif form == 'full':
            mat = self.full_form(mat, sparse, thresh)

        return mat


    def assign(self, form='block'):
        """Returns assignments of bra and ket basis states

        Args:
            form : str
                Form of the assignment output

        Returns:
            assign1, assign2 : dict
                If form == 'block', assign[J][sym]['m'] and assign[J][sym]['k']
                each contain list of quantum numbers describing M and K subspaces,
                repspectively, for all states with given values of J quantum
                number and symmetry sym.
                For other values of form parameter, assign['m'], assign['k'],
                assign['sym'], and assign['J'] contain list of quantum numbers
                describing M and K subspaces, list of symmetries, and list of values
                of J quantum number, respectively, for all sates in the basis
        """
        assert (form in ('block', 'full')), f"bad value of argument 'form' = '{form}' (use 'block' or 'full')"

        m1 = { J : { sym : self.assign_m1[J][sym] for sym in self.symlist1[J] } for J in self.Jlist1 }
        m2 = { J : { sym : self.assign_m2[J][sym] for sym in self.symlist2[J] } for J in self.Jlist2 }
        k1 = { J : { sym : self.assign_k1[J][sym] for sym in self.symlist1[J] } for J in self.Jlist1 }
        k2 = { J : { sym : self.assign_k2[J][sym] for sym in self.symlist2[J] } for J in self.Jlist2 }

        if form == 'block':

            assign1 = {J:{sym:{} for sym in self.symlist1[J]} for J in self.Jlist1}
            for J in self.Jlist1:
                for sym in self.symlist1[J]:
                    mk = [elem for elem in itertools.product(m1[J][sym], k1[J][sym])]
                    assign1[J][sym]['m'] = [elem[0] for elem in mk]
                    assign1[J][sym]['k'] = [elem[1] for elem in mk]

            assign2 = {J:{sym:{} for sym in self.symlist2[J]} for J in self.Jlist2}
            for J in self.Jlist2:
                for sym in self.symlist2[J]:
                    mk = [elem for elem in itertools.product(m2[J][sym], k2[J][sym])]
                    assign2[J][sym]['m'] = [elem[0] for elem in mk]
                    assign2[J][sym]['k'] = [elem[1] for elem in mk]

        else:

            assign1 = {'m':[], 'k':[], 'sym':[], 'J':[]}
            for J in self.Jlist1:
                for sym in self.symlist1[J]:
                    mk = [elem for elem in itertools.product(m1[J][sym], k1[J][sym])]
                    assign1['m'] += [elem[0] for elem in mk]
                    assign1['k'] += [elem[1] for elem in mk]
                    assign1['sym'] += [sym for elem in mk]
                    assign1['J'] += [J for elem in mk]

            assign2 = {'m':[], 'k':[], 'sym':[], 'J':[]}
            for J in self.Jlist2:
                for sym in self.symlist2[J]:
                    mk = [elem for elem in itertools.product(m2[J][sym], k2[J][sym])]
                    assign2['m'] += [elem[0] for elem in mk]
                    assign2['k'] += [elem[1] for elem in mk]
                    assign2['sym'] += [sym for elem in mk]
                    assign2['J'] += [J for elem in mk]

        return assign1, assign2


    def full_form(self, mat, sparse=None, thresh=None):
        """Converts block representation of tensor matrix into a full matrix form

        Args:
            mat : nested dict
                Block representation of matrix for different values of bra and
                ket J quanta and different symmetries, i.e. mat[(J1, J2)][(sym1, sym2)] -> array
            sparse : str
                Set to one of scipy.sparse sparse matrix classes to output matrix
                in sparse format, e.g., sparse="coo_matrix" or "csr_matrix"
            thresh : float
                Threshold for neglecting matrix elements when converting into
                sparse form

        Returns:
            res : matrix
                Matrix representation in full form as numpy ndarray or scipy sparse matrix
        """
        res = np.block([[ mat[(J1, J2)][(sym1, sym2)]
                          if (J1, J2) in mat.keys() and (sym1, sym2) in mat[(J1, J2)].keys()
                          else np.zeros((self.dim1[J1][sym1], self.dim2[J2][sym2])) \
                          for J2 in self.Jlist2 for sym2 in self.symlist2[J2] ]
                          for J1 in self.Jlist1 for sym1 in self.symlist1[J1] ])
        if sparse is not None:
            res = getattr(scipy.sparse, sparse)(res)
            if thresh is not None:
                mask = np.abs(res.data) < thresh
                res[mask] = 0
                res.eliminate_zeros()
        return res


    def block_form(self, mat):
        """Converts full matrix representation of tensor into a block form"""
        ind0 = np.cumsum([self.dim1[J][sym] for J in self.Jlist1 for sym in self.symlist1[J]])
        ind1 = np.cumsum([self.dim2[J][sym] for J in self.Jlist2 for sym in self.symlist2[J]])
        try:
            # input matrix is scipy sparse matrix
            mat_ = [np.split(mat2, ind1, axis=1)[:-1] for mat2 in np.split(mat.toarray(), ind0, axis=0)[:-1]]
        except AttributeError:
            # input matrix is ndarray
            mat_ = [np.split(mat2, ind1, axis=1)[:-1] for mat2 in np.split(mat, ind0, axis=0)[:-1]]


        Jsym1 = [(J, sym) for J in self.Jlist1 for sym in self.symlist1[J]]
        Jsym2 = [(J, sym) for J in self.Jlist2 for sym in self.symlist2[J]]

        res = dict()
        for i,(J1,sym1) in enumerate(Jsym1):
            for j,(J2,sym2) in enumerate(Jsym2):
                try:
                    res[(J1, J2)][(sym1, sym2)] = mat_[i][j]
                except KeyError:
                    res[(J1, J2)] = {(sym1, sym2) : mat_[i][j]}
        return res


    def mul(self, arg):
        """Multiplies tensor with a scalar"""
        scalar = (int, float, complex, np.int, np.int8, np.int16, np.int32, np.int64, np.float, \
                  np.float16, np.float32, np.float64, np.complex64, np.complex128)
        if isinstance(arg, scalar):
            # multiply K-tensor with a scalar
            for Jpair in self.kmat.keys():
                for sympair in self.kmat[Jpair].keys():
                    self.kmat[Jpair][sympair] = {key : val * arg for key, val in self.kmat[Jpair][sympair].items()}
        else:
            raise TypeError(f"bad argument type for 'arg': '{type(arg)}'") from None


    def add_cartens(self, arg):
        """Adds two tensors together

        Args:
            arg : CarTens
                Tensor operator, must be defined with respect to the same basis as self.

        Returns:
            res : CarTens
                Sum of self and arg tensor operators. Please note that functionality of the returned
                class is limited to class's matrix-vector operations. Some of the attributes are
                missing, such as, for example, mmat.
        """
        if not isinstance(arg, CarTens):
            raise TypeError(f"bad argument type for 'arg': '{type(arg)}'") from None

        # check  if two tensors are defined with respect to the same basis set

        check_dict = ("Jlist1", "Jlist2", "symlist1", "symlist2", "dim1", "dim2", "dim_k1", "dim_k2",
                "dim_m1", "dim_m2", "quanta_k1", "quanta_k2", "quanta_m1", "quanta_m2")

        nelem = 0
        for elem in check_dict:
            attr1 = getattr(self, elem)
            attr2 = getattr(arg, elem)
            if attr1 != attr2:
                print(f"can't add two tensors: {retrieve_name(self)}.{elem} != {retrieve_name(arg)}.{elem}")
                nelem+=1
        if nelem>0:
            raise ValueError(f"tensors {retrieve_name(self)} + {retrieve_name(arg)} defined with respect to different basis sets") from None

        # sum M and K tensors

        irreps1 = {elem : str(elem)+"_1" for elem in set(omega for (omega,sigma) in self.os)}
        irreps2 = {elem : str(elem)+"_2" for elem in set(omega for (omega,sigma) in arg.os)}

        res = CarTens()
        res.__name__ = retrieve_name(self) + "+" + retrieve_name(arg)
        res.__dict__.update(self.__dict__)

        mydict = lambda: defaultdict(mydict)
        res.kmat = mydict()
        res.mfmat = mydict()

        for Jpair in self.kmat.keys():
            for sympair in self.kmat[Jpair].keys():
                res.kmat[Jpair][sympair] = {irreps1[irrep] : val for irrep, val in self.kmat[Jpair][sympair].items()}

        for Jpair in self.mfmat.keys():
            for sympair in self.mfmat[Jpair].keys():
                res.mfmat[Jpair][sympair] = {irreps1[irrep] : val for irrep, val in self.mfmat[Jpair][sympair].items()}

        for Jpair in arg.kmat.keys():
            for sympair in arg.kmat[Jpair].keys():
                dct = {irreps2[irrep] : val for irrep, val in arg.kmat[Jpair][sympair].items()}
                res.kmat[Jpair][sympair].update(dct)

        for Jpair in arg.mfmat.keys():
            for sympair in arg.mfmat[Jpair].keys():
                dct = {irreps2[irrep] : val for irrep, val in arg.mfmat[Jpair][sympair].items()}
                res.mfmat[Jpair][sympair].update(dct)

        return res


    def field(self, field, tol=1e-12):
        """Multiplies tensor with field and sums over Cartesian components

        The result is stored in the attribute self.mfmat, which has the same structure
        as self.kmat, i.e., self.mfmat[(J1, J2)][(sym1, sym2)][irrep]

        Args:
            field : array (3)
                Contains field's X, Y, Z components
            tol : float
                Threshold for considering elements of field or elements of M-tensor
                contracted with field as zero
        """
        try:
            fx, fy, fz = field[:3]
            fxyz = np.array([fx, fy, fz])
        except (TypeError, IndexError):
            raise IndexError(f"field variable must be an iterable with three items, " + \
                f"which represent field's X, Y, and Z components") from None

        # screen out small field components
        field_prod = {"".join("xyz"[c] for c in comb) : np.prod(fxyz[list(comb)]) \
                      for comb in itertools.product((0,1,2), repeat=self.rank) \
                      if abs(np.prod(fxyz[list(comb)])) >= tol}

        field_prod["0"] = 1 # this is to compute mfmat for product of field with field-free part (H0)

        nirrep = len(set(omega for (omega,sigma) in self.os)) # number of tensor irreps
        res = np.zeros(nirrep, dtype=np.complex128)

        # contract M-tensor with field

        self.mfmat = dict()

        for (J1, J2), mmat_J in self.mmat.items():
            for (sym1, sym2), mmat_sym in mmat_J.items():

                for cart in list(set(mmat_sym.keys()) & set(field_prod.keys())):
                    fprod = field_prod[cart]
                    mmat_irrep = mmat_sym[cart]

                    res = dict()
                    for irrep, mmat in mmat_irrep.items():
                        try:
                            res[irrep] = res[irrep] + mmat * fprod
                        except KeyError:
                            res[irrep] = mmat * fprod

                if len(res) > 0:
                    try:
                        self.mfmat[(J1, J2)][(sym1, sym2)] = res
                    except KeyError:
                        self.mfmat[(J1, J2)] = {(sym1, sym2) : res}


    def vec(self, vec):
        """Computes product of tensor with vector

        Args:
            vec : nested dict
                Dictionary containing vector elements for different values  of J
                quantum number and different symmetries, i.e., vec[J][sym] -> ndarray

        Returns:
            vec2 : nested dict
                Resulting vector, has same structure as input vec
        """
        try:
            x = self.mfmat
        except AttributeError:
            raise AttributeError("you need to multiply tensor with field before applying it to a vector") from None

        if not isinstance(vec, dict):
            raise TypeError(f"bad argument type for 'vec': '{type(vec)}'") from None

        nirrep = len(set(omega for (omega,sigma) in self.os)) # number of tensor irreps
        res = np.zeros(nirrep, dtype=np.complex128)

        vec2 = dict()

        for (J1, J2) in list(set(self.mfmat.keys()) & set(self.kmat.keys())):

            mfmat_J = self.mfmat[(J1, J2)]
            kmat_J = self.kmat[(J1, J2)]
            vec2[J1] = dict()

            for (sym1, sym2) in list(set(mfmat_sym.keys()) & set(kmat_sym.keys())):

                mfmat = mfmat_J[(sym1, sym2)]
                kmat = kmat_J[(sym1, sym2)]

                dim1 = self.dim_m2[J2][sym2]
                dim2 = self.dim_k2[J2][sym2]
                dim = self.dim1[J1][sym1]

                try:
                    vecT = np.transpose(vec[J2][sym2].reshape(dim1, dim2))
                except KeyError:
                    continue

                res[:] = 0
                for irrep in list(set(mfmat.keys()) & set(kmat.keys())):
                    tmat = csr_matrix.dot(kmat[irrep], vecT)
                    v = csr_matrix.dot(mfmat[irrep], np.transpose(tmat))
                    res[irrep] = csr_matrix.dot(mfmat[irrep], np.transpose(tmat)).reshape(dim)

                try:
                    vec2[J1][sym1] += np.sum(res)
                except KeyError:
                    vec2[J1] = {sym1 : np.sum(res)}

        return vec2


    def __mul__(self, arg):
        scalar = (int, float, complex, np.int, np.int8, np.int16, np.int32, np.int64, np.float, \
                  np.float16, np.float32, np.float64, np.complex64, np.complex128)
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
            raise TypeError(f"unsupported operand type(s) for '*': '{self.__class__.__name__}' and " + \
                f"'{type(arg)}'") from None
        return res


    def __add__(self, arg):
        if isinstance(arg, CarTens):
            res = self.add_cartens(arg)
        else:
            raise TypeError(f"unsupported operand type(s) for '+': '{self.__class__.__name__}' and " + \
                f"'{type(arg)}'") from None
        return res


    __rmul__ = __mul__
    __radd__ = __add__


    def store(self, filename, name=None, comment=None, replace=False, thresh=None):
        """Stores object into HDF5 file
    
        Args:
            filename : str
                Name of HDF5 file
            name : str
                Name of the data group, by default name of the variable is used
            comment : str
                User comment
            replace : bool
                If True, the existing data set will be replaced
            thresh : float
                Threshold for neglecting matrix elements when writing into file
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
                    #print(f"found existing dataset '{name}' in file '{filename}', " + \
                    #    f"will append it, potentially replacing some data")
            else:
                group = fl.create_group(name)

            class_name = self.class_name()
            group.attrs["__class_name__"] = class_name

            # description of object

            try:
                doc = group.attrs["__doc__"]
            except KeyError:
                doc = "Cartesian tensor operator"

            date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            doc += ", store date: " + date.replace('\n','')
            if comment is not None:
                doc += ", comment: " + " ".join(elem for elem in comment.split())

            group.attrs["__doc__"] = doc

            # store attributes

            exclude = ["mmat", "kmat", "molecule", "basis"]
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

            # store M and K tensors

            # loop over pairs of coupled J quanta
            for (J1, J2) in list(set(self.mmat.keys()) & set(self.kmat.keys())):

                mmat_sym = self.mmat[(J1, J2)]
                kmat_sym = self.kmat[(J1, J2)]

                # loop over pairs of coupled symmetries
                for (sym1, sym2) in list(set(mmat_sym.keys()) & set(kmat_sym.keys())):

                    # store K-matrix

                    # remove elements smaller than 'thresh'
                    kmat = kmat_sym[(sym1, sym2)]
                    kmat_ = [k for k in kmat.values()]
                    if thresh is not None:
                        for k in kmat_:
                            mask = np.abs(k.data) < thresh
                            k.data[mask] = 0
                            k.eliminate_zeros()

                    data = [k.data for k in kmat_ if k.nnz > 0]
                    indices = [k.indices for k in kmat_ if k.nnz > 0]
                    indptr = [k.indptr for k in kmat_ if k.nnz > 0]
                    shape = [k.shape for k in kmat_ if k.nnz > 0]
                    irreps = [key for key,k in zip(kmat.keys(), kmat_) if k.nnz > 0]
                    if len(data) > 0:
                        try:
                            group_j = group[J_group_key(J1, J2)]
                        except:
                            group_j = group.create_group(J_group_key(J1, J2))
                        try:
                            group_sym = group_j[sym_group_key(sym1, sym2)]
                        except:
                            group_sym = group_j.create_group(sym_group_key(sym1, sym2))
                        group_sym.create_dataset("kmat_data", data=np.concatenate(data))
                        group_sym.create_dataset("kmat_indices", data=np.concatenate(indices))
                        group_sym.create_dataset("kmat_indptr", data=np.concatenate(indptr))
                        group_sym.attrs["kmat_nnz"] = [len(dat) for dat in data]
                        group_sym.attrs["kmat_nind"] = [len(ind) for ind in indices]
                        group_sym.attrs["kmat_nptr"] = [len(ind) for ind in indptr]
                        group_sym.attrs["kmat_irreps"] = irreps
                        group_sym.attrs["kmat_shape"] = shape

                    # store M-matrix

                    # remove elements smaller than 'thresh'
                    mmat = mmat_sym[(sym1, sym2)]
                    mmat_ = [m for mcart in mmat.values() for m in mcart.values()]
                    if thresh is not None:
                        for m in mmat_:
                            mask = np.abs(m.data) < thresh
                            m.data[mask] = 0
                            m.eliminate_zeros()

                    data = [m.data for m in mmat_ if m.nnz > 0]
                    indices = [m.indices for m in mmat_ if m.nnz > 0]
                    indptr = [m.indptr for m in mmat_ if m.nnz > 0]
                    shape = [m.shape for m in mmat_ if m.nnz > 0]
                    irreps_cart = [(irrep, cart) for cart in mmat.keys() for irrep in mmat[cart].keys()]
                    irreps_cart = [(irrep, cart) for (irrep,cart),m in zip(irreps_cart,mmat_) if m.nnz > 0]

                    if len(data) > 0:
                        try:
                            group_j = group[J_group_key(J1, J2)]
                        except:
                            group_j = group.create_group(J_group_key(J1, J2))
                        try:
                            group_sym = group_j[sym_group_key(sym1, sym2)]
                        except:
                            group_sym = group_j.create_group(sym_group_key(sym1, sym2))
                        group_sym.create_dataset("mmat_data", data=np.concatenate(data))
                        group_sym.create_dataset("mmat_indices", data=np.concatenate(indices))
                        group_sym.create_dataset("mmat_indptr", data=np.concatenate(indptr))
                        group_sym.attrs["mmat_nnz"] = [len(dat) for dat in data]
                        group_sym.attrs["mmat_nind"] = [len(ind) for ind in indices]
                        group_sym.attrs["mmat_nptr"] = [len(ind) for ind in indptr]
                        group_sym.attrs["mmat_irreps_cart"] = irreps_cart
                        group_sym.attrs["mmat_shape"] = shape


    def read(self, filename, name=None, thresh=None, bra=lambda **kw: True,
             ket=lambda **kw: True, **kwargs):
        """Reads object from HDF5 file

        Args:
            filename : str
                Name of HDF5 file
            name : str
                Name of the data group, by default name of the variable is used
            thresh : float
                Threshold for neglecting matrix elements when reading from file
            bra, ket : function(**kw)
                State filters for bra and ket basis sets, take as arguments
                state quantum numbers and energies J, sym, m, k, and enr
                and return True or False depending on if the corresponding state
                needs to be included or excluded form the basis.
                By default, all states stored in the file are included.
                The following keyword arguments are passed into the bra and ket functions:
                    J : float (round to first decimal)
                        Value of J quantum number
                    sym : str
                        State symmetry
                    enr : float
                        State energy
                    m : str
                        State assignment in the M subspace, usually just a value
                        of the m quantum number as a string
                    k : str
                        State assignment in the K subspace, which are the rotational
                        or ro-vibrational quanta joined in a string
        """
        J_key_re = re.sub(r'1.0', '\d+\.\d+', J_group_key(1, 1))
        sym_key_re = re.sub(r'A', '\w+', sym_group_key('A', 'A'))

        # name of HDF5 data group
        if name is None:
            name = retrieve_name(self)

        with h5py.File(filename, 'a') as fl:

            try:
                group = fl[name]
            except KeyError:
                raise KeyError(f"file '{filename}' has no dataset with the name '{name}'") from None

            class_name = group.attrs["__class_name__"]
            if class_name != self.class_name():
                raise TypeError(f"dataset with the name '{name}' in file '{filename}' " + \
                    f"has different type: '{class_name}'") from None

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

            # apply state filters, generate state indices that are included

            self.Jlist1 = [J for J in self.Jlist1 if bra(J=J)]
            self.Jlist2 = [J for J in self.Jlist2 if ket(J=J)]

            self.symlist1 = {J : [sym for sym in self.symlist1[J] if bra(J=J, sym=sym)]
                             for J in self.Jlist1}
            self.symlist2 = {J : [sym for sym in self.symlist2[J] if ket(J=J, sym=sym)]
                             for J in self.Jlist2}

            self.ind_k1 = {J : {sym : [i for i,(q,e) in enumerate(self.quanta_k1[J][sym]) if bra(J=J, sym=sym, k=q, enr=e)]
                           for sym in self.symlist1[J]}
                           for J in self.Jlist1}
            self.ind_k2 = {J : {sym : [i for i,(q,e) in enumerate(self.quanta_k2[J][sym]) if ket(J=J, sym=sym, k=q, enr=e)]
                           for sym in self.symlist2[J]}
                           for J in self.Jlist2}

            self.ind_m1 = {J : {sym : [i for i,q in enumerate(self.quanta_m1[J][sym]) if bra(J=J, sym=sym, m=q)]
                           for sym in self.symlist1[J]}
                           for J in self.Jlist1}
            self.ind_m2 = {J : {sym : [i for i,q in enumerate(self.quanta_m2[J][sym]) if ket(J=J, sym=sym, m=q)]
                           for sym in self.symlist2[J]}
                           for J in self.Jlist2}

            # self.quanta_k1 = {J : {sym : [(q,e) for (q,e) in self.quanta_k1[J][sym] if bra(J=J, sym=sym, k=q, enr=e)]
            #                   for sym in self.symlist1[J]}
            #                   for J in self.Jlist1}
            # self.quanta_k2 = {J : {sym : [(q,e) for (q,e) in self.quanta_k2[J][sym] if ket(J=J, sym=sym, k=q, enr=e)]
            #                   for sym in self.symlist2[J]}
            #                   for J in self.Jlist2}

            # self.quanta_m1 = {J : {sym : [q for q in self.quanta_m1[J][sym] if bra(J=J, sym=sym, m=q)]
            #                   for sym in self.symlist1[J]}
            #                   for J in self.Jlist1}
            # self.quanta_m2 = {J : {sym : [q for q in self.quanta_m2[J][sym] if ket(J=J, sym=sym, m=q)]
            #                   for sym in self.symlist2[J]}
            #                   for J in self.Jlist2}

            self.dim_k1 = {J : {sym : len(self.ind_k1[J][sym]) for sym in self.symlist1[J]} for J in self.Jlist1}
            self.dim_k2 = {J : {sym : len(self.ind_k2[J][sym]) for sym in self.symlist2[J]} for J in self.Jlist2}
            self.dim_m1 = {J : {sym : len(self.ind_m1[J][sym]) for sym in self.symlist1[J]} for J in self.Jlist1}
            self.dim_m2 = {J : {sym : len(self.ind_m2[J][sym]) for sym in self.symlist2[J]} for J in self.Jlist2}

            self.dim1 = {J : {sym : self.dim_k1[J][sym] * self.dim_m1[J][sym]
                         for sym in self.symlist1[J]} for J in self.Jlist1}
            self.dim2 = {J : {sym : self.dim_k2[J][sym] * self.dim_m2[J][sym]
                         for sym in self.symlist2[J]} for J in self.Jlist2}

            # read M and K tensors

            mydict = lambda: defaultdict(mydict)
            self.kmat = mydict()
            self.mmat = mydict()

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
                            if sym1 not in self.symlist1[J1] or sym2 not in self.symlist2[J2]:
                                continue

                            # indices of selected quanta in K and M tensors
                            ik1 = self.ind_k1[J1][sym1]
                            ik2 = self.ind_k2[J2][sym2]
                            im1 = self.ind_m1[J1][sym1]
                            im2 = self.ind_m2[J2][sym2]

                            # read K-matrix

                            kmat = None
                            try:
                                nnz = group_sym.attrs['kmat_nnz']
                                nind = group_sym.attrs['kmat_nind']
                                nptr = group_sym.attrs['kmat_nptr']
                                shape = group_sym.attrs['kmat_shape']
                                irreps = group_sym.attrs['kmat_irreps']
                                data = np.split(group_sym['kmat_data'], np.cumsum(nnz))[:-1]
                                indices = np.split(group_sym['kmat_indices'], np.cumsum(nind))[:-1]
                                indptr = np.split(group_sym['kmat_indptr'], np.cumsum(nptr))[:-1]
                                kmat = {irrep : csr_matrix((dat, ind, ptr), shape=sh)[ik1, :].tocsc()[:, ik2].tocsr()
                                        for irrep,dat,ind,ptr,sh in zip(irreps,data,indices,indptr,shape)}

                            except KeyError:
                                pass

                            # add K-matrix to tensor object
                            if kmat is not None:
                                # remove elements smaller that 'thresh'
                                if thresh is not None:
                                    for k in kmat.values():
                                        mask = np.abs(k.data) < thresh
                                        k.data[mask] = 0
                                        k.eliminate_zeros()
                                self.kmat[(J1, J2)][(sym1, sym2)] = kmat

                            # read M-matrix

                            mmat = None
                            try:
                                nnz = group_sym.attrs['mmat_nnz']
                                nind = group_sym.attrs['mmat_nind']
                                nptr = group_sym.attrs['mmat_nptr']
                                shape = group_sym.attrs['mmat_shape']
                                irreps_cart = group_sym.attrs['mmat_irreps_cart']
                                data = np.split(group_sym['mmat_data'], np.cumsum(nnz))[:-1]
                                indices = np.split(group_sym['mmat_indices'], np.cumsum(nind))[:-1]
                                indptr = np.split(group_sym['mmat_indptr'], np.cumsum(nptr))[:-1]
                                mmat = mydict()
                                for ielem, (irrep, cart) in enumerate(irreps_cart):
                                    dat = data[ielem]
                                    ind = indices[ielem]
                                    ptr = indptr[ielem]
                                    sh = shape[ielem]
                                    mat = csr_matrix((dat, ind, ptr), shape=sh)[im1, :].tocsc()[:, im2].tocsr()
                                    mmat[cart][int(irrep)] = mat
                            except KeyError:
                                pass

                            # add M-matrix to tensor object
                            if mmat is not None:
                                # remove elements smaller that 'thresh'
                                if thresh is not None:
                                    for mm in mmat.values():
                                        for m in mm.values():
                                            mask = np.abs(m.data) < thresh
                                            m.data[mask] = 0
                                            m.eliminate_zeros()
                                self.mmat[(J1, J2)][(sym1, sym2)] = mmat


    def read_states(self, filename, **kwargs):
        """Reads stationary basis states information from the old-format richmol
        states file produced, for example, by TROVE program

        Args:
            filename : str
                Name of richmol states file
        """
        # read states file

        mydict = lambda: defaultdict(mydict)
        energy = mydict()
        assign = mydict()
        map_kstates = dict()

        with open(filename, 'r') as fl:
            for line in fl:
                w = line.split()
                try:
                    J = round(float(w[0]),1)
                    id = np.int(w[1])
                    sym = w[2]
                    ndeg = int(w[3])
                    enr = float(w[4])
                    qstr = ' '.join([w[i] for i in range(5,len(w))])
                except (IndexError, ValueError):
                    raise ValueError(f"error while reading file '{filename}'") from None

                for ideg in range(ndeg):
                    try:
                        energy[J][sym].append(enr)
                        assign[J][sym].append(qstr)
                    except Exception:
                        energy[J][sym] = [enr]
                        assign[J][sym] = [qstr]

                    # mapping between (J,id,ideg) and basis set index in the group
                    # of states sharing the same J and symmetry
                    map_kstates[(J, id, ideg+1)] = (len(energy[J][sym])-1, sym)

        if len(list(energy.keys())) == 0:
            raise Exception(f"zero number of states in file '{filename}'") from None

        # list of m quanta for different J
        mquanta = {J : [m for m in np.linspace(-J, J, int(2*J)+1)] for J in energy.keys()}

        # generate mapping beteween m quanta and basis set index
        map_mstates = {(J, m) : ind_m for J in mquanta.keys() for ind_m,m in enumerate(mquanta[J]) }

        # generate attributes

        Jlist = list(energy.keys())
        symlist = {J : [sym for sym in energy[J].keys()] for J in Jlist}
        dim_m = {J : {sym : len(mquanta[J]) for sym in symlist[J]} for J in Jlist}
        dim_k = {J : {sym : len(energy[J][sym]) for sym in symlist[J]} for J in Jlist}
        dim = {J : {sym : dim_m[J][sym] * dim_k[J][sym] for sym in symlist[J]} for J in Jlist}
        quanta_m = {J : {sym : ["%4.1f"%m for m in mquanta[J]] for sym in symlist[J]} for J in Jlist}
        quanta_k = {J : {sym : [(q,e) for q,e in zip(assign[J][sym], energy[J][sym])] for sym in symlist[J]} for J in Jlist}

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

        self.cart = '0'
        self.os = [(0,0)]
        self.rank = 0

        self.mmat = {(J, J) : {(sym, sym) : {'0' : {0 : csr_matrix(np.eye(len(mquanta[J])), dtype=np.complex128)}}
                     for sym in symlist[J]} for J in Jlist}

        self.kmat = {(J, J) : {(sym, sym) : {0 : csr_matrix(np.diag(energy[J][sym]), dtype=np.complex128)}
                     for sym in symlist[J]} for J in Jlist}

        # ====== to be used only by read_trans ======
        self.Jlist = Jlist
        self.dim_k = dim_k
        self.dim_m = dim_m
        self.map_kstates = map_kstates
        self.map_mstates = map_mstates
        # exclude these attributes from stroring into HDF5 file
        # note that these attributes won't be affected by state filters in CarTens.__init_() => CarTens.read()
        self.store_exclude = ["Jlist", "dim_m", "dim_k", "map_mstates", "map_kstates"]
        #==========================================

        # write tensor into temp file

        tmp_file = "tmp_read_states.h5"
        self.store(tmp_file, replace=True, comment="loaded from richmol states file " + filename)

        # re-initialize, now with state filters applied (passed in kwargs)

        CarTens.__init__(self, tmp_file,  **kwargs)
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


    def read_trans(self, filename, thresh=None, **kwargs):
        """Reads matrix elements of Cartesian tensor from the old-format richmol
        matrix elements files produced, for example, by TROVE program

        NOTE: prior running of self.read_states(richmol_states_file) is compulsory,
        as this function loads the necessary information about the stationary states

        Args:
            filename : str
                In old-format style, matrix elements of Cartesian tensor are stored
                in separate files for different values of bra and ket J (or F) quanta.
                The argument 'filename' provides a template for generating the names
                of these files. For example, if filename = "matelem_alpha_j<j1>_j<j2>.rchm",
                the following files will be searched: "matelem_alpha_j0_j0.rchm",
                "matelem_alpha_j0_j1.rchm", "matelem_alpha_j0_j2.rchm",
                "matelem_alpha_j1_j1.rchm", etc., i.e., <j1> and <j2> are replaced
                by the integer values of bra and ket J quanta spanned by the basis
                of stationary states.
                For half-integer values of the F quantum number, replace <j1> and <j2>
                in the template by <f1> and <f2>, these will be substituted
                by the floating point values of F quanta rounded to the first decimal.
            thresh : float
                Threshold for neglecting matrix elements when reading from file
        """
        # irreps[(ncart, nirrep)]
        irreps = {(3,1) : [(1,-1), (1,0), (1,1)],                                               # rank-1 tensor
                  (9,1) : [(2,-2), (2,-1), (2,0), (2,1), (2,2)],                                # traceless and symmetric rank-2 tensor
                  (9,2) : [(0,0), (2,-2), (2,-1), (2,0), (2,1), (2,2)],                         # symmetric rank-2 tensor
                  (9,3) : [(0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0), (2,1), (2,2)]}   # non-symmetric rank-2 tensor

        # ranks[ncart]
        ranks = {3 : 1, 9 : 2}

        self.cart = []

        tens_cart = []
        tens_nirrep = None
        tens_ncart = None

        # name of temp HDF5 file
        tmp_file = "tmp_read_trans.h5"
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        for J1_ in self.Jlist:
            for J2_ in self.Jlist:

                J1, J2 = (J1_, J2_)
                transp = False

                F1_str = str(round(J1,1))
                F2_str = str(round(J2,1))
                J1_str = str(int(round(J1,0)))
                J2_str = str(int(round(J2,0)))

                fname = re.sub(r"\<f1\>", F1_str, filename)
                fname = re.sub(r"\<f2\>", F2_str, fname)
                fname = re.sub(r"\<j1\>", J1_str, fname)
                fname = re.sub(r"\<j2\>", J2_str, fname)

                if not os.path.exists(fname):
                    J1, J2 = (J2_, J1_)
                    transp = True
                    fname = re.sub(r"\<f1\>", F2_str, filename)
                    fname = re.sub(r"\<f2\>", F1_str, fname)
                    fname = re.sub(r"\<j1\>", J2_str, fname)
                    fname = re.sub(r"\<j2\>", J1_str, fname)
                    if not os.path.exists(fname):
                        continue

                mydict = lambda: defaultdict(mydict)
                mrow = mydict()
                mcol = mydict()
                mdata = mydict()
                kcol = mydict()
                krow = mydict()
                kdata = mydict()

                with open(fname, "r") as fl:

                    iline = 0
                    eof = False
                    read_m = False
                    read_k = False

                    for line in fl:
                        strline = line.rstrip('\n')

                        if iline == 0:
                            if strline != "Start richmol format":
                                raise Exception(f"file '{fname}' has bogus header '{strline}'")
                            iline+=1
                            continue

                        if strline == "End richmol format":
                            eof = True
                            break

                        if iline == 1:
                            w = strline.split()
                            try:
                                tens_name = w[0]
                                nirrep = int(w[1])
                                ncart = int(w[2])
                            except (IndexError, ValueError):
                                raise ValueError(f"error while reading file '{filename}', line = {iline}") from None
                            if tens_nirrep is not None and tens_nirrep != nirrep:
                                raise ValueError(f"'nirrep' = {nirrep} read from file {fname} is different from " + \
                                    f"the value {tens_nirrep} read from previous files") from None
                            if tens_ncart is not None and tens_ncart != ncart:
                                raise ValueError(f"'ncart' = {ncart} read from file {fname} is different from " + \
                                    f"the value {tens_ncart} read from previous files") from None
                            tens_nirrep = nirrep
                            tens_ncart = ncart
                            try:
                                self.os = irreps[(ncart, nirrep)]
                                irreps_list = sorted(list(set(omega for omega, sigma in self.os)))
                            except KeyError:
                                raise ValueError(f"can't infer Cartesian tensor irreps from the number " + \
                                    f"of Cartesian components = {ncart} and number of irreps = {nirrep}") from None
                            try:
                                self.rank = ranks[ncart]
                            except KeyError:
                                raise ValueError(f"can't infer rank of Cartesian tensor from the number " + \
                                    f"of Cartesian components = {ncart}") from None
                            iline+=1
                            continue

                        if strline == "M-tensor":
                            read_m = True
                            read_k = False
                            iline+=1
                            continue

                        if strline == "K-tensor":
                            read_m = False
                            read_k = True
                            iline+=1
                            continue

                        if read_m is True and strline.split()[0] == "alpha":
                            w = strline.split()
                            try:
                                icmplx = int(w[2])
                                cart = w[3].lower()
                            except (IndexError, ValueError):
                                raise ValueError(f"error while reading file '{filename}', line = {iline}") from None
                            self.cart = list(set(self.cart + [cart]))
                            cmplx_fac = (1j, 1)[icmplx+1]
                            iline+=1
                            continue

                        if read_m is True:
                            w = strline.split()
                            try:
                                m1 = round(float(w[0]),1)
                                m2 = round(float(w[1]),1)
                                mval = [ float(val) * cmplx_fac for val in w[2:] ]
                            except (IndexError, ValueError):
                                raise ValueError(f"error while reading file '{filename}', line = {iline}") from None
                            im1 = self.map_mstates[(J1, m1)]
                            im2 = self.map_mstates[(J2, m2)]
                            for i,irrep in enumerate(irreps_list):
                                if thresh is not None and abs(mval[i]) < thresh:
                                    continue
                                try:
                                    mrow[cart][irrep].append(im1)
                                    mcol[cart][irrep].append(im2)
                                    mdata[cart][irrep].append(mval[i])
                                except AttributeError:
                                    mrow[cart][irrep] = [im1]
                                    mcol[cart][irrep] = [im2]
                                    mdata[cart][irrep] = [mval[i]]

                        if read_k is True:
                            w = strline.split()
                            try:
                                id1 = int(w[0])
                                id2 = int(w[1])
                                ideg1 = int(w[2])
                                ideg2 = int(w[3])
                                kval = [float(val) for val in w[4:]]
                            except (IndexError, ValueError):
                                raise ValueError(f"error while reading file '{filename}', line = {iline}") from None
                            istate1, sym1 = self.map_kstates[(J1, id1, ideg1)]
                            istate2, sym2 = self.map_kstates[(J2, id2, ideg2)]
                            sym = (sym1, sym2)
                            for i,irrep in enumerate(irreps_list):
                                if thresh is not None and abs(kval[i]) < thresh:
                                    continue
                                try:
                                    krow[sym][irrep].append(istate1)
                                    kcol[sym][irrep].append(istate2)
                                    kdata[sym][irrep].append(kval[i])
                                except AttributeError:
                                    krow[sym][irrep] = [istate1]
                                    kcol[sym][irrep] = [istate2]
                                    kdata[sym][irrep] = [kval[i]]

                        iline +=1

                    if eof is False:
                        raise Exception(f"'{fname}' has bogus footer '{strline}'")

                self.mmat = dict()
                self.kmat = dict()

                mshape = {(sym1, sym2) : (self.dim_m[J1][sym1], self.dim_m[J2][sym2]) for (sym1, sym2) in kdata.keys()}
                kshape = {(sym1, sym2) : (self.dim_k[J1][sym1], self.dim_k[J2][sym2]) for (sym1, sym2) in kdata.keys()}

                if transp is True:
                    self.mmat[(J1_,J2_)] = {tuple(reversed(sym)) : {cart : {irrep :
                                          csr_matrix((np.conj(mdata[cart][irrep]), (mcol[cart][irrep], mrow[cart][irrep])), shape=tuple(reversed(mshape[sym])))
                                          for irrep in mdata[cart].keys()}
                                          for cart in mdata.keys()}
                                          for sym in mshape.keys()}

                    self.kmat[(J1_,J2_)] = {tuple(reversed(sym)) : {irrep :
                                          csr_matrix((np.conj(kdata[sym][irrep]), (kcol[sym][irrep], krow[sym][irrep])), shape=tuple(reversed(kshape[sym])))
                                          for irrep in kdata[sym].keys()}
                                          for sym in kshape.keys()}
                else:
                    self.mmat[(J1_,J2_)] = {sym : {cart : {irrep :
                                          csr_matrix((mdata[cart][irrep], (mrow[cart][irrep], mcol[cart][irrep])), shape=mshape[sym])
                                          for irrep in mdata[cart].keys()}
                                          for cart in mdata.keys()}
                                          for sym in mshape.keys()}

                    self.kmat[(J1_,J2_)] = {sym : {irrep :
                                          csr_matrix((kdata[sym][irrep], (krow[sym][irrep], kcol[sym][irrep])), shape=kshape[sym])
                                          for irrep in kdata[sym].keys()}
                                          for sym in kshape.keys()}

                # delete empty entries in kmat
                symlist = list(self.kmat[(J1_,J2_)].keys())
                for sym in symlist:
                    if all(mat.nnz == 0 for mat in self.kmat[(J1_,J2_)][sym].values()):
                        del self.kmat[(J1_,J2_)][sym]

                # delete empty entries in mmat
                symlist = list(self.mmat[(J1_,J2_)].keys())
                for sym in symlist:
                    if all(mat.nnz == 0 for tmat in self.mmat[(J1_,J2_)][sym].values() for mat in tmat.values()):
                        del self.mmat[(J1_,J2_)][sym]

                # update tensor in temp HDF5 file

                self.store(tmp_file, replace=False)

        # re-initialize, now with state filters applied (passed in kwargs)

        CarTens.__init__(self, tmp_file, **kwargs)

        # delete temp HDF5 file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)



    def class_name(self):
        """Generates '__class_name__' attribute for the tensor data group in HDF5 file"""
        return self.__module__ + '.' + self.__class__.__name__



def J_group_key(J1, J2):
    """Generates HDF5 data group name for matrix elements between states
    with bra and ket J quanta equal to J1 and J2, respectively
    """
    return 'J:' + str(round(float(J1), 1)) + ',' + str(round(float(J2), 1))


def sym_group_key(sym1, sym2):
    """Generates HDF5 data group name for matrix elements between states
    with bra and ket state symmetries equal to sym1 and sym2, respectively
    """
    return 'sym:' + str(sym1) + ',' + str(sym2)


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
