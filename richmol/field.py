import numpy as np
from scipy.sparse import kron, csr_matrix
import itertools
from itertools import chain
import copy
import h5py
import inspect
import datetime
import time
import re
from collections import defaultdict
from richmol import json
import os


class CarTens():
    """General class for laboratory-frame Cartesian tensor operators

    Kwargs:
        filename : str
            Name of HDF5 file, from which the complete tensor is loaded.
            Alternatively, one can load tensor matrix elements from the old-format
            richmol files, e.g., created by TROVE, using 'states' and 'trans'
            keyword arguments.
        name : str
            Name of the tensor data group in HDF5 file.
        states : str
            Name of old-format states richmol file, from which the field-free
            basis states are loaded
        trans : str
            Name of old-format trans richmol file, from which the matrix elements
            of tensor are loaded
        thresh : float
            Threshold for neglecting matrix elements when reading from files
    """
    prefac = 1.0

    def __init__(self, **kwargs):

        if "filename" in kwargs:
            # read data from HDF5 file
            filename = kwargs["filename"]
            kw = {elem : kwargs[elem] for elem in ("thresh", "name", "bra", "ket") if elem in kwargs}
            self.read(filename=filename, **kw)

        elif "states" in kwargs:
            # read field-free basis from old-format richmol files
            kw = {elem : kwargs[elem] for elem in ("filter",) if elem in kwargs}
            states = self.read_states(kwargs["states"], **kw)
            if "trans" in kwargs:
                # read matrix elements from old-format richmol files
                self.read_trans(states, kwargs["trans"], thresh=thresh)
            else:
                self = states

        # apply filters
        # TODO: move the filters out of reading functions here


    def tomat(self, form='block', **kwargs):
        """Returns full MxK matrix representation of tensor

        Args:
            form : str
                Form of the output matrix, if form == 'full', a 2D dense matrix
                will be returned, if form == 'block' (default), only non-zero matrix
                elements will be returned in blocks for different pairs of J quanta
                and different symmetries, i.e., mat[(J1, J2)][(sym1, sym2)] -> ndarray

        Kwargs:
            cart : str
                String denoting desired Cartesian component of tensor, e.g., cart='xx'
                In case cart is not provided, function will assume that tensor
                has been multiplied with external field and will return
                the corresponding matrix representation

        Returns:
            mat : nested dict or 2D ndarray
                For form='block', returns dictionary containing matrix elements
                of tensor operator for different pairs of J quanta and different
                pairs of symmetries, i.e., mat[(J1, J2)][(sym1, sym2)] -> ndarray.
                For form='full', returns 2D dense matrix, the order of blocks in full
                matrix corresponds to [(J, sym) for J in self.Jlist for sym in self.symlist[J]].
        """
        assert (form in ('block', 'full')), f"bad value of argument 'form' = '{form}' (use 'block' or 'full')"

        if 'cart' in kwargs:
            cart = kwargs['cart']
            if cart not in self.cart:
                raise ValueError(f"can't find input Cartesian component '{cart}' " + \
                    f"amongst tensor components {self.cart}") from None
        else:
            try:
                x = self.mfmat
            except AttributeError:
                raise AttributeError(f"you need to specify Cartesian component of tensor " + \
                    f"of multiply tensor with field before computing its matrix representation") from None

        mat = dict()

        for Jpair in list(set(self.mmat.keys()) & set(self.kmat.keys())):

            mmat_J = self.mmat[Jpair]
            kmat_J = self.kmat[Jpair]
            mat[Jpair] = dict()

            for sympair in list(set(mmat_J.keys()) & set(kmat_J.keys())):

                # choose M[cart] or field-contracted MF tensor
                if 'cart' in kwargs:
                    try:
                        mmat = mmat_J[sympair][cart]
                    except KeyError:
                        continue   # all MEs are zero for current J and symmetry pairs
                else:
                    mmat = mfmat[Jpair][sympair]

                # K tensor
                kmat = kmat_J[sympair]

                # do M \otimes K
                me = np.sum( kron(mmat[irrep], kmat[irrep]).todense()
                             for irrep in list(set(mmat.keys()) & set(kmat.keys())) )

                # add to matrix if non zero and multiply by tensor prefactor (=1 by default)
                if not np.isscalar(me):
                    mat[Jpair][sympair] = me * self.prefac

        if form == 'full':
            mat = self.full_form(mat)
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


    def full_form(self, mat):
        """Converts block representation of tensor matrix into a full matrix form"""
        res = np.block([[ mat[(J1, J2)][(sym1, sym2)]
                          if (J1, J2) in mat.keys() and (sym1, sym2) in mat[(J1, J2)].keys()
                          else np.zeros((self.dim1[J1][sym1], self.dim2[J2][sym2])) \
                          for J2 in self.Jlist2 for sym2 in self.symlist2[J2] ]
                          for J1 in self.Jlist1 for sym1 in self.symlist1[J1] ])
        return res


    def block_form(self, mat):
        """Converts full matrix representation of tensor into a block form"""
        ind0 = np.cumsum([self.dim1[J][sym] for J in self.Jlist1 for sym in self.symlist1[J]])
        ind1 = np.cumsum([self.dim2[J][sym] for J in self.Jlist2 for sym in self.symlist2[J]])
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
            self.prefac = self.prefac * arg
        else:
            raise TypeError(f"bad argument type for 'arg': '{type(arg)}'") from None


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
        field_prod = {comb : np.prod(fxyz[list(comb)]) \
                      for comb in itertools.product((0,1,2), repeat=self.rank) \
                      if abs(np.prod(fxyz[list(comb)])) >= tol}

        nirrep = len(set(omega for (omega,sigma) in self.os)) # number of tensor irreps
        res = np.zeros(nirrep, dtype=np.complex128)

        # contract M-tensor with field

        self.mfmat = dict()

        for (J1, J2), mmat_J in self.mmat.items():
            for (sym1, sym2), mmat_sym in mmat_J.items():

                res[:] = 0
                for cart, mmat_irrep in mmat_sym.items():
                    try:
                        fprod = field_prod[cart]
                    except KeyError:
                        continue     # the product of field components can be neglected

                    for irrep, mmat in mmat_irrep.items():
                        res[irrep] = res[irrep] + mmat * fprod

                res_dict = {irrep : elem for elem in res if np.abs(elem) > tol}
                if len(res_dict) > 0:
                    try:
                        self.mfmat[(J1, J2)][(sym1, sym2)] = res_dict
                    except KeyError:
                        self.mfmat[(J1, J2)] = {(sym1, sym2) : res_dict}


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
                    vec2[J1][sym1] += np.sum(res) * self.prefac
                except KeyError:
                    vec2[J1] = {sym1 : np.sum(res) * self.prefac}

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

    __rmul__ = __mul__


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
                else:
                    raise RuntimeError(f"found existing dataset '{name}' in file " + \
                        f"'{filename}', use replace=True to replace it") from None

            group = fl.create_group(name)
            class_name = self.class_name()
            group.attrs["__class_name__"] = class_name

            # description of object
            doc = "Cartesian tensor operator"

            # add date/time
            date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            doc += ", store date: " + date.replace('\n','')

            # add user comment
            if comment is not None:
                doc += ", comment: " + " ".join(elem for elem in comment.split())

            group.attrs['__doc__'] = doc

            # store attributes

            attrs = list(set(vars(self).keys()) - set(['mmat', 'kmat', 'molecule', 'basis']))
            for attr in attrs:
                val = getattr(self, attr)
                try:
                    group.attrs[attr] = val
                except TypeError:
                    jd = json.dumps(val)
                    group.attrs[attr + "__json"] = jd

            # store basis functions (dict(Solution()))

            if hasattr(self, 'basis'):
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
                        group_sym.create_dataset('kmat_data', data=np.concatenate(data))
                        group_sym.create_dataset('kmat_indices', data=np.concatenate(indices))
                        group_sym.create_dataset('kmat_indptr', data=np.concatenate(indptr))
                        group_sym.attrs['kmat_nnz'] = [len(dat) for dat in data]
                        group_sym.attrs['kmat_nind'] = [len(ind) for ind in indices]
                        group_sym.attrs['kmat_nptr'] = [len(ind) for ind in indptr]
                        group_sym.attrs['kmat_irreps'] = irreps
                        group_sym.attrs['kmat_shape'] = shape

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
                        group_sym.create_dataset('mmat_data', data=np.concatenate(data))
                        group_sym.create_dataset('mmat_indices', data=np.concatenate(indices))
                        group_sym.create_dataset('mmat_indptr', data=np.concatenate(indptr))
                        group_sym.attrs['mmat_nnz'] = [len(dat) for dat in data]
                        group_sym.attrs['mmat_nind'] = [len(ind) for ind in indices]
                        group_sym.attrs['mmat_nptr'] = [len(ind) for ind in indptr]
                        group_sym.attrs['mmat_irreps_cart'] = irreps_cart
                        group_sym.attrs['mmat_shape'] = shape


    def read(self, filename, name=None, thresh=None, bra=lambda **kw: True, ket=lambda **kw: True):
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

            # apply filters, generate different sets of quanta for bra and ket states

            self.Jlist1 = [J for J in self.Jlist if bra(J=J)]
            self.Jlist2 = [J for J in self.Jlist if ket(J=J)]
            self.symlist1 = {J : [sym for sym in self.symlist[J] if bra(J=J, sym=sym)]
                             for J in self.Jlist1}
            self.symlist2 = {J : [sym for sym in self.symlist[J] if ket(J=J, sym=sym)]
                             for J in self.Jlist2}
            self.quanta_k1 = {J : {sym : [(q,i) for i,q in enumerate(self.quanta_k[J][sym]) if bra(J=J, sym=sym, k=q)]
                              for sym in self.symlist1[J]}
                              for J in self.Jlist1}
            self.quanta_k2 = {J : {sym : [(q,i) for i,q in enumerate(self.quanta_k[J][sym]) if ket(J=J, sym=sym, k=q)]
                              for sym in self.symlist2[J]}
                              for J in self.Jlist2}
            self.quanta_m1 = {J : {sym : [(q,i) for i,q in enumerate(self.quanta_m[J][sym]) if bra(J=J, sym=sym, m=q)]
                              for sym in self.symlist1[J]}
                              for J in self.Jlist1}
            self.quanta_m2 = {J : {sym : [(q,i) for i,q in enumerate(self.quanta_m[J][sym]) if ket(J=J, sym=sym, m=q)]
                              for sym in self.symlist2[J]}
                              for J in self.Jlist2}
            self.dim_k1 = {J : {sym : len(self.quanta_k1[J][sym])
                           for sym in self.symlist1[J]}
                           for J in self.Jlist1}
            self.dim_k2 = {J : {sym : len(self.quanta_k2[J][sym])
                           for sym in self.symlist2[J]}
                           for J in self.Jlist2}
            self.dim_m1 = {J : {sym : len(self.quanta_m1[J][sym])
                           for sym in self.symlist1[J]}
                           for J in self.Jlist1}
            self.dim_m2 = {J : {sym : len(self.quanta_m2[J][sym])
                           for sym in self.symlist2[J]}
                           for J in self.Jlist2}
            self.dim1 = {J : {sym : self.dim_k1[J][sym] * self.dim_m1[J][sym]
                           for sym in self.symlist1[J]}
                           for J in self.Jlist1}
            self.dim2 = {J : {sym : self.dim_k2[J][sym] * self.dim_m2[J][sym]
                           for sym in self.symlist2[J]}
                           for J in self.Jlist2}

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
                            ind_k1 = [elem[1] for elem in self.quanta_k1[J1][sym1]]
                            ind_k2 = [elem[1] for elem in self.quanta_k2[J2][sym2]]
                            ind_m1 = [elem[1] for elem in self.quanta_m1[J1][sym1]]
                            ind_m2 = [elem[1] for elem in self.quanta_m2[J2][sym2]]

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
                                kmat = {irrep : csr_matrix((dat, ind, ptr), shape=sh)[ind_k1, :].tocsc()[:, ind_k2].tocsr()
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
                                    mat = csr_matrix((dat, ind, ptr), shape=sh)[ind_m1, :].tocsc()[:, ind_m2].tocsr()
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


    def read_states(self, filename, filter=lambda **kw: True):
        """Reads basis states information from old-format richmol states file,
        for example, files that are produced by TROVE program

        Args:
            filename : str
                Name of richmol states file
            filter : function(**kw)
                States filter function, takes as arguments state assignment
                quanta and energy, J, sym, m, k, and enr, and returns True or False
                depending on if the corresponding state needs to be included or excluded
                form the basis. By default, all states stored in the file are included.
                The following keyword arguments are passed into the filter function:
                    J : float (round to first decimal)
                        Value of J (or F) quantum number
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
        # read states file

        mydict = lambda: defaultdict(mydict)
        energy = mydict()
        assign = mydict()
        mquanta = mydict()
        map_kstates = mydict()

        with open(filename, 'r') as fl:
            for line in fl:
                w = line.split()
                J = round(float(w[0]),1)
                id = np.int64(w[1])
                sym = w[2]
                ndeg = int(w[3])
                enr = float(w[4])
                qstr = ' '.join([w[i] for i in range(5,len(w))])

                # apply state filters
                if not filter(J=J, sym=sym, k=qstr, enr=enr, id=id):
                    continue
                mlist = [m for m in np.linspace(-J, J, int(2*J)+1) if filter(J=J, sym=sym, m=m)]
                if len(mlist) == 0:
                    continue

                for ideg in range(ndeg):
                    try:
                        energy[J][sym].append(enr)
                        assign[J][sym].append(qstr)
                    except Exception:
                        energy[J][sym] = [enr]
                        assign[J][sym] = [qstr]

                    # mapping between (J,id,ideg) and basis set index in the group
                    # of states sharing the same J and symmetry
                    map_kstates[(J, id, ideg)] = (len(energy[J][sym])-1, sym)

                mquanta[J][sym] = mlist

        # check how many states have passed filter

        if len(list(energy.keys())) == 0:
            raise Exception(f"zero number of states, perhaps selection filters cast out all states") from None

        # generate mapping beteween m quanta and basis set index
        #map_mstates = {(J, m) : ind_m for J in mquanta.keys() for ind_m,m in enumerate(mdict[J]) }

        # generate attributes

        self.Jlist = list(energy.keys())
        self.symlist = {J : [sym for sym in energy[J].keys()] for J in self.Jlist}
        self.dim_m = {J : {sym : len(mquanta[J][sym])
                      for sym in self.symlist[J]}
                      for J in self.Jlist}
        self.dim_k = {J : {sym : len(energy[J][sym])
                      for sym in self.symlist[J]}
                      for J in self.Jlist}
        self.dim = {J : {sym : self.dim_m[J][sym] * self.dim_k[J][sym]
                    for sym in self.symlist[J]}
                    for J in self.Jlist}
        self.quanta_m = {J : {sym : ["%4.1f"%m for m in mquanta[J][sym]] 
                         for sym in self.symlist[J]}
                         for J in self.Jlist}
        self.quanta_k = {J : {sym : assign[J][sym]
                         for sym in self.symlist[J]}
                         for J in self.Jlist}
        self.enr_k = {J : {sym :[enr for enr in energy[J][sym]]
                      for sym in self.symlist[J]}
                      for J in self.Jlist}

        self.cart = '0'
        self.os = [(0,0)]
        self.rank = 0

        self.mmat = {(J, J) : {(sym, sym) : {'0' : {0 : csr_matrix(np.eye(len(mquanta[J][sym])), dtype=np.complex128)}}
                     for sym in self.symlist[J]} for J in self.Jlist}

        self.kmat = {(J, J) : {(sym, sym) : {0 : csr_matrix(np.diag(energy[J][sym]), dtype=np.complex128)}
                     for sym in self.symlist[J]} for J in self.Jlist}

        # write tensor into temp file and reload in CarTens()
        filename = "tmp.h5"
        self.store(filename=filename, name=retrieve_name(self))
        CarTens.__init__(self, filename=filename,  name=retrieve_name(self))
        if os.path.exists("tmp.h5"):
            os.remove("tmp.h5")


    def read_trans(self, states, filename, thresh=None):
        """Reads matrix elements of Cartesian tensor from old-format richmol
        matrix elements files, for example, files that are produced by TROVE program

        Args:
            states : field.CarTens
                Field-free basis states (see read_states)
            filename : str
                In the old format, matrix elements for different bra and ket J quantum
                numbers are stored in separate files. The argument 'filename' provides
                a template for generating the names of these files.
                For example, for filename = "matelem_alpha_j<j1>_j<j2>.rchm",
                the following files will be searched: matelem_alpha_j0_j0.rchm,
                matelem_alpha_j0_j1.rchm, matelem_alpha_j0_j2.rchm,
                matelem_alpha_j1_j1.rchm, etc., where <j1> and <j2> are replaced
                by integer numbers running through all J quanta spanned by the basis.
                For half-integer numbers (e.g., F quanta), replace <j1> and <j2>
                in the template by <f1> and <f2>, these will be replaced by
                floating point numbers rounded to the first decimal.
            thresh : float
                Threshold for neglecting matrix elements when reading from file
        """
        # read M and K tensors for different pairs of J quanta

        Jlist1 = states.Jlist1
        Jlist2 = states.Jlist2

        self.mmat = dict()
        self.kmat = dict()
        tens_cart = []
        tens_nirrep = None
        tens_ncart = None

        for J1 in Jlist1:
            for J2 in Jlist2:

                F1_str = str(round(J1,1))
                F2_str = str(round(J2,1))
                J1_str = str(int(round(J1,0)))
                J2_str = str(int(round(J2,0)))

                fname = re.sub(r"\<f1\>", F1_str, filename)
                fname = re.sub(r"\<f2\>", F2_str, fname)
                fname = re.sub(r"\<j1\>", J1_str, fname)
                fname = re.sub(r"\<j2\>", J2_str, fname)

                if not opersys.path.exists(fname):
                    continue

                with open(fname, "r") as fl:

                    iline = 0
                    eof = False
                    read_m = False
                    read_k = False

                    for line in fl:
                        strline = line.rstrip('\n')

                        if iline == 0:
                            if strline != "Start richmol format":
                                raise Exception(f"matrix elements file '{fname}' has bogus header '{strline}'")
                            iline+=1
                            continue

                        if strline == "End richmol format":
                            eof = True
                            break

                        if iline == 1:
                            w = strline.split()
                            tens_name = w[0]
                            nirrep = int(w[1])
                            ncart = int(w[2])
                            if tens_nirrep is not None and tens_nirrep != nirrep:
                                raise Exception(f"'nirrep' = {nirrep} read from file {fname} is different from " + \
                                    f"the value {tens_nirrep} read from previous files") from None
                            if tens_ncart is not None and tens_ncart != ncart:
                                raise Exception(f"'ncart' = {ncart} read from file {fname} is different from " + \
                                    f"the value {tens_ncart} read from previous files") from None
                            tens_nirrep = nirrep
                            tens_ncart = ncart
                            iline+=1
                            continue

                        if strline == "M-tensor":
                            read_m = True
                            read_k = False
                            mdata = dict()
                            mrow = dict()
                            mcol = dict()
                            iline+=1
                            continue

                        if strline == "K-tensor":
                            read_m = False
                            read_k = True
                            kdata = {(sym1, sym2) : {} for sym1 in states.symlist1[J1] for sym2 in states.symlist2[J2]}
                            krow = {(sym1, sym2) : {} for sym1 in states.symlist1[J1] for sym2 in states.symlist2[J2]}
                            kcol = {(sym1, sym2) : {} for sym1 in states.symlist1[J1] for sym2 in states.symlist2[J2]}
                            iline+=1
                            continue

                        if read_m is True and strline.split()[0] == "alpha":
                            w = strline.split()
                            icmplx = int(w[2])
                            cart = w[3].lower()
                            tens_cart = list(set(tens_cart + [cart]))
                            cmplx_fac = (1j, 1)[icmplx+1]
                            mdata[cart] = dict()
                            mrow[cart] = dict()
                            mcol[cart] = dict()
                            iline+=1
                            continue

                        if read_m is True:
                            w = strline.split()
                            m1 = float(w[0])
                            m2 = float(w[1])
                            try:
                                im1 = states.map_mstates[(J1, m1)]
                            except:
                                continue
                            try:
                                im2 = states.map_mstates[(J2, m2)]
                            except:
                                continue
                            mval = [ float(val) * cmplx_fac for val in w[2:] ]
                            if thresh is not None:
                                irreps = [i for i in range(nirrep) if abs(mval[i]) > thresh]
                            else:
                                irreps = [i for i in range(nirrep)]
                            for irrep in irreps:
                                try:
                                    mrow[cart][irrep].append(im1)
                                    mcol[cart][irrep].append(im2)
                                    mdata[cart][irrep].append(mval[irrep])
                                except KeyError:
                                    mrow[cart][irrep] = [im1]
                                    mcol[cart][irrep] = [im2]
                                    mdata[cart][irrep] = [mval[irrep]]

                        if read_k is True:
                            w = strline.split()
                            id1 = int(w[0])
                            id2 = int(w[1])
                            ideg1 = int(w[2])
                            ideg2 = int(w[3])
                            kval = [float(val) for val in w[4:]]
                            try:
                                istate1, sym1 = states.map_kstates[(J1, id1, ideg1)]
                            except:
                                continue
                            try:
                                istate2, sym2 = states.map_kstates[(J2, id2, ideg2)]
                            except:
                                continue
                            if thresh is not None:
                                irreps = [i for i in range(nirrep) if abs(kval[i]) > thresh]
                            else:
                                irreps = [i for i in range(nirrep)]
                            sym = (sym1, sym2)
                            for irrep in irreps:
                                try:
                                    krow[sym][irrep].append(istate1)
                                    kcol[sym][irrep].append(istate2)
                                    kdata[sym][irrep].append(kval[irrep])
                                except IndexError:
                                    krow[sym][irrep] = [istate1]
                                    kcol[sym][irrep] = [istate2]
                                    kdata[sym][irrep] = [kval[irrep]]

                        iline +=1

                    if eof is False:
                        raise Exception(f"matrix-elements file '{fname}' has bogus footer '{strline}'")

                fshape = lambda sym: (states.dim_m1[J1][sym[0]], states.dim_m2[J2][sym[1]])
                self.mmat[(J1,J2)] = {sym : {cart : {irrep :
                                      csr_matrix((mdata[cart][irrep], (mrow[cart][irrep], mcol[cart][irrep])), shape=fshape(sym))
                                      for irrep in mdata[cart].keys()}
                                      for cart in mdata.keys()}
                                      for sym in kdata.keys()}

                fshape = lambda sym: (states.dim_k1[J1][sym[0]], states.dim_k2[J2][sym[1]])
                self.kmat[(J1,J2)] = {sym : {irrep :
                                      csr_matrix((kdata[sym][irrep], (krow[sym][irrep], kcol[sym][irrep])), shape=fshape(sym))
                                      for irrep in kdata[sym].keys()}
                                      for sym in kdata.keys()}

                # delete empty entries in kmat
                symlist = list(self.kmat[(J1, J2)].keys())
                for sym in symlist:
                    if all(mat.nnz == 0 for mat in self.kmat[(J1, J2)][sym].values()):
                        del self.kmat[(J1, J2)][sym]

                # delete empty entries in mmat
                symlist = list(self.mmat[(J1, J2)].keys())
                for sym in symlist:
                    if all(mat.nnz == 0 for tmat in self.mmat[(J1, J2)][sym].values() for mat in tmat.values()):
                        del self.mmat[(J1, J2)][sym]

        # copy some attributes from states

        self.symlist1 = states.symlist1
        self.symlist2 = states.symlist2
        self.dim_m1 = states.dim_m1
        self.dim_m2 = states.dim_m2
        self.dim_k1 = states.dim_k1
        self.dim_k2 = states.dim_k2
        self.dim1 = states.dim1
        self.dim2 = states.dim2
        self.assign_m1 = states.assign_m1
        self.assign_m2 = states.assign_m2
        self.assign_k1 = states.assign_k1
        self.assign_k2 = states.assign_k2
        self.cart = tens_cart

        # irreps[(ncart, nirrep)]
        irreps = {(3,1) : [(1,-1), (1,0), (1,1)],                                               # rank-1 tensor
                  (9,1) : [(2,-2), (2,-1), (2,0), (2,1), (2,2)],                                # traceless and symmetric rank-2 tensor
                  (9,2) : [(0,0), (2,-2), (2,-1), (2,0), (2,1), (2,2)],                         # symmetric rank-2 tensor
                  (9,3) : [(0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0), (2,1), (2,2)]}   # non-symmetric rank-2 tensor

        # ranks[ncart]
        ranks = {3 : 1, 9 : 2}

        # infer list of spherical-tensor indices
        try:
            self.os = irreps[(tens_ncart, tens_nirrep)]
        except KeyError:
            raise ValueError(f"can't infer Cartesian tensor irreps from the number " + \
                f"of Cartesian components = {tens_ncart} and number of irreps = {tens_nirrep}") from None

        # infer rank
        try:
            self.rank = ranks[tens_ncart]
        except KeyError:
            raise ValueError(f"can't infer rank of Cartesian tensor from the number " + \
                f"of Cartesian components = {tens_ncart}") from None


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
