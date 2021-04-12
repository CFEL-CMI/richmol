import numpy as np
from scipy.sparse import kron, csr_matrix
import itertools
from itertools import chain
import copy


class CarTens():
    """General class for laboratory-frame Cartesian tensor operators
    """

    prefac = 1.0

    def __init__(self):
        pass


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
