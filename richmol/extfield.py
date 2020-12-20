import numpy as np
import rchm
import warnings
from scipy.sparse import coo_matrix, csr_matrix, kron
import sys
import itertools
import copy
import inspect

# allow for repetitions of warning for the same source location
warnings.simplefilter('always', UserWarning)

# to convert electric moments from atomic units to SI units
mu_au_to_Cm = 8.4783536255e-30              # dipole moment to C*m
alpha_au_to_C2m2_per_J = 1.64877727436e-41  # polarizability to C^2*m^2/J
beta_au_to_C3m3_per_J2 = 3.2063613061e-53   # hyperpolarizability to C^3*m^3/J^2
gamma_au_to_C4m4_per_J3 = 6.2353799905e-65  # second hyperpolarizability to C^4*m^4/J^3
theta_au_to_Cm2 = {1: 4.4865515246e-40}     # electric quadrupole to C*m^2

planck = 6.62607015e-34 # J/Hz
c_vac = 299792458 # m/s


class States():
    """Molecular field-free states.

    Args:
        filename : str
            Name of richmol HDF5 file.
        J_list : list
            List of quantum numbers of the total angular momentum spanned by basis.
    Kwargs:
        Additional set of parameters controlling the basis set.
        emin, emax : float
            Minimal and maximal energy of states.
        sym : list of str
            List of symmetries of states.
        m_list : list
            List of m quantum numbers.
        verbose : bool
            Set to True to print out some data.

    Attributes:
        enr : array(no_states)
            Basis state energies.
        id : array(no_states)
            State ID numbers.
        ideg : array(no_states)
            State degeneracy indices.
        sym : array(no_states)
            State symmetries.
        assign : array(no_states)
            State assignments.
        units : str
            Energy units.
        J_list : list
            List of J quanta spanned by basis.
        m_list : dict
            List of m quanta spanned by basis (dict.values) for each J (dict.keys).
        id_to_istate : array()
        mJ_to_im : array()
        dim_m : dict
        dim_k : dict
        prefac : int, float, or complex
    """

    def __init__(self, filename, tens_name, J_list, verbose=False, **kwargs):
        """Reads molecular field-free states from richmol HDF5 file and generates basis set indices.
        """

        if verbose == True:
            print(f"\nLoad states of {tens_name} from file {filename} ... ")

        self.prefac = 1
        self.enr = {}
        self.id = {}
        self.ideg = {}
        self.sym = {}
        self.assign = {}
        self.J_list = []

        maxid = {}

        for J_ in list(set(J_list)):

            J = round(float(J_), 1)

            # read states for fixed J

            enr, id, ideg, sym, assign, units = rchm.read_states(filename, tens_name, J)

            maxid[J] = max(id) # maximal state id number will be needed for mapping id -> istate

            # apply some filters

            if enr is None:
                raise Exception(f"File {filename} does not contain states' energies ('enr' dataset) for J = {J}") from None
            if id is None:
                raise Exception(f"File {filename} does not contain states' IDs ('id' dataset) for J = {J}") from None
            if ideg is None:
                warnings.warn(f"File {filename} does not contain states' degeneracy indices ('ideg' dataset) for J = {J}", stacklevel=2)
            if sym is None:
                warnings.warn(f"File {filename} does not contain states' symmetries ('sym' dataset) for J = {J}", stacklevel=2)
            if assign is None:
                warnings.warn(f"File {filename} does not contain states' assignments ('assign' dataset) for J = {J}", stacklevel=2)
            ind = [ i for i in range(len(enr))]
            if 'emin' in kwargs:
                ind = np.where(enr >= kwargs['emin'])
            if 'emax' in kwargs:
                ind = np.where(enr[ind] <= kwargs['emax'])
            if 'sym' in kwargs:
                ind = np.where(np.array(sym)[ind] in kwargs['sym'])
            if ind[0].shape[0] > 0:
                self.enr[J] = np.array(enr)[ind]
                self.id[J] = np.array(id)[ind]
                try:
                    self.ideg[J] = np.array(ideg)[ind]
                except IndexError:
                    pass
                try:
                    self.sym[J] = np.array(sym)[ind]
                except IndexError:
                    pass
                try:
                    self.assign[J] = np.array(assign)[ind]
                except IndexError:
                    pass
                self.J_list.append(J)

        self.units = units

        if len(self.J_list) == 0:
            raise Exception(f"State selection filters cast out all molecular states") from None

        J_out = [J for J in J_list if J not in self.J_list]
        if len(J_out) > 0:
            warnings.warn(f"State selection filters cast out all molecular states with J = {J_out}", stacklevel=2)

        # generate list of m quanta

        if 'm_list' in kwargs:
            m_list = list(set(round(float(elem), 1) for elem in kwargs['m_list']))
        else:
            maxJ = max(self.J_list)
            m_list = [round(float(m), 1) for m in np.linspace(-maxJ, maxJ, int(2*maxJ+1))]

        self.m_list = {}
        for J in self.J_list:
            self.m_list[J] = [round(float(m), 1) for m in np.linspace(-J, J, int(2*J+1)) if m in m_list]

        # print J and m quanta
        if verbose == True:
            print(f"energy units: {self.units}")
            print(f"list of J quanta spanned by basis: {self.J_list}")
            print(f"no. states and list of m quanta spanned by basis, for each J:")
            nonzero = False
            for J in self.J_list:
                print(f"    J = {J}, no. states = {len(self.enr[J])}, m = {[m for m in self.m_list[J]] if len(self.m_list[J]) > 0 else None}")

        J_out = [J for J in self.J_list if len(self.m_list[J]) == 0]
        if len(J_out) > 0:
            warnings.warn(f"m-quanta selection filters cast out all molecular states with J = {J_out}", stacklevel=2)
        self.J_list = [J for J in self.J_list if len(self.m_list[J]) > 0]
        self.J_pairs = [(J,J) for J in self.J_list]

        # mapping id -> istate
        self.id_to_istate = {}
        self.dim_k = {}
        for J in self.J_list:
            self.id_to_istate[J] = np.array([-1 for i in range(maxid[J]+1)], dtype=np.int64)
            for istate, id in enumerate(self.id[J]):
                self.id_to_istate[J][id] = istate
            self.dim_k[J] = max(self.id_to_istate[J]) + 1

        self.dim_k1 = self.dim_k
        self.dim_k2 = self.dim_k

        # mapping int(m + J) -> ind_m
        self.mJ_to_im = {}
        self.dim_m = {}
        for J in self.J_list:
            self.mJ_to_im[J] = np.array([-1 for i in range(int(2*J+1))], dtype=np.int64)
            for im, m in enumerate(self.m_list[J]):
                self.mJ_to_im[J][int(m+J)] = im
            self.dim_m[J] = max(self.mJ_to_im[J]) + 1

        self.dim_m1 = self.dim_m
        self.dim_m2 = self.dim_m

        # generate basis set indices

        for J in self.J_list:
            for m in self.m_list[J]:
                for istate in range(len(self.enr[J])):
                    pass
                    #print(J, m, istate, self.enr[J][istate])


    def mul(self, arg):
        scalar = (int, float, complex, np.int, np.int8, np.int16, np.int32, np.int64, np.float, \
                  np.float16, np.float32, np.float64, np.complex64, np.complex128)
        if isinstance(arg, scalar):
            self.prefac = self.prefac * arg
        else:
            raise TypeError(f"Unsupported argument type '{type(arg)}'") from None


    def tomat(self):
        """ Returns full diagonal matrix representation """
        mat = {}
        for J in self.J_list:
            diag = [enr * self.prefac for m in self.m_list[J] for enr in self.enr[J]]
            mat[(J,J)] = np.diag(diag)
        return {key : val * self.prefac for key,val in mat.items()}


    def vec(self, vec):
        """Computes product of diagonal matrix representation with vector.

        Vector 'vec' must be a dictionary with keys as J quanta (rounded to the first decimal)
        that are among those coupled by tensor. The values of vec must be numpy.ndarray arrays
        with dimension equal to the dimension of basis for selected J.
        """
        if not isinstance(vec, dict):
            raise TypeError(f"Unsupported argument type '{type(vec)}'") from None
        vec_new = {}
        for J in self.J_list:
            try:
                v = vec[J]
            except KeyError:
                continue
            diag = np.array([enr * self.prefac for m in self.m_list[J] for enr in self.id[J]])
            vec_new[J] = np.dot(diag, v)
        return {key : val * self.prefac for key,val in vec_new.items()}


    def __mul__(self, arg):
        scalar = (int, float, complex, np.int, np.int8, np.int16, np.int32, np.int64, np.float, \
                  np.float16, np.float32, np.float64, np.complex64, np.complex128)
        if isinstance(arg, scalar):
            # multiply with a scalar
            res = copy.deepcopy(self)
            res.mul(arg)
        elif isinstance(arg, dict):
            # multiply with wavepacket coefficient vector
            res = copy.deepcopy(self)
            res.vec(arg)
        else:
            raise TypeError(f"Unsupported operand type(s) for '*': '{self.__class__.__name__}' and " + \
                f"'{type(arg)}'") from None
        return res

    __rmul__ = __mul__



class Tensor():
    """Matrix elements of molecular laboratory-frame Cartesian tensor operator.

    Args:
        filename : str
            Name of richmol HDF5 file.
        tens_name : str
            String identifying tensor, as stored in the HDF5 file.
        states1 : States
            Field-free basis of bra functions.
        states2 : States
            Field-free basis of ket functions.
    Kwargs:
        verbose : bool
            Set to True to print out some data.

    Attributes:
        prefac : int, float, complex
        rank : int
            Tensor rank
        irreps : list
            List of tensor irreducible representations.
        cart : list
        units : str
        J_pairs : list
        dim_m1 : dict
        dim_k1 : dict
        dim_m2 : dict
        dim_k2 : dict
        mmat : dict
        kmat : dict
        mfmat : dict
    """
    def __init__(self, filename, tens_name, states1, states2, verbose=False, **kwargs):

        if verbose == True:
            print(f"\nLoad matrix elements for tensor {tens_name} from file {filename} ... ")

        self.prefac = 1.0

        # generate list of bra and ket J pairs that are coupled by tensor and spanned by basis set

        tens = rchm.inspect_tensors(filename)

        try:
            J_pairs = tens[tens_name]['J_pairs']
        except KeyError:
            raise KeyError(f"Can't find tensor {tens_name} in file {filename}, " + \
                f"here is by the way a list of all stored tensors: {[elem for elem in tens.keys()]}") from None

        self.J_pairs = [(J1, J2) for J1 in states1.J_list for J2 in states2.J_list \
                        if (J1, J2) in J_pairs or (J2, J1) in J_pairs]

        if len(self.J_pairs) == 0:
            raise Exception(f"Can't find any pair of J quanta that is spanned by bra and ket basis functions " + \
                +f"and coupled by tensor {tens_name} at the same time, reading file {filename}") from None

        for attr in ('rank', 'irreps', 'cart'):
            try:
                self.__dict__[attr] = tens[tens_name][attr]
            except KeyError:
                raise KeyError(f"Can't find tensor {tens_name}'s attribute = '{attr}' in file {filename}") from None
        try:
            self.units = tens[tens_name]['units']
        except KeyError:
            self.units = None

        self.dim_m1 = {J : states1.dim_m[J] for J in set(J_pair[0] for J_pair in self.J_pairs)}
        self.dim_k1 = {J : states1.dim_k[J] for J in set(J_pair[0] for J_pair in self.J_pairs)}
        self.dim_m2 = {J : states2.dim_m[J] for J in set(J_pair[1] for J_pair in self.J_pairs)}
        self.dim_k2 = {J : states2.dim_k[J] for J in set(J_pair[1] for J_pair in self.J_pairs)}

        if verbose == True:
            print(f"pairs of coupled J quanta: {self.J_pairs}")
            print(f"selection rules |J-J'|: {list(set(abs(J1 - J2) for (J1, J2) in self.J_pairs))}")
            print(f"rank: {self.rank}")
            print(f"Cartesian components ({len(self.cart)}): {self.cart}")
            print(f"irreps ({len(self.irreps)}): {self.irreps}")
            print(f"units: {self.units}")

        # read M-matrix

        self.mmat = {}

        for (J1, J2) in self.J_pairs:

            # read richmol file
            swapJ, mmat = rchm.read_mmat(filename, tens_name, J1, J2)

            m_ind1 = states1.mJ_to_im[J1]
            m_ind2 = states2.mJ_to_im[J2]
            dim1 = states1.dim_m[J1]
            dim2 = states2.dim_m[J2]
            J_pair = (J1, J2)

            if swapJ == True:
                m_ind1, m_ind2 = m_ind2, m_ind1
                dim1, dim2 = dim2, dim1

            self.mmat[J_pair] = []

            for m_cart in mmat:  # loop over Cartesian components of M-matrix
                cart_label, irreps, nnz, data, row, col = m_cart
                try:
                    cart_ind = tuple('xyz'.index(elem.lower()) for elem in cart_label)
                except ValueError:
                    raise ValueError(f"Illegal label for Cartesian component of tensor = '{cart_label}'") from None
                # split data into arrays for each irrep (opposite to np.concatenate)
                data = np.split(data, np.cumsum(nnz))[:-1]
                row = np.split(row, np.cumsum(nnz))[:-1]
                col = np.split(col, np.cumsum(nnz))[:-1]
                # select elements that are spanned by basis (eventually need to do it with numpy)
                data_ = [[d for d,r,c in zip(dd,rr,cc) if m_ind1[r]>=0 and m_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
                row_ = [[m_ind1[r] for d,r,c in zip(dd,rr,cc) if m_ind1[r]>=0 and m_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
                col_ = [[m_ind2[c] for d,r,c in zip(dd,rr,cc) if m_ind1[r]>=0 and m_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
                # store in sparse format
                if swapJ == True:
                    mat = [coo_matrix((d, (c, r)), shape=(dim2, dim1)).tocsr() for d,r,c in zip(data_,row_,col_)]
                    mat = np.conjugate(mat)
                else:
                    mat = [coo_matrix((d, (r, c)), shape=(dim1, dim2)).tocsr() for d,r,c in zip(data_,row_,col_)]
                self.mmat[J_pair].append((cart_ind, {irrep:m for irrep,m in zip(irreps,mat)}))

        # read K-matrix

        self.kmat = {}

        for (J1, J2) in self.J_pairs:

            # read richmol file
            swapJ, kmat = rchm.read_kmat(filename, tens_name, J1, J2)

            id_ind1 = states1.id_to_istate[J1]
            id_ind2 = states2.id_to_istate[J2]
            dim1 = states1.dim_k[J1]
            dim2 = states2.dim_k[J2]
            J_pair = (J1, J2)

            if swapJ == True:
                id_ind1, id_ind2 = id_ind2, id_ind1
                dim1, dim2 = dim2, dim1

            irreps, nnz, data, row, col = kmat
            # split data into arrays for each irrep (opposite to np.concatenate)
            data = np.split(data, np.cumsum(nnz))[:-1]
            row = np.split(row, np.cumsum(nnz))[:-1]
            col = np.split(col, np.cumsum(nnz))[:-1]
            # select elements that are spanned by basis (eventually need to do it with numpy)
            data_ = [[d for d,r,c in zip(dd,rr,cc) if id_ind1[r]>=0 and id_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
            row_ = [[id_ind1[r] for d,r,c in zip(dd,rr,cc) if id_ind1[r]>=0 and id_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
            col_ = [[id_ind2[c] for d,r,c in zip(dd,rr,cc) if id_ind1[r]>=0 and id_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
            # store in sparse format
            if swapJ == True:
                mat = [coo_matrix((d, (c, r)), shape=(dim2, dim1)).tocsr() for d,r,c in zip(data_,row_,col_)]
                mat = np.conjugate(mat)
            else:
                mat = [coo_matrix((d, (r, c)), shape=(dim1, dim2)).tocsr() for d,r,c in zip(data_,row_,col_)]
            self.kmat[J_pair] = {irrep:m for irrep,m in zip(irreps,mat)}


    def mul(self, arg):
        scalar = (int, float, complex, np.int, np.int8, np.int16, np.int32, np.int64, np.float, \
                  np.float16, np.float32, np.float64, np.complex64, np.complex128)
        if isinstance(arg, scalar):
            self.prefac = self.prefac * arg
        else:
            raise TypeError(f"Unsupported argument type '{type(arg)}'") from None


    def field(self, field, field_tol=1e-12):
        """ Multiplies tensor with field and contracts over Cartesian components """

        try:
            fx, fy, fz = field[:3]
            fxyz = np.array([fx, fy, fz])
        except IndexError:
            field_name = retrieve_name(field)
            raise IndexError(f"'{field_name}' must be an iterable with three items, which represent " + \
                f"field's X, Y, and Z components") from None

        # screen out small field components

        field_prod = {comb : np.prod(fxyz[list(comb)]) \
                      for comb in itertools.product((0,1,2), repeat=self.rank) \
                      if abs(np.prod(fxyz[list(comb)])) >= field_tol}

        # contract M-tensor with field

        self.mfmat = {}

        for (J1, J2), mmat in self.mmat.items():
            for (cart_ind, mat) in mmat:
                try:
                    fprod = field_prod[cart_ind]
                except KeyError:
                    continue     # this means the corresponding product of field components can be neglected
                for irrep, m in mat.items():
                    try:
                        self.mfmat[(J1,J2)][irrep] = self.mfmat[(J1,J2)][irrep] + m * fprod
                    except KeyError:
                        self.mfmat[(J1,J2)] = {irrep : m * fprod}


    def vec(self, vec):
        """Computes product of tensor with vector.

        Vector 'vec' must be a dictionary with keys as J quanta (rounded to the first decimal)
        that are among those coupled by tensor. The values of vec must be numpy.ndarray arrays
        with dimension equal to the dimension of basis for selected J.
        """
        try:
            x = self.mfmat
        except AttributeError:
            raise AttributeError("You need to multiply tensor with field before applying it to a vector") from None
        if not isinstance(vec, dict):
            raise TypeError(f"Unsupported argument type '{type(vec)}'") from None
        vec_new = {}
        for J_pair in list(set(self.mfmat.keys()) & set(self.kmat.keys())):
            mfmat = self.mfmat[J_pair]
            kmat = self.kmat[J_pair]
            J1, J2 = J_pair
            dim1 = self.dim_m2[J2]
            dim2 = self.dim_k2[J2]
            dim = self.dim_m1[J1] * self.dim_k1[J1]
            try:
                vecT = np.transpose(vec[J2].reshape(dim1, dim2))
            except KeyError:
                continue
            for irrep in list(set(mfmat.keys()) & set(kmat.keys())):
                tmat = csr_matrix.dot(kmat[irrep], vecT)
                v = csr_matrix.dot(mfmat[irrep], np.transpose(tmat))
                try:
                    vec_new[J1] = vec_new[J1] + csr_matrix.dot(mfmat[irrep], np.transpose(tmat)).reshape(dim)
                except KeyError:
                    vec_new[J1] = csr_matrix.dot(mfmat[irrep], np.transpose(tmat)).reshape(dim)
        return {key : val * self.prefac for key,val in vec_new.items()}


    def tomat(self, **kwargs):
        """Returns dense-matrix representation of tensor contracted with external field, or of its
        selected Cartesian component.
        """
        if 'cart' in kwargs:
            cart = kwargs['cart']
            ifcart = True
            try:
                cart_ind = tuple('xyz'.index(elem.lower()) for elem in cart)
            except ValueError:
                raise ValueError(f"Illegal label for Cartesian component of tensor = '{cart}'") from None
        else:
            ifcart = False
            try:
                x = self.mfmat
            except AttributeError:
                raise AttributeError(f"You need to multiply tensor with field before computing " + \
                    f"its matrix representation, or provide as argument a label for selected " + \
                    f"Cartesian component of tensor") from None

        mat = {}
        for J_pair in list(set(self.mmat.keys()) & set(self.kmat.keys())):
            J1, J2 = J_pair
            if ifcart == True:
                try:
                    ind = [elem[0] for elem in self.mmat[J_pair]].index(cart_ind)
                except ValueError:
                    print(f"Note: matrix elements <J={J1}|{retrieve_name(self)}({cart})|J={J2}> = 0")
                    continue
                mmat = self.mmat[J_pair][ind][1]
            else:
                mmat = self.mfmat[J_pair]
            kmat = self.kmat[J_pair]
            for irrep in list(set(mmat.keys()) & set(kmat.keys())):
                try:
                    mat[J_pair] = mat[J_pair] + kron(mmat[irrep], kmat[irrep]).todense()
                except KeyError:
                    mat[J_pair] = kron(mmat[irrep], kmat[irrep]).todense()
        return {key : val * self.prefac for key,val in mat.items()}


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
            raise TypeError(f"Unsupported operand type(s) for '*': '{self.__class__.__name__}' and " + \
                f"'{type(arg)}'") from None
        return res

    __rmul__ = __mul__



class Hamiltonian():
    """ Collects sum of tensors """

    def __init__(self, **kwargs):
        self.tensors = {}
        for key,val in kwargs.items():
            if isinstance(val, (Tensor, States)):
                self.tensors[key] = val
            else:
                raise TypeError(f"Unsupported '{key}' argument type '{type(val)}'") from None

        for key,tens in self.tensors.items():
            if isinstance(tens, Tensor):
                try:
                    x = tens.mfmat
                except AttributeError:
                    raise AttributeError(f"You need to multiply tensor '{key}' with field before " + \
                        f"passing it to Hamiltonian") from None

        # check consistency of dimensions
        dim_m1 = {}
        dim_m2 = {}
        dim_k1 = {}
        dim_k2 = {}
        past_keys = []
        for key,tens in self.tensors.items():
            for J_pair in tens.J_pairs:
                J1, J2 = J_pair
                # check m-dimensions
                dim1 = tens.dim_m1[J1]
                dim2 = tens.dim_m2[J2]
                try:
                    if (dim1, dim2) != (dim_m1[J1], dim_m2[J2]):
                        raise ValueError(f"Shape of M-part of tensor '{key}' = {(dim1, dim2)} " + \
                            f"is not aligned with the shape of tensor(s) {past_keys} = " + \
                            f"{(dim_m1[J1], dim_m2[J2])}, for J1, J2 = {J1}, {J2}") from None
                except KeyError:
                    dim_m1[J1] = dim1
                    dim_m2[J2] = dim2
                # check k-dimensions
                dim1 = tens.dim_k1[J1]
                dim2 = tens.dim_k2[J2]
                try:
                    if (dim1, dim2) != (dim_k1[J1], dim_k2[J2]):
                        raise ValueError(f"Shape of K-part of tensor '{key}' = {(dim1, dim2)} " + \
                            f"is not aligned with the shape of tensor(s) {past_keys} = " + \
                            f"{(dim_k1[J1], dim_k2[J2])}, for J1, J2 = {J1}, {J2}") from None
                except KeyError:
                    dim_k1[J1] = dim1
                    dim_k2[J2] = dim2
            past_keys.append(key)



    def tomat(self):
        res = {}
        for key,tens in self.tensors.items():
            mat = tens.tomat()
            for J_pair,elem in mat.items():
                try:
                    res[J_pair] = res[J_pair] + elem
                except KeyError:
                    res[J_pair] = elem
        return res


    def vec(self, vec):
        vec_new = {}
        for key,tens in self.tensors.items():
            v = tens.vec(vec)
            for J,elem in v.items():
                try:
                    vec_new[J] = vec_new[J] + elem
                except KeyError:
                    vec_new[J] = elem
        return vec_new



def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]



if __name__ == '__main__':

    filename = '../examples/watie/OCS_j0_j30.h5'

    # list tensors stored in file filename
    tens = rchm.inspect_tensors(filename)
    for key in tens.keys():
        print("\n", key, tens[key])

    # field-free basis
    states = States(filename, 'h0', [J for J in range(31)], emin=0, emax=10000, sym=['A'], verbose=True)

    # dipole matrix elements
    mu = Tensor(filename, 'mu', states, states, verbose=True)
    mu.mul(-1.0)

    # external field in V/m
    field = [0,0,1000 * 100]

    # multiply dipole with external field
    mu.field(field)

    # convert dipole[au]*field[V/m] into [cm^-1]
    fac = mu_au_to_Cm / (planck * c_vac) / 100
    mu.mul(fac)

    # mat = mu.tomat()
    # print(mat)
    h = Hamiltonian(mu=mu, h0=states)
    mat = h.tomat()
    #print(mat)
