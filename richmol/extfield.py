import numpy as np
import rchm
import warnings
from scipy.sparse import coo_matrix
import sys
import itertools

# allow for repetitions of warning for the same source location
warnings.simplefilter('always', UserWarning)


class States():
    """Molecular field-free states.

    Args:
        filename : str
            Name of richmol HDF5 file, to read the field-free states from.
        J_list : list
            List of quantum numbers of the total angular momentum spanned by the basis.
    Kwargs:
        emin, emax : float
            Minimal and maximal energy of states included in the basis.
        sym : list of str
            List of basis symmetries of states included in the basis
        m_list : list
            List of m quantum numbers spanned by the basis.
        verbose : bool
            Set to True to print out some data.

    Attributes:
        enr
        id
        ideg
        sym
        assign
        J_list
        m_list
        id_to_istate
        mJ_to_im
    """

    def __init__(self, filename, J_list, verbose=False, **kwargs):
        """Reads molecular field-free states from richmol HDF5 file, and generates basis set indices.
        """

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

            descr, dim, enr, id, ideg, sym, assign = rchm.read_states(filename, J)

            maxid[J] = max(id) # maximal state id number will be needed for mapping id -> istate

            # apply some filters

            if enr is None:
                raise Exception(f"File {filename} does not contain states' energies ('enr' dataset) for J = {J}")
            if id is None:
                raise Exception(f"File {filename} does not contain states' IDs ('id' dataset) for J = {J}")
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
            print(f"List of J quanta spanned by basis: {self.J_list}")
            print(f"No. states and list of m quanta spanned by basis, for each J:")
            nonzero = False
            for J in self.J_list:
                print(f"    J = {J}, no. states = {len(self.enr[J])}, m = {[m for m in self.m_list[J]] if len(m_list) > 0 else None}")

        J_out = [J for J in self.J_list if len(self.m_list[J]) == 0]
        if len(J_out) > 0:
            warnings.warn(f"m-quanta selection filters cast out all molecular states with J = {J_out}", stacklevel=2)
        self.J_list = [J for J in self.J_list if len(self.m_list[J]) > 0]

        # mapping id -> istate
        self.id_to_istate = {}
        self.dim_k = {}
        for J in self.J_list:
            self.id_to_istate[J] = np.array([-1 for i in range(maxid[J]+1)], dtype=np.int64)
            for istate, id in enumerate(self.id[J]):
                self.id_to_istate[J][id] = istate
            self.dim_k[J] = max(self.id_to_istate[J]) + 1

        # mapping int(m + J) -> ind_m
        self.mJ_to_im = {}
        self.dim_m = {}
        for J in self.J_list:
            self.mJ_to_im[J] = np.array([-1 for i in range(int(2*J+1))], dtype=np.int64)
            for im, m in enumerate(self.m_list[J]):
                self.mJ_to_im[J][int(m+J)] = im
            self.dim_m[J] = max(self.mJ_to_im[J]) + 1

        # generate basis set indices

        for J in self.J_list:
            for m in self.m_list[J]:
                for istate in range(len(self.enr[J])):
                    pass
                    #print(J, m, istate, self.enr[J][istate])



class Tensor():
    """Matrix elements of molecular laboratory-frame Cartesian tensor operator.

    Args:
        filename : str
            Name of richmol HDF5 file.
        tens_name : str
            String identifying tensor, as stored in the HDF5 file.
        states : States
            Field-free basis.
    Kwargs:
        verbose : bool
            Set to True to print out some data.

    Attributes:
        rank : int
            Tensor rank
        irreps : list
            List of tensor irreducible representations.
        Jpairs : list
        mmat : dict
        kmat : dict
    """
    def __init__(self, filename, tens_name, states, verbose=False, **kwargs):

        if verbose == True:
            print(f"\nLoad matrix elements for tensor {tens_name} ...")

        # generate list of bra and ket J pairs that are coupled by tensor and spanned by basis set

        tens = rchm.inspect_tensors(filename)

        try:
            J_pairs = tens[tens_name]
        except KeyError:
            raise KeyError(f"Can't find tensor {tens_name} in file {filename}, " + \
                f"here is by the way a list of all stored tensors: {[elem for elem in tens.keys()]}") from None

        self.J_pairs = [(J1, J2) for J1 in states.J_list for J2 in states.J_list \
                        if (J1, J2) in J_pairs or (J2, J1) in J_pairs]

        if len(self.J_pairs) == 0:
            raise Exception(f"Can't find any pair of J quanta that is spanned by basis and coupled " + \
                +f"by tensor {tens_name} at the same time, reading file {filename}") from None

        if verbose == True:
            print(f"pairs of coupled J quanta: {self.J_pairs}")
            print(f"selection rules |J-J'|: {list(set(abs(J1 - J2) for (J1, J2) in self.J_pairs))}")

        self.rank = []
        self.irreps = []

        # read M-matrix

        self.mmat = {}

        for (J1, J2) in self.J_pairs:

            # read richmol file
            swapJ, mmat = rchm.read_mmat(filename, tens_name, J1, J2)

            m_ind1 = states.mJ_to_im[J1] # m_ind[int(m+J)] gives the index of m quantum number in the m-subset of basis
            m_ind2 = states.mJ_to_im[J2] # ... or -1 if m is not contained in the m-subset
            dim1 = states.dim_m[J1] # dimension of m-subset for J1
            dim2 = states.dim_m[J2] # dimension of m-subset for J2
            Jpair = (J1, J2)

            if swapJ == True:
                m_ind1, m_ind2 = m_ind2, m_ind1
                dim1, dim2 = dim2, dim1

            self.mmat[Jpair] = []

            for m_cart in mmat:  # loop over Cartesian components of M-matrix
                cart_label, irreps, nnz, data, row, col = m_cart
                try:
                    cart_ind = tuple('xyz'.index(elem.lower()) for elem in cart_label)
                except ValueError:
                    raise ValueError(f"Illegal Cartesian label = '{cart_label}'") from None
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
                self.mmat[Jpair].append((cart_ind, {irrep:m for irrep,m in zip(irreps,mat)}))

            # update rank and irreps
            self.rank = set(list(self.rank) + [len(elem[0]) for elem in self.mmat[Jpair]])
            self.irreps = set(list(self.irreps) + [irrep for elem in self.mmat[Jpair] for irrep in elem[1].keys()])

        if len(self.rank) != 1:
            raise Exception(f"Multiple values of tensor rank = {list(rank)} across different J-quanta") from None

        self.rank = list(self.rank)[0]
        if verbose == True:
            print(f"tensor rank: {self.rank}")

        # read K-matrix

        self.kmat = {}

        for (J1, J2) in self.J_pairs:

            # read richmol file
            swapJ, kmat = rchm.read_kmat(filename, tens_name, J1, J2)

            id_ind1 = states.id_to_istate[J1] # id_ind[id] gives the index of state in the k-subset of basis
            id_ind2 = states.id_to_istate[J2] # ... or -1 if id is not contained in the k-subset
            dim1 = states.dim_k[J1] # dimension of k-subset for J1
            dim2 = states.dim_k[J2] # dimension of k-subset for J2
            Jpair = (J1, J2)

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
            self.kmat[Jpair] = {irrep:m for irrep,m in zip(irreps,mat)}

            # collect irreps
            self.irreps = set(list(self.irreps) + [irrep for irrep in self.kmat[Jpair].keys()])

        if verbose == True:
            print(f"tensor irreps: {self.irreps}")


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
        """ Computes product Tensor * vec = (MF x K) * vec, where 'x' and '*' are tensor and dot products.
        """
        for Jpair in self.Jpairs:
            try:
                self.mfmat[Jpair]
                fac = 1.0
            except KeyError:
                self.mfmat[(Jpair[1], Jpair[0])]
            dim1 = self.dim_m[J2]
            dim2 = self.dim_k[J2]
            dim = self.dim_m[J1] * self.dim_k[J1]
            vecT = np.transpose(vec[J2].reshape(dim1, dim2))
            pass


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


if __name__ == '__main__':

    filename = '../examples/watie/OCS_energies_j0_j10.h5'
    a = States(filename, [i for i in range(11)], emin=0, emax=1000, sym=['A'], m_list=[i for i in range(-10,2000)], verbose=True)
    mu = Tensor(filename, 'alpha', a, verbose=True)
    field = [2,0,0.5]
    mu.field(field)

