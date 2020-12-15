import numpy as np
import rchm
import warnings
from scipy.sparse import coo_matrix
import sys

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
        for J in self.J_list:
            self.id_to_istate[J] = np.array([-1 for i in range(maxid[J]+1)], dtype=np.int64)
            for istate, id in enumerate(self.id[J]):
                self.id_to_istate[J][id] = istate

        # mapping int(m + J) -> ind_m
        self.mJ_to_im = {}
        for J in self.J_list:
            self.mJ_to_im[J] = np.array([-1 for i in range(int(2*J+1))], dtype=np.int64)
            for im, m in enumerate(self.m_list[J]):
                self.mJ_to_im[J][int(m+J)] = im

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
                f"list of stored tensors: {[elem for elem in tens.keys()]}") from None

        J_pairs = [(J1, J2) for J1 in states.J_list for J2 in states.J_list \
                    if (J1, J2) in J_pairs or (J2, J1) in J_pairs]

        if len(J_pairs) == 0:
            raise Exception(f"Can't find any of the pairs of J quanta spanned by the basis in file " + \
                +f"{filename} for tensor {tens_name}") from None

        if verbose == True:
            print(f"pairs of coupled J quanta: {J_pairs}")
            print(f"selection rules |J-J'|: {list(set(abs(J1 - J2) for (J1, J2) in J_pairs))}")


        rank = []
        irreps = []

        # read M-matrix

        self.mmat = {}

        for (J1, J2) in J_pairs:

            if (J1, J2) in self.mmat:
                continue

            swapJ, mmat = rchm.read_mmat(filename, tens_name, J1, J2)

            # set of m-quanta indices spanned by basis for J1 and J2
            m_ind1 = states.mJ_to_im[J1]
            m_ind2 = states.mJ_to_im[J2]
            dim1 = len([im for im in m_ind1 if im>=0])
            dim2 = len([im for im in m_ind2 if im>=0])
            Jpair = (J1, J2)

            if swapJ == True:
                m_ind1, m_ind2 = m_ind2, m_ind1
                dim1, dim2 = dim2, dim1
                Jpair = (J2, J1)

            self.mmat[Jpair] = []

            for m_cart in mmat:
                cart_label, numtype, irreps, nnz, data, row, col = m_cart
                cart_ind = ['xyz'.index(elem.lower()) for elem in cart_label] # for cart_label = 'xyzx' gives cart_ind = [0,1,2,0]
                # split data into arrays for each irrep (opposite to np.concatenate)
                data = np.split(data, np.cumsum(nnz))[:-1]
                row = np.split(row, np.cumsum(nnz))[:-1]
                col = np.split(col, np.cumsum(nnz))[:-1]
                # select elements that are spanned by basis (eventually need to do it with numpy)
                data_ = [[d for d,r,c in zip(dd,rr,cc) if m_ind1[r]>=0 and m_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
                row_ = [[m_ind1[r] for d,r,c in zip(dd,rr,cc) if m_ind1[r]>=0 and m_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
                col_ = [[m_ind2[c] for d,r,c in zip(dd,rr,cc) if m_ind1[r]>=0 and m_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
                # store in sparse format
                mat = [coo_matrix((d, (r, c)), shape=(dim1, dim2), dtype=np.float64).tocsr() for d,r,c in zip(data_,row_,col_)]
                self.mmat[Jpair].append((cart_ind, numtype, irreps, mat))

            rank = set(list(rank) + [len(elem[0]) for elem in self.mmat[Jpair]])
            irreps = set(list(irreps) + [irrep for elem in self.mmat[Jpair] for irrep in elem[2]])

        if len(rank) != 1:
            raise Exception(f"Inconsistent rank of tensor {tens_name} across different J-quanta: " + \
                f"rank = {list(rank)}, file = {filename}") from None

        self.rank = list(rank)[0]
        if verbose == True:
            print(f"tensor rank: {self.rank}")

        # read K-matrix

        self.kmat = {}

        for (J1, J2) in J_pairs:

            if (J1, J2) in self.kmat:
                continue

            swapJ, kmat = rchm.read_kmat(filename, tens_name, J1, J2)

            # set of id indices spanned by basis for J1 and J2
            id_ind1 = states.id_to_istate[J1]
            id_ind2 = states.id_to_istate[J2]
            dim1 = len([id for id in id_ind1 if id>=0])
            dim2 = len([id for id in id_ind2 if id>=0])
            Jpair = (J1, J2)

            if swapJ == True:
                id_ind1, id_ind2 = id_ind2, id_ind1
                dim1, dim2 = dim2, dim1
                Jpair = (J2, J1)

            self.kmat[Jpair] = []

            numtype, irreps, nnz, data, row, col = kmat
            # split data into arrays for each irrep (opposite to np.concatenate)
            data = np.split(data, np.cumsum(nnz))[:-1]
            row = np.split(row, np.cumsum(nnz))[:-1]
            col = np.split(col, np.cumsum(nnz))[:-1]
            # select elements that are spanned by basis (eventually need to do it with numpy)
            data_ = [[d for d,r,c in zip(dd,rr,cc) if id_ind1[r]>=0 and id_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
            row_ = [[id_ind1[r] for d,r,c in zip(dd,rr,cc) if id_ind1[r]>=0 and id_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
            col_ = [[id_ind2[c] for d,r,c in zip(dd,rr,cc) if id_ind1[r]>=0 and id_ind2[c]>=0] for dd,rr,cc in zip(data,row,col)]
            # store in sparse format
            mat = [coo_matrix((d, (r, c)), shape=(dim1, dim2), dtype=np.float64).tocsr() for d,r,c in zip(data_,row_,col_)]
            self.kmat[Jpair].append((numtype, irreps, mat))

            irreps = set(list(irreps) + [irrep for elem in self.kmat[Jpair] for irrep in elem[1]])

        self.irreps = list(irreps)

        if verbose == True:
            print(f"tensor irreps: {self.irreps}")


    def __mul__(self, arg):
        """
        """
        if isinstance(arg, (np.ndarray, list, tuple)):

            # multiply tensor with field and contract over Cartesian components

            try:
                fx, fy, fz = arg[:3]
                field = np.array([fx, fy, fz])
            except IndexError:
                arg_name = retrieve_name(arg)
                raise IndexError(f"'{arg_name}' must be an iterable with three items, which represent " + \
                    f"field's X, Y, and Z components") from None

            # pre-screen small field components
            # ....

            # dtype = {'real' : np.float64, 'imag' : np.float64, 'cmplx' : np.complex128}
            # numfac = {'real' : 1.0, 'imag' : 1j, 'cmplx' : 1.0}

            # for (J1, J2), mmat in self.mmat.items():
            #     # kmat_irreps = self.kmat[]
            #     print(J1, J2, '-------------------')
            #     for cart in mmat:
            #         cart_label = cart[0]
            #         icart = ['xyz'.index(elem.lower()) for elem in cart_label]
            #         field_prod = np.prod(field[icart])
            #         numtype = cart[1]
            #         irreps = cart[2]
            #         nnz = cart[3]
            #         data = cart[4] * field_prod
            #         row = cart[5]
            #         col = cart[6]
            #         data = np.split(data, np.cumsum(nnz))[:-1]
            #         row = np.split(row, np.cumsum(nnz))[:-1]
            #         col = np.split(col, np.cumsum(nnz))[:-1]
            #         mat = [coo_matrix((d, r, c), dtype=dtype[numtype]) for d, r, c in zip(data, row, col)]
            #         print(nnz, np.cumsum(nnz), irreps, len(data), len(col))





if __name__ == '__main__':

    filename = '../examples/watie/OCS_energies_j0_j10.h5'
    a = States(filename, [i for i in range(11)], emin=0, emax=1000, sym=['A'], m_list=[i for i in range(-5,1)], verbose=True)
    mu = Tensor(filename, 'alpha', a, verbose=True)
    field = [2,1,0.5]
    a = mu * field

