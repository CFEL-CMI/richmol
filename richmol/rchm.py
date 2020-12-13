import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import re
import sys

"""
Data structure of richmol HDF5 file

  /J:<J1>,<J2>                              (J1 bra and J2 ket total angular momentum quantum numbers)
   |
   |---/descr                               (description of data)
   |---/enr                                 (state energies, for J1 == J2)
   |---/sym                                 (state symmetries, for J1 == J2)
   |---/assign                              (state assignments, for J1 == J2)
   |---/id                                  (state IDs, for J1 == J2)
   |---/dim                                 ((dim1, dim2) - number of states for J1 and J2)
   |---/tens:<tens>                         (matrix element data for specific tensor)
        |
        |---/kmat_data                      (vector of non-zero elements of K-matrix, concatenated across all irreducible representations)
        |---/kmat_data.numtype              (type of elements, 'real', 'imag', or 'cmplx')
        |---/kmat_data.irreps               (list of irreducible representations)
        |---/kmat_data.nnz                  (number of elements in kmat_data for each irreducible representation)
        |---/kmat_row                       (row indices of non-zero elements)
        |---/kmat_col                       (column indices of non-zero elements)
        |
        |---/mmat_<cart>_data               (...)
        |---/mmat_<cart>_data.numtype
        |---/mmat_<cart>_data.irreps
        |---/mmat_<cart>_data.nnz
        |---/mmat_<cart>_data.cart
        |---/mmat_<cart>_row
        |---/mmat_<cart>_col

 <J1> and <J2> (float): bra and ket quantum numbers of the total angular momentum
    operator, rounded to first decimal, e.g., 1.0 and 2.0 or 1.5 and 2.5
 <tens> (str): name of tensor operator, e.g., 'alpha' for polarizability
 <irrep> (int): index of irreducible representation, e.g., 0 and 2 for polarizability
 <cart> (str): cartesian component of tensor, e.g., 'xx', 'xy', 'xz', 'yx', .. for polarizability
"""

def J_group_key(J1, J2):
    return 'J:' + str(round(float(J1), 1)) + "," + str(round(float(J2), 1))

def tens_group_key(tens):
    return 'tens:' + tens

# def irrep_group_key(irrep):
#     return 'irrep:' + str(irrep)


def store(filename, J1, J2, replace=False, thresh=1e-14, **kwargs):
    """Stores field-free state energies, assignments, and matrix elements
    of Cartesian tensor operators in richmol HDF5 file.

    Args:
        filename : str
            Name of the output HDF5 file, if file exists it will be appended,
            it can be overwritten by passing replace=True.
        J1, J2 : int or half-integer float
            Values of J quantum numbers for bra (J1) and ket (J2) states.
        replace : bool
            If True, the existing HDF5 file will be overwritten.

    Kwargs:
        descr : str or list of str
            Description of the data.
        enr : array (no_states) of float
            State energies, no_states is the number of states for J = J1 == J2.
        sym : array (no_states) of str
            State symmetries.
        assign : array (no_states) of str
            State assignments.
        id : array (no_states, 2) of int
            State ID numbers [:,0] and degenerate component indices [:,1].
        tens : str
            Name of Cartesian tensor operator.
        irreps : list of int
            List of tensor irreducible representations.
        cart : str
            Matrix elements for different Cartesian components of a tensor
            are stored in separate calls. Here 'cart' is used to label the Cartesian
            component which is currently stored.
        kmat : list of coo_matrix(no_states(J1), no_states(J2))
            List of K-matrices for different irreducible representations
            of tensor, in the same order as in 'irreps'.
        mmat : list of coo_matrix(2*J1+1, 2*J2+1)
            List of M-matrices for different irreducible representations
            of tensor, in the same order as in 'irreps'.
            Only single Cartesian component of M-matrix (labelled by 'cart')
            can be stored at a call.
        thresh : float
            Threshold for neglecting matrix elements.
    """
    def _J1_eq_J2(argname):
        # check if J1 == J2
        assert (round(float(J1), 1) == round(float(J2), 1)), \
            f"Bra and ket J quanta = {J1} and {J2} must be equal " + \
            f"to each other when storing {argname}"

    def _check_dim(dim, argname):
        # check if dimensions in dim match those for J1_J2 group
        try:
            dim_ = J_grp['dim'][()]
            assert (all(dim == dim_)), \
                f"Dimensions of '{argname}' = {dim} do not match " + \
                f"basis set dimensions = {dim_} for bra and ket J quanta = {J1}, {J2} " + \
                f"already stored in file {filename}"
        except KeyError:
            dset = J_grp.create_dataset('dim', data=dim)
        # check if dimensions in dim[0] and dim[1] match those for J1_J1 and J2_J2 groups
        if round(float(J1), 1) != round(float(J2), 1):
            key1 = J_group_key(J1, J1)
            key2 = J_group_key(J2, J2)
            try:
                dim_ = fl[key1+'/dim'][()]
                assert (dim[0] == dim_[0]), \
                    f"First dimension of '{argname}' = {dim[0]} does not match " + \
                    f"basis set dimensions = {dim_[0]} for bra J = {J1} " + \
                    f"already stored in file {filename}"
            except KeyError:
                dset = fl.create_dataset(key1+'/dim', data=[dim[0], dim[0]])
            try:
                dim_ = fl[key2+'/dim'][()]
                assert (dim[1] == dim_[0]), \
                    f"Second dimension of '{argname}' = {dim[1]} does not match " + \
                    f"basis set dimensions = {dim_[0]} for ket J = {J2} " + \
                    f"already stored in file {filename}"
            except KeyError:
                dset = fl.create_dataset(key2+'/dim', data=[dim[1], dim[1]])

    fl = h5py.File(filename, {True:"w", False:"a"}[replace])

    key = J_group_key(J1, J2)
    if key not in fl:
        J_grp = fl.create_group(key)
    else:
        J_grp = fl[key]

    # store metadata

    try:
        descr = kwargs['descr']
        if 'descr' in fl:
            del fl['descr']
        if isinstance(descr, str):
            data = descr
        elif all(isinstance(elem, str) for elem in descr):
            data = " ".join(elem for elem in descr)
        else:
            raise TypeError(f"bad argument type for 'descr'") from None
        dset = fl.create_dataset('descr', data=[str.encode(elem) for elem in data])
    except KeyError:
        pass

    # store energy

    try:
        enr = kwargs['enr']
        _J1_eq_J2("state energies")
        _check_dim((len(enr), len(enr)), 'enr')
        if 'enr' in J_grp:
            del J_grp['enr']
        dset = J_grp.create_dataset('enr', data=enr)
    except KeyError:
        pass

    # store assignment

    try:
        assign = kwargs['assign']
        _J1_eq_J2("state assignments")
        _check_dim((len(assign), len(assign)), 'assign')
        if 'assign' in J_grp:
            del J_grp['assign']
        dset = J_grp.create_dataset('assign', data=[str.encode(elem) for elem in assign])
    except KeyError:
        pass

    # store IDs

    try:
        id = kwargs['id']
        _J1_eq_J2("state IDs")
        _check_dim((len(id), len(id)), 'id')
        if 'id' in J_grp:
            del J_grp['id']
        dset = J_grp.create_dataset('id', data=id)
    except KeyError:
        pass

    # store symmetries

    try:
        sym = kwargs['sym']
        _J1_eq_J2("state symmetries")
        _check_dim((len(sym), len(sym)), 'sym')
        if 'sym' in J_grp:
            del J_grp['sym']
        dset = J_grp.create_dataset('sym', data=[str.encode(elem) for elem in sym])
    except KeyError:
        pass

    # store K-matrix

    try:
        kmat = kwargs['kmat']
        for k_irrep in kmat:
            _check_dim(k_irrep.shape, 'kmat')

        if 'irreps' not in kwargs:
            raise Exception(f"'irreps' argument must be passed together with 'kmat'") from None
        else:
            irreps = kwargs['irreps']
            assert (len(irreps) == len(kmat)), \
                f"Shapes of 'irreps' = {len(irreps)} and 'kmat' = {len(kmat)} are not aligned"

        k_data = None
        k_row = None
        k_col = None
        k_nnz = []
        k_irrep = []
        numtype = []

        for km, irrep in zip(kmat, irreps):

            data = km.data
            row = km.row
            col = km.col

            if len(data) == 0:
                continue
            if np.any(np.abs(data.real) >= thresh) and np.all(np.abs(data.imag) < thresh):
                numtype.append('real')
                data = data.real
            elif np.all(np.abs(data.real) < thresh) and np.any(np.abs(data.imag) >= thresh):
                numtype.append('imag')
                data = data.imag
            elif np.all(np.abs(data.real) < thresh) and np.all(np.abs(data.imag) < thresh):
                continue
            else:
                numtype.append('cmplx')

            k_nnz.append(len(data))
            k_irrep.append(irrep)

            if k_data is None:
                k_data = data
                k_row = row
                k_col = col
            else:
                k_data = np.concatenate((k_data, data))
                k_row = np.concatenate((k_row, row))
                k_col = np.concatenate((k_col, col))

        if len(set(numtype)) > 1:
            raise ValueError(f"K-matrix elements for different irreps = {k_irrep} " + \
                f"have different numerical types = {numtype}, for J1, J2 = {J1}, {J2}") from None

        if k_data is not None:

            if 'tens' not in kwargs:
                raise Exception(f"'tens' argument must be passed together with 'kmat'") from None
            else:
                tens = kwargs['tens']
            if tens_group_key(tens) not in J_grp:
                tens_grp = J_grp.create_group(tens_group_key(tens))
            else:
                tens_grp = J_grp[tens_group_key(tens)]

            if 'kmat_data' in tens_grp:
                del tens_grp['kmat_data']
            if 'kmat_row' in tens_grp:
                del tens_grp['kmat_row']
            if 'kmat_col' in tens_grp:
                del tens_grp['kmat_col']

            dset = tens_grp.create_dataset('kmat_data', data=k_data)
            dset.attrs['numtype'] = list(set(numtype))[0]
            dset.attrs['irreps'] = k_irrep
            dset.attrs['nnz'] = k_nnz
            dset = tens_grp.create_dataset('kmat_row', data=k_row)
            dset = tens_grp.create_dataset('kmat_col', data=k_col)

    except KeyError:
        pass

    # store M-matrix

    try:
        mmat = kwargs['mmat']

        for irrep, m_irrep in enumerate(mmat):
            assert (m_irrep.shape == (int(2*J1)+1, int(2*J2)+1)), \
                f"Shapes of {irrep}-th element of 'mmat' = {m_irrep.shape} " + \
                f"and (2*J1+1, 2*J2+1) = {int(2*J1)+1, int(2*J2)+1} are not aligned" + \
                f"for J1, J2 = {J1}, {J2}"

        if 'cart' not in kwargs:
            raise Exception(f"'cart' argument must be passed together with 'mmat'") from None
        else:
            cart = kwargs['cart']

        if 'irreps' not in kwargs:
            raise Exception(f"'irreps' argument must be passed together with 'mmat'") from None
        else:
            irreps = kwargs['irreps']
            assert (len(irreps) == len(mmat)), \
                f"Shapes of 'irreps' = {len(irreps)} and 'mmat' = {len(mmat)} are not aligned"

        m_data = None
        m_row = None
        m_col = None
        m_nnz = []
        m_irrep = []
        numtype = []

        for mm, irrep in zip(mmat, irreps):

            data = mm.data
            row = mm.row
            col = mm.col

            if len(data) == 0:
                continue
            if np.any(np.abs(data.real) >= thresh) and np.all(np.abs(data.imag) < thresh):
                numtype.append('real')
                data = data.real
            elif np.all(np.abs(data.real) < thresh) and np.any(np.abs(data.imag) >= thresh):
                numtype.append('imag')
                data = data.imag
            elif np.all(np.abs(data.real) < thresh) and np.all(np.abs(data.imag) < thresh):
                continue
            else:
                numtype.append('cmplx')

            m_nnz.append(len(data))
            m_irrep.append(irrep)

            if m_data is None:
                m_data = data
                m_row = row
                m_col = col
            else:
                m_data = np.concatenate((m_data, data))
                m_row = np.concatenate((m_row, row))
                m_col = np.concatenate((m_col, col))

        if len(set(numtype)) > 1:
            raise ValueError(f"M-matrix elements for different irreps = {m_irrep} " + \
                f"have different numerical types = {numtype}, for J1, J2 = {J1}, {J2}") from None

        if m_data is not None:

            if 'tens' not in kwargs:
                raise Exception(f"'tens' argument must be passed together with 'mmat'") from None
            else:
                tens = kwargs['tens']
            if tens_group_key(tens) not in J_grp:
                tens_grp = J_grp.create_group(tens_group_key(tens))
            else:
                tens_grp = J_grp[tens_group_key(tens)]

            if 'mmat_'+cart+'_data' in tens_grp:
                del tens_grp['mmat_'+cart+'_data']
            if 'mmat_'+cart+'_row' in tens_grp:
                del tens_grp['mmat_'+cart+'_row']
            if 'mmat_'+cart+'_col' in tens_grp:
                del tens_grp['mmat_'+cart+'_col']

            dset = tens_grp.create_dataset('mmat_'+cart+'_data', data=m_data)
            dset.attrs['numtype'] = list(set(numtype))[0]
            dset.attrs['irreps'] = m_irrep
            dset.attrs['nnz'] = m_nnz
            dset.attrs['cart'] = cart
            dset = tens_grp.create_dataset('mmat_'+cart+'_row', data=m_row)
            dset = tens_grp.create_dataset('mmat_'+cart+'_col', data=m_col)

    except KeyError:
        pass


def inspect_tensors(filename):
    """Returns list of coupled J quanta for each tensor stored in file.

    Args:
        filename : str
            Name of richmol HDF5 file.

    Returns:
        tens : dict
            Dictionary with keys as names of tensors and values as
            lists of pairs of J quanta coupled by tensor.
    """
    fl = h5py.File(filename, "r")

    tensors = {}

    J_key_re = re.sub(r"1.0", '\d+\.\d+', J_group_key(1, 1))
    tens_key_re = re.sub(r"dummy", '(\w+)', tens_group_key('dummy'))

    for key in fl.keys():

        if not re.match(r''+J_key_re, key): 
            continue
        Jpair = re.findall(r'\d+\.\d+', key)
        if len(Jpair)==0:
            continue
        else:
            J1, J2 = (round(float(elem), 1) for elem in Jpair)

        for key2 in fl[key]:
            if re.match(r''+tens_key_re, key2):
                tens = re.search(tens_key_re, key2).group(1)
                try:
                    tensors[tens].append( (J1, J2) )
                except KeyError:
                    tensors[tens] = []
                    tensors[tens].append( (J1, J2) )
    return tensors


def read_mmat(filename, tens, J1, J2):
    """Reads M-matrix for selected tensor and pair of J quanta.

    Args:
        filename : str
            Name of richmol HDF5 file.
        tens : str
            Name of tensor (as stored in file).
        J1, J2 : float
            Values of bra (J1) and ket (J2) rotational angular momentum quanta.

    Returns:
        swapJ : bool
            If True, the returned M-matrix data is for swapped J1 and J2 quanta.
        mmat : list
            List of sets for each Cartesian component of M-matrix.
            The elements of each set are the following:
                cart (str): Cartesian-component label
                numtype (str): type of elements, 'real, 'imag', or 'cmplx'
                irreps (list): list of irreducible representations
                nnz (list) : number of non-zero elements for each irreducible representation in irreps
                data (array(nnz)): non-zero elements concatenated across all irreps
                row (array(nnz)): row indices of non-zero elements, concatenated across all irreps
                col (array(nnz)): column indices of non-zero elements, concatenated across all irreps
    """

    fl = h5py.File(filename, "r")

    J_key_re = re.sub(r"1.0", '\d+\.\d+', J_group_key(1, 1))

    try:
        J_key = J_group_key(J1, J2)
        J_group = fl[J_key]
        swapJ = False
    except KeyError:
        try:
            J_key2 = J_group_key(J2, J1)
            J_group = fl[J_key2]
            swapJ = True
            J_key = J_key2
        except KeyError:
            raise KeyError(f"Can't locate data group {J_key} or {J_key2} in file {filename}") from None

    try:
        tens_key = tens_group_key(tens)
        tens_group = J_group[tens_key]
    except KeyError:
        raise KeyError(f"Can't locate data group {J_key}/{tens_key} in file {filename}") from None

    mmat = []
    for key in tens_group:
        if re.match(r'mmat_(\w+)_data', key):
            cart = re.search('mmat_(\w+)_data', key).group(1)
            dat = tens_group[key]
            data = dat[()]

            try:
                numtype = dat.attrs['numtype']
            except KeyError:
                raise KeyError(f"Can't locate attribute 'numtype' for dataset = {J_key}/{tens_key}/{key} in file = {filename}") from None
            try:
                irreps = dat.attrs['irreps']
            except KeyError:
                raise KeyError(f"Can't locate attribute 'irreps' for dataset = {J_key}/{tens_key}/{key} in file = {filename}") from None
            try:
                nnz = dat.attrs['nnz']
            except KeyError:
                raise KeyError(f"Can't locate attribute 'nnz' for dataset = {J_key}/{tens_key}/{key} in file = {filename}") from None
            try:
                cart_ = dat.attrs['cart']
            except KeyError:
                raise KeyError(f"Can't locate attribute 'cart' for dataset = {J_key}/{tens_key}/{key} in file = {filename}") from None

            assert (len(nnz) == len(irreps)), \
                f"Number of elements in nnz and irreps are not equal: {len(nnz)} != {len(irreps)}; " + \
                f"dataset = {J_key}/{tens_key}/{key}, file = {filename}"
            assert (sum(nnz) == len(data)), \
                f"Number of elements in data in sum(nnz) are not equal: {len(data)} != {sum(nnz)}; " + \
                f"dataset = {J_key}/{tens_key}/{key}, file = {filename}"
            assert (cart == cart_), \
                f"Cartesian label {cart} in dataset key does not match that in attribute = {cart_}; " + \
                f"dataset = {J_key}/{tens_key}/{key}, file = {filename}"

            try:
                key_row = 'mmat_'+cart+'_row'
                row = tens_group[key_row][()]
            except KeyError:
                raise KeyError(f"Can't locate dataset {J_key}/{tens_key}/{key_row} in file {filename}") from None
            try:
                key_col = 'mmat_'+cart+'_col'
                col = tens_group[key_col][()]
            except KeyError:
                raise KeyError(f"Can't locate dataset {J_key}/{tens_key}/{key_col} in file {filename}") from None

            assert (len(row) == len(data)), \
                f"Number of elements in row and data are not equal: {len(row)} != {len(data)}; " + \
                f"dataset = {J_key}/{tens_key}/{key}, file = {filename}"
            assert (len(col) == len(data)), \
                f"Number of elements in col and data are not equal: {len(col)} != {len(data)}; " + \
                f"dataset = {J_key}/{tens_key}/{key}, file = {filename}"

            mmat.append((cart, numtype, irreps, nnz, data, row, col))

    return swapJ, mmat



def inspect_kmat(filename, tens):
    """Returns information about K-matrix of tensor for different pairs
    of coupled J quanta.

    Args:
        filename : str
            Name of the output HDF5 file.
        tens : str
            Name of tensor, use 'inspect_tensors' to list all tensors
            stored in file 'filename'.

    Returns:
        kmat : dict
            Dictionary with keys as different pairs of coupled J quanta and values
            as dictionaries with data about the corresponding K-matrix.
    """
    fl = h5py.File(filename, "r")

    kmat = {}

    J_key_re = re.sub(r"1.0", '\d+\.\d+', J_group_key(1, 1))
    tens_key = tens_group_key(tens)

    for key in fl.keys():

        if not re.match(r''+J_key_re, key): 
            continue
        Jpair = re.findall(r'\d+\.\d+', key)
        if len(Jpair)==0:
            continue
        else:
            J1, J2 = (round(float(elem), 1) for elem in Jpair)

        try:
            data = fl[key][tens_key]['kmat']
            kmat[(J1, J2)] = {key2:elem2 for key2, elem2 in data.attrs.items()}
        except KeyError:
            pass
    return kmat


def inspect_mmat(filename, tens):
    """Returns information about M-matrix of tensor for different pairs
    of coupled J quanta.

    Args:
        filename : str
            Name of the output HDF5 file.
        tens : str
            Name of tensor, use 'inspect_tensors' to list all tensors
            stored in file 'filename'.

    Returns:
        mmat : dict
            Dictionary with keys as different pairs of coupled J quanta and values
            as dictionaries with data about the corresponding K-matrix.
    """
    fl = h5py.File(filename, "r")

    mmat = {}

    J_key_re = re.sub(r"1.0", '\d+\.\d+', J_group_key(1, 1))
    tens_key = tens_group_key(tens)

    for key in fl.keys():

        if not re.match(r''+J_key_re, key): 
            continue
        Jpair = re.findall(r'\d+\.\d+', key)
        if len(Jpair)==0:
            continue
        else:
            J1, J2 = (round(float(elem), 1) for elem in Jpair)

        try:
            for key2 in fl[key][tens_key]:
                if re.match(r'(mmat)', key2):
                    data = fl[key][tens_key][key2]
                    #cart = re.search('mmat:(\w+)', key2).group(1)
                    cart = data.attrs['cart']
                    mmat[cart] = {(J1, J2):{key2:elem2 for key2, elem2 in data.attrs.items()}}
        except KeyError:
            pass
    return mmat


def read_states(filename, J):
    """Reads field-free state indices, energies and assignments from HDF5 file.

    Args:
        filename : str
            Name of the output HDF5 file.

    Returns:
        descr : str
            Description of data
        dim : int
            Number of states, no_states (basis dimension for given J).
        enr : array (no_states) of floats
            State energies.
        id : array (no_states, 2) of integers
            State ID numbers [[:,0] and degenerate component indices [:,1].
        sym : array (no_states) of strings
            State symmetries.
        assign : array (no_states) of strings
            State assignments.
    """
    fl = h5py.File(filename, "r")

    key = J_group_key(J, J)
    try:
        J_grp = fl[key]
    except KeyError:
        raise KeyError(f"No data found for J1, J2 = {J}, {J}, file = {filename}") from None

    # read metadata

    try:
        descr = "".join(elem.decode() for elem in fl['descr'])
    except KeyError:
        descr = ""

    # read dimensions
    try:
        dim = J_grp['dim'][()]
        dim = dim[0]
    except KeyError:
        dim = None

    # read energies
    try:
        enr = J_grp['enr'][()]
    except KeyError:
        enr = None

    # read assignments
    try:
        assign = [elem.decode() for elem in J_grp['assign']]
    except KeyError:
        assign = None

    # read symmetries
    try:
        sym = [elem.decode() for elem in J_grp['sym']]
    except KeyError:
        sym = None

    # read IDs
    try:
        id = J_grp['id'][()]
    except KeyError:
        id = None

    return descr, dim, enr, id, sym, assign

'''
def read_explore(filename):
    """
    """
    fl = h5py.File(filename, "r")

    # metadata

    try:
        descr = "".join(elem.decode() for elem in fl['descr'])
    except KeyError:
        descr = ""


    J_key_re = re.sub(r"1.0", '\d+\.\d+', J_group_key(1, 1))
    tens_key_re = re.sub(r"dummy", '(\w+)', tens_group_key('dummy'))

    states = {}
    tensors = {}

   # explore pairs of J quanta
    for key in fl.keys():

        if not re.match(r''+J_key_re, key): 
            continue
        Jpair = re.findall(r'\d+\.\d+', key)
        if len(Jpair)==0:
            continue
        else:
            J1, J2 = (round(float(elem), 1) for elem in Jpair)

        for key2 in fl[key]:

            if re.match(r''+tens_key_re, key2):
                # explore tensors
                tens = re.search(tens_key_re, key2).group(1)

                # M- and K-matrices
                tens_data = [J1, J2]
                for key3 in fl[key][key2]:
                    numtype = fl[key][key2][key3].attrs['numtype']
                    irreps = fl[key][key2][key3].attrs['irreps']
                    thresh = fl[key][key2][key3].attrs['thresh']
                    tens_data.append((key3, numtype, irreps, thresh))

                tensors[tens] = tens_data

            else:
                # states data
                states[key2] = [J1, J2]

    for key in tensors:
        print(key, tensors[key])

    for key in states:
        print(key, states[key])




def read_tensor(filename, J1, J2, tens, read_k=True, ream_m=False, cart=None):
    """Reads field-free matrix elements of Cartesian tensor operator from HDF5 file.

    Args:
        filename : str
            Name of the output HDF5 file, if the file exists it will be appended,
            if desired it can be overwritten by passing replace=True.
        J1, J2 : int or half-integer float
            Values of J quantum numbers for bra (J1) and ket (J2) states.
        tens : string
            Name of Cartesian tensor operator for which the matrix elements
            wil be read.
    Kwargs:
        cart : string

    Returns:
        irreps : list of integers
            List of tensor irreducible representations.
        cart : string
            Since matrix elements for different Cartesian components of tensor
            (in the laboratory frame) are stored in separate calls,
                'cart' is used to label currently stored Cartesian component.
                Use together with 'mmat'
        kmat : list of csr_matrix(no_states(J1), no_states(J2))
            List of K-matrices for different irreducible representations
            of tensor, in the same order as in 'irreps'.
        mmat : list of csr_matrix(2*J1+1, 2*J2+1)
            List of M-matrices for different irreducible representations
            of tensor, in the same order as in 'irreps'.
            Only single Cartesian component of M-matrix can be stored
            at a call, the corresponding component is labelled by 'cart'.
        thresh : float
            Threshold for storing the matrix elements.

    """
    fl = h5py.File(filename, "r")

    key1 = str(round(float(J1), 1))
    key2 = str(round(float(J2), 1)) 
    key = key1 + "_" + key2
    try:
        J_grp = fl[key]
    except KeyError:
        raise KeyError(f"No data found for J1, J2 = {J1}, {J2}, file = {filename}") from None

    try:
        dim1 = fl[key1+'_'+key1+'/dim'][()]
        dim2 = fl[key2+'_'+key2+'/dim'][()]
    except KeyError:
        pass

    try:
        tens_J_grp = J_grp[tens]
    except KeyError:
        raise KeyError(f"Tensor '{tens}' is not found, J1, J2 = {J1}, {J2}, file = {filename}") from None

    try:
        kmat_data = tens_J_grp['kmat']
        kmat = []
        for k_irrep in kmat_data:
            kmat.append( csr_matrix((k_irrep[0], k_irrep[1], k_irrep[2]), shape=(dim1, dim2)) )
        irreps = kmat_data.attrs['irreps']
        numtype = kmat_data.attrs['numtype']
        thresh = kmat_data.attrs['thresh']
    except KeyError:
        raise KeyError(f"K-matrix is not found, tensor = '{tens}', J1, J2 = {J1}, {J2}, file = {filename}") from None
'''

if __name__ == "__main__":
    import random
    import string

    nstates = 3
    assign = []
    id = []

    for istate in range(nstates):
        assign.append( "".join(random.choice(string.ascii_lowercase) for i in range(10)) )
        id.append([istate,1])

    enr = np.random.uniform(low=0.5, high=13.3, size=(3,))


    dat = np.array([0.3+1j*5, 0.2+1j*0.3, 1.3+0.8*1j ,0.9], dtype=np.complex128)
    a = coo_matrix((dat, ([0,0,1,2],[0,1,2,1])), shape=(3,3))
    b = a.tocsr()

    a = coo_matrix((dat.real, ([0,0,1,2],[0,1,2,1])), shape=(3,3))
    c = a.tocsr()

    a = coo_matrix((dat.imag*1j, ([0,0,1,2],[0,1,2,1])), shape=(3,3))
    d = a.tocsr()

    a = coo_matrix((dat.imag*0, ([0,0,1,2],[0,1,2,1])), shape=(3,3))
    e = a.tocsr()

    kmat = [b,b,e,c,d]
    irreps = [0, 1,2,3,4]
    store("test_file.h5", 1, 1, replace=True, descr=['CCSD(T)/aug-cc-pwCVTZ', 'some other stuff'])
    store("test_file.h5", 1, 1, replace=False, enr=enr)
    store("test_file.h5", 1, 1, replace=False, assign=assign)
    store("test_file.h5", 1, 1, replace=False, kmat=kmat, irreps=irreps, cart="xx", tens='alpha')
    sys.exit()
    store("test_file.h5", 1, 1, replace=False, mmat=kmat, irreps=irreps, cart="xx", tens='alpha')
    store("test_file.h5", 1, 1, replace=False, mmat=kmat, irreps=irreps, cart="xy", tens='alpha')
    store("test_file.h5", 1, 1, replace=False, kmat=kmat, irreps=irreps, cart="xy", tens='alpha')

    store("test_file.h5", 1, 2, replace=False, kmat=kmat, irreps=irreps, cart="xx", tens='mu')
    store("test_file.h5", 1, 2, replace=False, kmat=kmat, irreps=irreps, cart="xy", tens='mu')
    store("test_file.h5", 1, 2, replace=False, kmat=kmat, irreps=irreps, cart="xy", tens='mu')


    tens = inspect_tensors("test_file.h5")
    print(tens)

    mmat = inspect_mmat("test_file.h5", 'alpha')
    print(mmat)

    #descr, enr, id, sym, assign, kmat, mmat = read("test_file.h5", 1,1)
    #print(enr)
    #print(descr)
    #print(assign)
    #print(sym)
