import h5py
import numpy as np
import re
import sys

"""
Data structure of richmol HDF5 file

  /descr                               (description of the data)
   |
  /tens:<tens>                         (data for specific tensor)
  /tens:<tens>.units                   (units)
  /tens:<tens>.rank                    (rank)
  /tens:<tens>.irreps                  (list of irreducible representations)
  /tens:<tens>.cart                    (list of Cartesian components)
   |
   |---/J:<J1>,<J1>                    (H0 basis states, <J1|H0|J1> = E)
   |    |
   |    |---/enr                       (state energies)
   |    |---/sym                       (state symmetries)
   |    |---/assign                    (state assignments)
   |    |---/id                        (state IDs)
   |    |---/ideg                      (state degeneracy indices)
   |
   |---/J:<J1>,<J2>                    (tensor matrix elements between H0 basis states, <J1|<tens>|J2>
        |
        |---/kmat_data                 (vector of non-zero elements of K-matrix, concatenated across all irreducible representations)
        |---/kmat_data.irreps          (list of irreducible representations)
        |---/kmat_data.nnz             (number of elements in kmat_data for each irreducible representation)
        |---/kmat_row                  (bra state IDs of non-zero elements)
        |---/kmat_col                  (ket state IDs of non-zero elements)
        |
        |---/mmat_<cart>_data          (...)
        |---/mmat_<cart>_data.irreps   (...)
        |---/mmat_<cart>_data.nnz      (...)
        |---/mmat_<cart>_data.cart     (Cartesian component label cart == <cart>)
        |---/mmat_<cart>_row           (bra state m quanta, as m+J, of non-zero elements)
        |---/mmat_<cart>_col           (ket state m quanta, as m+J, of non-zero elements)

 <J1> and <J2> (float): bra and ket quantum numbers of the total angular momentum operator, rounded
    to the first decimal, e.g., 1.0 and 2.0 or 1.5 and 2.5
 <tens> (str): name of tensor operator
 <cart> (str): Cartesian component of tensor
"""


def J_group_key(J1, J2):
    return 'J:' + str(round(float(J1), 1)) + "," + str(round(float(J2), 1))


def tens_group_key(tens):
    return 'tens:' + tens


def store(filename, tens, J1, J2, replace=False, thresh=1e-14, **kwargs):
    """Stores field-free state energies, assignments, and matrix elements
    of Cartesian tensor operators in richmol HDF5 file.

    Args:
        filename : str
            Name of the output HDF5 file, if file exists it will be appended, it can be overwritten
            by passing replace=True.
        tens : str
            Name of tensor operator.
        J1, J2 : int or float
            Values of J quantum numbers for bra (J1) and ket (J2) states.
        replace : bool
            If True, existing HDF5 file will be overwritten.

    Kwargs:
        descr : str or list
            Description of the data.
        enr : array (no_states)
            State energies, no_states is the number of states for J = J1 == J2.
        sym : array (no_states)
            State symmetries.
        assign : array (no_states)
            State assignments.
        id : array (no_states)
            State ID numbers.
        ideg : array (no_states)
            State degeneracy indices.
        irreps : list
            List of tensor irreducible representations.
        cart : str
            Cartesian component of tensor.
        kmat : list of scipy.coo_matrix, where coo_matrix.shape = (no_states(J1), no_states(J2))
            List of K-matrices for different irreducible representations of tensor.
            Each K-matrix is stored in a sparse scipy.coo_matrix format, where kmat[irrep].row and
            kmat[irrep].col contain IDs of bra and ket states, respectively, that are coupled
            by tensor with corresponding matrix element given by kmat[irrep].data.
        mmat : list of scipy.coo_matrix, where coo_matrix.shape = (2*J1+1, 2*J2+1)
            List of M-matrices for different irreducible representations of tensor.
            Each M-matrix is stored in a sparse scipy.coo_matrix format, where mmat[irrep].row and
            mmat[irrep].col contain m1=-J1..J1 bra and m2=-J2..J2 ket quanta as int(m+J), respectively,
            that are coupled by tensor with corresponding matrix element given by mmat[irrep].data.
            Only single Cartesian component of M-matrix (labelled by 'cart') can be stored at a call.
        units : str
            Tensor units.
        thresh : float
            Threshold for neglecting matrix elements.
    """

    fl = h5py.File(filename, {True:'w', False:'a'}[replace])

    if tens_group_key(tens) not in fl:
        tens_grp = fl.create_group(tens_group_key(tens))
    else:
        tens_grp = fl[tens_group_key(tens)]

    if J_group_key(J1, J2) not in tens_grp:
        J_grp = tens_grp.create_group(J_group_key(J1, J2))
    else:
        J_grp = tens_grp[J_group_key(J1, J2)]

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
        if 'enr' in J_grp:
            del J_grp['enr']
        dset = J_grp.create_dataset('enr', data=enr)
    except KeyError:
        pass

    # store assignment

    try:
        assign = kwargs['assign']
        if 'assign' in J_grp:
            del J_grp['assign']
        dset = J_grp.create_dataset('assign', data=[str.encode(elem) for elem in assign])
    except KeyError:
        pass

    # store IDs

    try:
        id = kwargs['id']
        assert (len(list(set(id))) == len(id)), f"State IDs contain duplicates for J = {J1}"
        if 'id' in J_grp:
            del J_grp['id']
        dset = J_grp.create_dataset('id', data=id)
    except KeyError:
        pass

    # store degeneracy indices

    try:
        ideg = kwargs['ideg']
        if 'ideg' in J_grp:
            del J_grp['ideg']
        dset = J_grp.create_dataset('ideg', data=ideg)
    except KeyError:
        pass

    # store symmetries

    try:
        sym = kwargs['sym']
        if 'sym' in J_grp:
            del J_grp['sym']
        dset = J_grp.create_dataset('sym', data=[str.encode(elem) for elem in sym])
    except KeyError:
        pass

    # store tensor units

    try:
        units = kwargs['units']
        tens_grp.attrs['units'] = units
    except KeyError:
        pass

    # store K-matrix

    try:
        kmat = kwargs['kmat']

        if 'irreps' not in kwargs:
            raise Exception(f"'irreps' argument must be passed together with 'kmat'") from None
        else:
            irreps = kwargs['irreps']
            assert (len(irreps) == len(kmat)), \
                f"Lengths of 'irreps' and 'kmat' are not equal: {len(irreps)} != {len(kmat)}"

        k_nnz = []
        k_irrep = []
        k_data = None
        k_row = None
        k_col = None

        for km, irrep in zip(kmat, irreps):

            data = km.data
            row = km.row
            col = km.col

            if len(data) == 0:
                continue
            if np.all(np.abs(data) < thresh):
                continue

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

        if k_data is not None:

            try:
                tens_irreps = tens_grp.attrs['irreps']
                tens_grp.attrs['irreps'] = list(set(list(tens_irreps) + list(k_irrep)))
            except KeyError:
                tens_grp.attrs['irreps'] = k_irrep

            if 'kmat_data' in J_grp:
                del J_grp['kmat_data']
            if 'kmat_row' in J_grp:
                del J_grp['kmat_row']
            if 'kmat_col' in J_grp:
                del J_grp['kmat_col']

            dset = J_grp.create_dataset('kmat_data', data=k_data)
            dset.attrs['irreps'] = k_irrep
            dset.attrs['nnz'] = k_nnz
            dset = J_grp.create_dataset('kmat_row', data=k_row)
            dset = J_grp.create_dataset('kmat_col', data=k_col)

    except KeyError:
        pass

    # store M-matrix

    try:
        mmat = kwargs['mmat']

        if 'cart' not in kwargs:
            raise Exception(f"'cart' argument must be passed together with 'mmat'") from None
        else:
            cart = kwargs['cart']

        assert all(c.lower() in 'xyz' for c in cart), f"Illegal Cartesian label = '{cart}'"

        if 'irreps' not in kwargs:
            raise Exception(f"'irreps' argument must be passed together with 'mmat'") from None
        else:
            irreps = kwargs['irreps']
            assert (len(irreps) == len(mmat)), \
                f"Lengths of 'irreps' and 'mmat' are not equal: {len(irreps)} != {len(mmat)}"

        m_data = None
        m_row = None
        m_col = None
        m_nnz = []
        m_irrep = []

        for mm, irrep in zip(mmat, irreps):

            data = mm.data
            row = mm.row
            col = mm.col

            if len(data) == 0:
                continue
            if np.all(np.abs(data) < thresh):
                continue

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

        if m_data is not None:

            rank = len(cart)
            try:
                rank_ = tens_grp.attrs['rank']
                assert (rank == rank_), f"Rank of '{tens}' = {rank} derived from Cartesian label '{cart}' " + \
                    f"for J1, J2 = {J1}, {J2} does not match that = {rank_} already stored in file {filename}"
            except KeyError:
                tens_grp.attrs['rank'] = rank

            try:
                tens_cart = tens_grp.attrs['cart']
                tens_grp.attrs['cart'] = list(set(list(tens_cart) + [cart]))
            except KeyError:
                tens_grp.attrs['cart'] = [cart]

            try:
                tens_irreps = tens_grp.attrs['irreps']
                tens_grp.attrs['irreps'] = list(set(list(tens_irreps) + list(m_irrep)))
            except KeyError:
                tens_grp.attrs['irreps'] = list(m_irrep)

            if 'mmat_'+cart+'_data' in J_grp:
                del J_grp['mmat_'+cart+'_data']
            if 'mmat_'+cart+'_row' in J_grp:
                del J_grp['mmat_'+cart+'_row']
            if 'mmat_'+cart+'_col' in J_grp:
                del J_grp['mmat_'+cart+'_col']

            dset = J_grp.create_dataset('mmat_'+cart+'_data', data=m_data)
            dset.attrs['irreps'] = m_irrep
            dset.attrs['nnz'] = m_nnz
            dset.attrs['cart'] = cart
            dset = J_grp.create_dataset('mmat_'+cart+'_row', data=m_row)
            dset = J_grp.create_dataset('mmat_'+cart+'_col', data=m_col)

    except KeyError:
        pass

    fl.close()


def inspect_tensors(filename):
    """Returns list of tensors stored in file, for each tensor returns rank, list of irreps,
    list or Cartesian components, units, and list of coupled J quanta.

    Args:
        filename : str
            Name of richmol HDF5 file.

    Returns:
        tensors : dict
            Dictionary with keys as names of tensors and values as another dictionaries
            containing rank of tensor ('rank'), list of irreps ('irreps'), list of Cartesian
            components ('cart'), and list of coupled J quanta ('J_pairs').
    """
    fl = h5py.File(filename, 'r')

    tensors = {}

    J_key_re = re.sub(r'1.0', '\d+\.\d+', J_group_key(1, 1))
    tens_key_re = re.sub(r'dummy', '(\w+)', tens_group_key('dummy'))

    for key in fl.keys():

        if not re.match(r''+tens_key_re, key): 
            continue
        tens = re.search(tens_key_re, key).group(1)

        tensors[tens] = {'J_pairs':[]}

        for attr in ('rank', 'irreps', 'cart', 'units'):
            try:
                tensors[tens][attr] = fl[key].attrs[attr]
            except KeyError:
                continue

        for key2 in fl[key]:
            if not re.match(J_key_re, key2):
                continue
            Jpair = re.findall(r'\d+\.\d+', key2)
            if len(Jpair)==0:
                continue
            J1, J2 = (round(float(elem), 1) for elem in Jpair)

            if 'kmat_data' in fl[key][key2] and any(re.match(r'mmat_(\w+)_data', key3) for key3 in fl[key][key2]):
                tensors[tens]['J_pairs'].append((J1, J2))
            elif 'enr' in fl[key][key2] and 'id' in fl[key][key2]:
                tensors[tens]['J_pairs'].append((J1, J2))

    fl.close()

    return tensors


def read_mmat(filename, tens, J1, J2):
    """Reads M-matrix for selected tensor and pair of J quanta.

    Args:
        filename : str
            Name of richmol HDF5 file.
        tens : str
            Name of tensor.
        J1, J2 : float
            Values of bra (J1) and ket (J2) rotational angular momentum quanta.

    Returns:
        swapJ : bool
            If True, the returned M-matrix data is for swapped J1 and J2 quanta.
        mmat : list
            List of sets for each Cartesian component of M-matrix.
            The elements of each set are the following:
                cart (str): Cartesian-component label
                irreps (list): list of irreducible representations
                nnz (list) : number of non-zero elements for each irreducible representation in irreps
                data (array(nnz)): non-zero elements concatenated across all irreps
                row (array(nnz)): row indices of non-zero elements, concatenated across all irreps
                col (array(nnz)): column indices of non-zero elements, concatenated across all irreps
    """
    fl = h5py.File(filename, 'r')

    try:
        tens_key = tens_group_key(tens)
        tens_group = fl[tens_key]
    except KeyError:
        raise KeyError(f"Can't locate data group '{tens_key}' in file {filename}") from None

    try:
        J_key = J_group_key(J1, J2)
        J_group = tens_group[J_key]
        swapJ = False
    except KeyError:
        try:
            J_key2 = J_group_key(J2, J1)
            J_group = tens_group[J_key2]
            swapJ = True
            J_key = J_key2
        except KeyError:
            raise KeyError(f"Can't locate data group '{tens_key}/{J_key}' neither '{tens_key}/{J_key2}' " + \
                f"in file {filename}") from None

    mmat = []
    for key in J_group:
        if re.match(r'mmat_(\w+)_data', key):
            cart = re.search('mmat_(\w+)_data', key).group(1)
            dat = J_group[key]
            data = dat[()]

            try:
                irreps = dat.attrs['irreps']
            except KeyError:
                raise KeyError(f"Can't locate attribute 'irreps' for dataset = '{tens_key}/{J_key}/{key}' in file = {filename}") from None
            try:
                nnz = dat.attrs['nnz']
            except KeyError:
                raise KeyError(f"Can't locate attribute 'nnz' for dataset = '{tens_key}/{J_key}/{key}' in file = {filename}") from None
            try:
                cart_ = dat.attrs['cart']
            except KeyError:
                raise KeyError(f"Can't locate attribute 'cart' for dataset = '{tens_key}/{J_key}/{key}' in file = {filename}") from None

            assert (len(nnz) == len(irreps)), \
                f"Number of elements in 'nnz' and 'irreps' are not equal: {len(nnz)} != {len(irreps)}; " + \
                f"dataset = '{tens_key}/{J_key}/{key}', file = {filename}"
            assert (sum(nnz) == len(data)), \
                f"Number of elements in 'data' in sum('nnz') are not equal: {len(data)} != {sum(nnz)}; " + \
                f"dataset = '{tens_key}/{J_key}/{key}', file = {filename}"
            assert (cart == cart_), \
                f"Cartesian label '{cart}' in dataset key does not match that in attribute = '{cart_}'; " + \
                f"dataset = '{tens_key}/{J_key}/{key}', file = {filename}"

            try:
                key_row = 'mmat_'+cart+'_row'
                row = J_group[key_row][()]
            except KeyError:
                raise KeyError(f"Can't locate dataset '{tens_key}/{J_key}/{key_row}' in file {filename}") from None
            try:
                key_col = 'mmat_'+cart+'_col'
                col = J_group[key_col][()]
            except KeyError:
                raise KeyError(f"Can't locate dataset '{tens_key}/{J_key}/{key_col}' in file {filename}") from None

            assert (len(row) == len(data)), \
                f"Number of elements in 'row' and 'data' are not equal: {len(row)} != {len(data)}; " + \
                f"dataset = '{tens_key}/{J_key}/{key}', file = {filename}"
            assert (len(col) == len(data)), \
                f"Number of elements in 'col' and 'data' are not equal: {len(col)} != {len(data)}; " + \
                f"dataset = '{tens_key}/{J_key}/{key}', file = {filename}"

            mmat.append((cart, irreps, nnz, data, row, col))

    fl.close()

    return swapJ, mmat


def read_kmat(filename, tens, J1, J2):
    """Reads K-matrix for selected tensor and pair of J quanta.

    Args:
        filename : str
            Name of richmol HDF5 file.
        tens : str
            Name of tensor.
        J1, J2 : float
            Values of bra (J1) and ket (J2) rotational angular momentum quanta.

    Returns:
        swapJ : bool
            If True, the returned K-matrix data is for swapped J1 and J2 quanta.
        kmat : set
            The elements of set are the following:
                irreps (list): list of irreducible representations
                nnz (list) : number of non-zero elements for each irreducible representation in irreps
                data (array(nnz)): non-zero elements concatenated across all irreps
                row (array(nnz)): row indices of non-zero elements, concatenated across all irreps
                col (array(nnz)): column indices of non-zero elements, concatenated across all irreps
    """
    fl = h5py.File(filename, 'r')

    try:
        tens_key = tens_group_key(tens)
        tens_group = fl[tens_key]
    except KeyError:
        raise KeyError(f"Can't locate data group '{tens_key}' in file {filename}") from None

    try:
        J_key = J_group_key(J1, J2)
        J_group = tens_group[J_key]
        swapJ = False
    except KeyError:
        try:
            J_key2 = J_group_key(J2, J1)
            J_group = tens_group[J_key2]
            swapJ = True
            J_key = J_key2
        except KeyError:
            raise KeyError(f"Can't locate data group '{tens_key}/{J_key}' neither '{tens_key}/{J_key2}' " + \
                f"in file {filename}") from None

    try:
        key = 'kmat_data'
        dat = J_group[key]
    except KeyError:
        raise KeyError(f"Can't locate data group '{tens_key}/{J_key}/{key}'") from None

    data = dat[()]

    try:
        irreps = dat.attrs['irreps']
    except KeyError:
        raise KeyError(f"Can't locate attribute 'irreps' for dataset = '{tens_key}/{J_key}/{key}' in file = {filename}") from None
    try:
        nnz = dat.attrs['nnz']
    except KeyError:
        raise KeyError(f"Can't locate attribute 'nnz' for dataset = '{tens_key}/{J_key}/{key}' in file = {filename}") from None

    assert (len(nnz) == len(irreps)), \
        f"Number of elements in 'nnz' and 'irreps' are not equal: {len(nnz)} != {len(irreps)}; " + \
        f"dataset = '{tens_key}/{J_key}/{key}', file = {filename}"
    assert (sum(nnz) == len(data)), \
        f"Number of elements in 'data' in sum('nnz') are not equal: {len(data)} != {sum(nnz)}; " + \
        f"dataset = '{tens_key}/{J_key}/{key}', file = {filename}"

    try:
        key_row = 'kmat_row'
        row = J_group[key_row][()]
    except KeyError:
        raise KeyError(f"Can't locate dataset '{tens_key}/{J_key}/{key_row}' in file {filename}") from None
    try:
        key_col = 'kmat_col'
        col = J_group[key_col][()]
    except KeyError:
        raise KeyError(f"Can't locate dataset '{tens_key}/{J_key}/{key_col}' in file {filename}") from None

    assert (len(row) == len(data)), \
        f"Number of elements in 'row' and 'data' are not equal: {len(row)} != {len(data)}; " + \
        f"dataset = '{tens_key}/{J_key}/{key}', file = {filename}"
    assert (len(col) == len(data)), \
        f"Number of elements in 'col' and 'data' are not equal: {len(col)} != {len(data)}; " + \
        f"dataset = '{tens_key}/{J_key}/{key}', file = {filename}"

    kmat  = (irreps, nnz, data, row, col)

    fl.close()

    return swapJ, kmat


def read_descr(filename):
    fl = h5py.File(filename, 'r')
    try:
        descr = "".join(elem.decode() for elem in fl['descr'])
    except KeyError:
        descr = None
    return descr


def read_states(filename, tens, J):
    """Reads field-free states.

    Args:
        filename : str
            Name of richmol HDF5 file.
        tens : str
            Name of dataset containing diagonal tensor H0 that stores basis states.
        J : int or float
            J quantum number for which stated to be pulled out.

    Returns:
        enr : array (no_states)
            State energies.
        id : array (no_states)
            State ID numbers.
        ideg : array (no_states)
            State degenerate component indices.
        sym : array (no_states)
            State symmetries.
        assign : array (no_states)
            State assignments.
    """
    fl = h5py.File(filename, 'r')

    try:
        J_grp = fl[tens_group_key(tens)][J_group_key(J, J)]
    except KeyError:
        raise KeyError(f"Can't locate data group {tens_group_key(tens)}/{J_group_key(J, J)} in file {filename}") from None

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

    # read idegs
    try:
        ideg = J_grp['ideg'][()]
    except KeyError:
        ideg = None

    fl.close()

    return enr, id, ideg, sym, assign

