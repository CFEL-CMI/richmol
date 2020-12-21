import h5py
import numpy as np
import re
import sys
import os
from scipy.sparse import coo_matrix
import platform
import time


"""
Data structure of richmol HDF5 file

  /descr                               (description of the data)
   |
  /tens:<tens>                         (data for specific tensor)
  /tens:<tens>.units                   (units)
  /tens:<tens>.rank                    (rank)
  /tens:<tens>.irreps                  (list of irreducible representations)
  /tens:<tens>.cart                    (list of Cartesian components)
  /tens:<tens>.descr                   (description of tensor)
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
        tens_descr : str or list
            Description of tensor.
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
        if descr is None:
            raise KeyError()
        if 'descr' in fl:
            del fl['descr']
        if isinstance(descr, str):
            data = descr
        elif all(isinstance(elem, str) for elem in descr):
            data = " ".join(elem for elem in descr)
        else:
            raise TypeError(f"bad argument type for 'descr': '{type(descr)}'") from None
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
        if units is None:
            raise KeyError()
        tens_grp.attrs['units'] = units
    except KeyError:
        pass

    # store tensor description

    try:
        descr = kwargs['tens_descr']
        if descr is None:
            raise KeyError()
        if isinstance(descr, str):
            data = descr
        elif all(isinstance(elem, str) for elem in descr):
            data = " ".join(elem for elem in descr)
        else:
            raise TypeError(f"bad argument type for 'tens_descr': '{type(descr)}'") from None
        tens_grp.attrs['descr'] = data
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
            components ('cart'), tensor description ('descr'), and list of coupled J quanta ('J_pairs').
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

        for attr in ('rank', 'irreps', 'cart', 'units', 'descr'):
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
        units : str
            Energy units.
    """
    fl = h5py.File(filename, 'r')

    try:
        J_grp = fl[tens_group_key(tens)][J_group_key(J, J)]
    except KeyError:
        raise KeyError(f"Can't locate data group {tens_group_key(tens)}/{J_group_key(J, J)} in file {filename}") from None

    # read units
    try:
        units = fl[tens_group_key(tens)].attrs['units']
    except KeyError:
        units = None

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

    return enr, id, ideg, sym, assign, units


def old_to_new_richmol(h5_file, states_file, tens_file=None, replace=False, store_states=True, \
                       me_tol=1e-40, **kwargs):
    """Converts richmol old formatted text file format into to new hdf5 file format.

    Args:
        h5_fname : str
            Name of new richmol hdf5 file.
        states_fname : str
            Name of old richmol states file.
        tens_fname : str
            Template for generating names of old richmol tensor matrix elements files.
            For example, for filename="matelem_alpha_j<j1>_j<j2>.rchm" following files will be
            searched: matelem_alpha_j0_j0.rchm, matelem_alpha_j0_j1.rchm,
            matelem_alpha_j0_j2.rchm and so on, where <j1> and <j2> will be replaced by integer
            numbers running through all J quanta spanned by all states listed in file states_fname.
            NOTE: in old richmol format, j1 and j2 are treated as ket and bra state quanta, respectively.
            For half-integer numbers (e.g., F quanta), substitute <j1> and <j2> in the template
            tens_fname by <f1> and <f2>, which will then be replaced by floating point numbers
            rounded to the first decimal.
        replace : bool
            Set True to replace existing hdf5 file.
        store_states : bool
            Set True to write richmol states data into hdf5 file, or False if you call this function
            multiple times to convert different tensors and don't want each time to rewrite the states
            data. The states file however need to be loaded for every new tensor.
        me_tol : float
            Threshold for neglecting matrix elements.
    Kwargs:
        descr : str or list
            Description of richmol data. Old richmol file format does not contain description
            of the data, this can be added into hdf5 file by passing a string or list of strings
            in descr variable. The description can be added only if store_states=True.
        tens_descr : str or list
            Description of tensor.
        tens_units : str
            Tensor units.
        enr_units : str
            Energy units.
        tens_name : str
            If provided, the name of tensor read from richmol file will be replaced with tens_name.
            Tensor name is used to generate data group name for matrix elements of tensor in hdf5 file.
            For state energies, the data group name is 'h0'.
    """
    if replace == True:
        store_states_ = True
    else:
        store_states_ = store_states

    try:
        descr = kwargs['descr']
    except KeyError:
        descr = None

    try:
        tens_descr = kwargs['tens_descr']
    except KeyError:
        tens_descr = None

    try:
        tens_units = kwargs['tens_units']
    except KeyError:
        tens_units = None

    try:
        enr_units = kwargs['enr_units']
    except KeyError:
        enr_units = None

    try:
        tens_name = kwargs['tens_name']
    except KeyError:
        tens_name = None

    # read states data
    print(f"Read states data from richmol formatted text file {states_file}")
    fl = open(states_file, "r")
    maxid = {}
    for line in fl:
        w = line.split()
        J = round(float(w[0]),1)
        id = np.int64(w[1])
        try:
            maxid[J] = max(maxid[J], id)
        except KeyError:
            maxid[J] = id
    fl.seek(0)
    map_id_to_istate = {J : np.zeros(maxid[J]+1, dtype=np.int64) for J in maxid.keys()}
    for J, elem in map_id_to_istate.items():
        elem[:] = -1
    states = {}
    nstates = {J : 0 for J in maxid.keys()}
    for line in fl:
        w = line.split()
        J = round(float(w[0]),1)
        id = np.int64(w[1])
        sym = w[2]
        ndeg = int(w[3])
        enr = float(w[4])
        qstr = ' '.join([w[i] for i in range(5,len(w))])
        try:
            map_id_to_istate[J][id] = nstates[J]
        except KeyError:
            map_id_to_istate[J][id] = 0
        try:
            x = states[J][0]
        except (IndexError, KeyError):
            states[J] = []
        for ideg in range(ndeg):
            states[J].append((nstates[J], sym, ideg, enr, qstr))
            nstates[J] += 1
    fl.close()

    # store states data
    if store_states_ == True:
        print(f"Write states data into richmol hdf5 file {h5_file}, replace={replace}")
        for iJ, J in enumerate(states.keys()):
            id = [elem[0] for elem in states[J]]
            sym = [elem[1] for elem in states[J]]
            ideg = [elem[2] for elem in states[J]]
            enr = [elem[3] for elem in states[J]]
            qstr = [elem[4] for elem in states[J]]
            store(h5_file, 'h0', J, J, enr=enr, id=id, sym=sym, ideg=ideg, assign=qstr, \
                  replace=(replace if iJ == 0 else False), descr=descr, units=enr_units)

    # read matrix elements data and store into hdf5 file
    print(f"Convert richmol matrix elements data {tens_file} --> {h5_file}, tol={me_tol}")
    for J1 in states.keys():
        for J2 in states.keys():

            F1_str = str(round(J1,1))
            F2_str = str(round(J2,1))
            J1_str = str(int(round(J1,0)))
            J2_str = str(int(round(J2,0)))

            fname = re.sub(r"\<f1\>", F1_str, tens_file)
            fname = re.sub(r"\<f2\>", F2_str, fname)
            fname = re.sub(r"\<j1\>", J1_str, fname)
            fname = re.sub(r"\<j2\>", J2_str, fname)

            if not os.path.exists(fname):
                continue

            print(f"Read matrix elements from richmol formatted text file {fname}")
            fl = open(fname, "r")

            iline = 0
            eof = False
            read_m = False
            read_k = False
            icart = None

            for line in fl:
                strline = line.rstrip('\n')

                if iline==0:
                    if strline!="Start richmol format":
                        raise RuntimeError(f"Matrix-elements file '{fname}' has bogus header = '{strline}'")
                    iline+=1
                    continue

                if strline == "End richmol format":
                    eof = True
                    break

                if iline==1:
                    w = strline.split()
                    if tens_name is None:
                        tens_name = w[0]
                    nomega = int(w[1])
                    ncart = int(w[2])
                    iline+=1
                    continue

                if strline=="M-tensor":
                    read_m = True
                    read_k = False
                    mvec = {}
                    im1 = {}
                    im2 = {}
                    iline+=1
                    continue

                if strline=="K-tensor":
                    read_m = False
                    read_k = True
                    kvec = [[] for i in range(nomega)]
                    ik1 = [[] for i in range(nomega)]
                    ik2 = [[] for i in range(nomega)]
                    iline+=1
                    continue

                if read_m is True and strline.split()[0]=="alpha":
                    w = strline.split()
                    icart = int(w[1])
                    icmplx = int(w[2])
                    scart = w[3].lower()
                    cmplx_fac = (1j, 1)[icmplx+1]
                    mvec[scart] = [[] for i in range(nomega)]
                    im1[scart] = [[] for i in range(nomega)]
                    im2[scart] = [[] for i in range(nomega)]
                    iline+=1
                    continue

                if read_m is True:
                    w = strline.split()
                    m1 = float(w[0])
                    m2 = float(w[1])
                    i1, i2 = int(m1 + J1), int(m2 + J2)
                    mval = [ float(val) * cmplx_fac for val in w[2:] ]
                    ind_omega = [i for i in range(nomega) if abs(mval[i]) > me_tol]
                    for iomega in ind_omega:
                        im1[scart][iomega].append(i1)
                        im2[scart][iomega].append(i2)
                        mvec[scart][iomega].append(mval[iomega])

                if read_k is True:
                    w = strline.split()
                    id1 = int(w[0])
                    id2 = int(w[1])
                    ideg1 = int(w[2])
                    ideg2 = int(w[3])
                    kval = [float(val) for val in w[4:]]
                    istate1 = map_id_to_istate[J1][id1] + ideg1 - 1
                    istate2 = map_id_to_istate[J2][id2] + ideg2 - 1
                    ind_omega = [i for i in range(nomega) if abs(kval[i]) > me_tol]
                    for iomega in ind_omega:
                        ik1[iomega].append(istate1)
                        ik2[iomega].append(istate2)
                        kvec[iomega].append(kval[iomega])

                iline +=1
            fl.close()

            if eof is False:
                raise RuntimeError(f"Matrix-elements file '{fname}' has bogus footer = '{strline}'")

            print(f"Write matrix elements into richmol hdf5 file {h5_file}, me_tol = {me_tol}")

            # write M-matrix into hdf5 file
            for cart in mvec.keys():
                print(f"    M-tensor's {cart} component, irreps = {ind_omega}")
                ind_omega = [iomega for iomega in range(nomega) if len(mvec[cart][iomega]) > 0]
                mat = [ coo_matrix((mvec[cart][iomega], (im2[cart][iomega], im1[cart][iomega])), \
                            shape=(int(2*J2+1), int(2*J1+1)) ) for iomega in ind_omega ]
                store(h5_file, tens_name, J2, J1, thresh=me_tol, irreps=ind_omega, cart=cart, mmat=mat)

            # write K-matrix into hdf5 file
            print(f"    K-tensor, irreps = {ind_omega}")
            ind_omega = [iomega for iomega in range(nomega) if len(kvec[iomega]) > 0]
            mat = [ coo_matrix((kvec[iomega], (ik2[iomega], ik1[iomega])), shape=(nstates[J2], nstates[J1]) ) \
                    for iomega in ind_omega ]
            store(h5_file, tens_name, J2, J1, thresh=me_tol, irreps=ind_omega, kmat=mat, \
                  tens_descr=tens_descr, units=tens_units)


