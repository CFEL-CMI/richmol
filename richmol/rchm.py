import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import re
import sys


def J_group_key(J1, J2):
    return 'J:' + str(round(float(J1), 1)) + "," + str(round(float(J2), 1))

def tens_group_key(tens):
    return 'tens:' + tens


def store(filename, J1, J2, replace=False, thresh=1e-14, **kwargs):
    """Stores field-free states energies, assignments, and matrix elements
    of Cartesian tensor operators in HDF5 file.

    Args:
        filename : str
            Name of the output HDF5 file, if the file exists it will be appended,
            if desired it can be overwritten by passing replace=True.
        J1, J2 : int or half-integer float
            Values of J quantum numbers for bra (J1) and ket (J2) states.
        replace : bool
            If True, the HDF5 file when existing will be overwritten.

    Kwargs:
        descr : string or list of strings
            Description of the data.
        enr : array (no_states) of floats
            State energies, no_states is the number of states for J = J1 == J2.
        sym : array (no_states) of strings
            State symmetries.
        assign : array (no_states) of strings
            State assignments.
        id : array (no_states, 2) of integers
            State ID numbers [[:,0] and degenerate component indices [:,1].
        tens : string
            Name of Cartesian tensor operator for which the matrix elements
            to be stored. Use together with 'kmat' and 'mmat'.
        irreps : list of integers
            List of tensor irreducible representations.
            Use together with 'kmat' and 'mmat'.
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
    def _J1_eq_J2(argname):
        # check if J1 == J2
        assert (round(float(J1), 1) == round(float(J2), 1)), \
            f"Bra and ket J1 and J2 quanta (= {J1} and {J2}) must be equal " + \
            f"to each other when storing {argname}"

    def _check_dim(dim, argname):
        # check if dimensions in dim match those for J1_J2 group
        try:
            dim_ = J_grp['dim'][()]
            assert (all(dim == dim_)), \
                f"Dimensions of '{argname}' = {dim} do not match " + \
                f"basis set dimensions = {dim_} for J1, J2 = {J1}, {J2} " + \
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
                    f"basis set dimensions = {dim_[0]} for J1 = {J1} " + \
                    f"already stored in file {filename}"
            except KeyError:
                dset = fl.create_dataset(key1+'/dim', data=[dim[0], dim[0]])
            try:
                dim_ = fl[key2+'/dim'][()]
                assert (dim[1] == dim_[0]), \
                    f"Second dimension of '{argname}' = {dim[1]} does not match " + \
                    f"basis set dimensions = {dim_[0]} for J2 = {J2} " + \
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
            print(f"replace state energies for J1, J2 = {J1}, {J2} in file {filename}")
        else:
            print(f"write state energies for J1, J2 = {J1}, {J2} into file {filename}")
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
            print(f"replace state assignments for J1, J2 = {J1}, {J2} in file {filename}")
        else:
            print(f"write state assignments for J1, J2 = {J1}, {J2} into file {filename}")
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
            print(f"replace state IDs for J1, J2 = {J1}, {J2} in file {filename}")
        else:
            print(f"write state IDs for J1, J2 = {J1}, {J2} into file {filename}")
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
            print(f"replace state symmetries for J1, J2 = {J1}, {J2} in file {filename}")
        else:
            print(f"write state symmetries for J1, J2 = {J1}, {J2} into file {filename}")
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
                f"Number of elements in 'irreps' = {len(irreps)} does not match " + \
                f"the number of elements in 'kmat' = {len(kmat)}"

        numtype = []
        k_data = []
        for k_irrep in kmat:
            data = k_irrep.data
            indices = k_irrep.indices
            indptr = k_irrep.indptr
            if np.any(np.abs(data.real) >= thresh) and np.all(np.abs(data.imag) < thresh):
                numtype.append('real')
                data = data.real
            elif np.all(np.abs(data.real) < thresh) and np.any(np.abs(data.imag) >= thresh):
                numtype.append('imag')
                data = data.imag
            elif np.all(np.abs(data.real) < thresh) and np.all(np.abs(data.imag) < thresh):
                numtype.append('zero')
                continue
            else:
                numtype.append('cmplx')
            k_data.append([data, indices, indptr])

        if 'tens' not in kwargs:
            raise Exception(f"'tens' argument must be passed together with 'kmat'") from None
        else:
            tens = kwargs['tens']
        if tens_group_key(tens) not in J_grp:
            tens_J_grp = J_grp.create_group(tens_group_key(tens))
        else:
            tens_J_grp = J_grp[tens_group_key(tens)]

        if 'kmat' in tens_J_grp:
            del tens_J_grp['kmat']
            print(f"replace {tens} K-matrix for J1, J2 = {J1}, {J2} in file {filename} (thresh = {thresh})")
        else:
            print(f"write {tens} K-matrix for J1, J2 = {J1}, {J2} into file {filename} (thresh = {thresh})")

        dset = tens_J_grp.create_dataset('kmat', data=k_data)
        dset.attrs['irreps'] = [irrep for irrep, numt in zip(irreps, numtype)]
        dset.attrs['numtype'] = [numt for numt in numtype]
        dset.attrs['thresh'] = thresh

    except KeyError:
        pass

    # store M-matrix

    try:
        mmat = kwargs['mmat']

        for irrep, m_irrep in enumerate(mmat):
            assert (m_irrep.shape == (int(2*J1)+1, int(2*J2)+1)), \
                f"Dimensions of {irrep}-th element of 'mmat' = {m_irrep.shape} " + \
                f"do not match the values (2*J1+1, 2*J2+1) = {int(2*J1)+1, int(2*J2)+1} " +\
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
                f"Number of elements in 'irreps' = {len(irreps)} does not match " + \
                f"the number of elements in 'mmat' = {len(mmat)}"

        numtype = []
        m_data = []
        for m_irrep in mmat:
            data = m_irrep.data
            indices = m_irrep.indices
            indptr = m_irrep.indptr
            if np.any(np.abs(data.real) >= thresh) and np.all(np.abs(data.imag) < thresh):
                numtype.append('real')
                data = data.real
            elif np.all(np.abs(data.real) < thresh) and np.any(np.abs(data.imag) >= thresh):
                numtype.append('imag')
                data = data.imag
            elif np.all(np.abs(data.real) < thresh) and np.all(np.abs(data.imag) < thresh):
                numtype.append('zero')
                continue
            else:
                numtype.append('cmplx')
            m_data.append([data, indices, indptr])

        if 'tens' not in kwargs:
            raise Exception(f"'tens' argument must be passed together with 'mmat'") from None
        else:
            tens = kwargs['tens']
        if tens_group_key(tens) not in J_grp:
            tens_J_grp = J_grp.create_group(tens_group_key(tens))
        else:
            tens_J_grp = J_grp[tens_group_key(tens)]

        if 'mmat:'+cart in tens_J_grp:
            del tens_J_grp['mmat:'+cart]
            print(f"replace {tens} M-matrix({cart}) for J1, J2 = {J1}, {J2} in file {filename} (thresh = {thresh})")
        else:
            print(f"write {tens} M-matrix({cart}) for J1, J2 = {J1}, {J2} into file {filename} (thresh = {thresh})")

        dset = tens_J_grp.create_dataset('mmat:'+cart, data=m_data)
        dset.attrs['irreps'] = [irrep for irrep, numt in zip(irreps, numtype)]
        dset.attrs['numtype'] = [numt for numt in numtype]
        dset.attrs['thresh'] = thresh
        dset.attrs['cart'] = cart

    except KeyError:
        pass


def inspect_tensors(filename):
    """Returns list of Cartesian tensors (matrix elements) stored in HDF5 file,
    together with the list of coupled J quanta.

    Args:
        filename : str
            Name of the output HDF5 file.

    Returns:
        tens : dict
            Dictionary with keys as names of Cartesian tensors and values as
            lists of pairs of J quanta coupled by the tensor.
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
                tensors[tens] = [(J1, J2)]
    return tensors


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
