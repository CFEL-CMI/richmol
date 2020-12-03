import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


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

        kwargs:
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
        assert (round(float(J1), 1) == round(float(J2), 1)), \
            f"Bra and ket J1 and J2 quanta (= {J1} and {J2}) must be equal " + \
            f"to each other when storing {argname}"

    def _check_dim(dim, argname):
        if "dim" in J_grp:
            dim_ = J_grp['dim'][()]
            assert (all(dim == dim_)), \
                f"Dimensions of '{argname}' = {dim} do not match " + \
                f"basis set dimensions = {dim_} already stored in file {filename}"
        else:
            dset = J_grp.create_dataset('dim', data=dim)


    fl = h5py.File(filename, {False:"w", True:"a"}[replace])

    key = str(round(float(J1), 1)) + "_" + str(round(float(J2), 1))
    if key not in fl:
        J_grp = fl.create_group(key)
    else:
        J_grp = fl[key]

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
        if tens not in J_grp:
            tens_J_grp = J_grp.create_group(tens)
        else:
            tens_J_grp = J_grp[tens]

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
        if tens not in J_grp:
            tens_J_grp = J_grp.create_group(tens)
        else:
            tens_J_grp = J_grp[tens]

        if 'mmat_'+cart in tens_J_grp:
            del tens_J_grp['mmat_'+cart]
            print(f"replace {tens} M-matrix({cart}) for J1, J2 = {J1}, {J2} in file {filename} (thresh = {thresh})")
        else:
            print(f"write {tens} M-matrix({cart}) for J1, J2 = {J1}, {J2} into file {filename} (thresh = {thresh})")

        dset = tens_J_grp.create_dataset('mmat_'+cart, data=m_data)
        dset.attrs['irreps'] = [irrep for irrep, numt in zip(irreps, numtype)]
        dset.attrs['numtype'] = [numt for numt in numtype]
        dset.attrs['thresh'] = thresh
        dset.attrs['cart'] = cart

    except KeyError:
        pass




if __name__ == "__main__":
    import random
    import string

    nstates = 500
    assign = []
    id = []

    for istate in range(nstates):
        assign.append( "".join(random.choice(string.ascii_lowercase) for i in range(10)) )
        id.append([istate,1])

    enr = np.random.uniform(low=0.5, high=13.3, size=(500,))


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
    store("test_file.h5", 1, 1, replace=True, mmat=kmat, irreps=irreps, cart="xx", tens='alpha')
