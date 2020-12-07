import numpy as np
import rchm
import warnings

# allow for repetitions of warning for the same source location
warnings.simplefilter('always', UserWarning)


class States():
    """Molecular field-free basis.

    Args:
        filename : str
            Name of richmol HDF5 file.
        J_list : list
            List of total angular momentum quanta spanned by the basis.
    Kwargs:
        emin, emax : float
            Minimal and maximal basis state energy.
        sym : list of str
            List of basis state symmetries.
        m_list : list
            List of m-quanta spanned by the basis.
        verbose : bool
            Set to True to print out some data.
    """

    def __init__(self, filename, J_list, **kwargs):
        """ Reads molecular energies and assignments from richmol HDF5 file,
        and generates basis set indices.
        """
        # read field free states from richmol file and apply state filters

        self.enr = {}
        self.id = {}
        self.sym = {}
        self.assign = {}
        self.J_list = []

        for J in J_list:
            descr, dim, enr, id, sym, assign = rchm.read_states(filename, J)
            if enr is None:
                raise Exception(f"File {filename} does not contain state energies ('enr' dataset)")
            if id is None:
                raise Exception(f"File {filename} does not contain state IDs ('id' dataset)")
            if sym is None:
                raise Exception(f"File {filename} does not contain state symmetries ('sym' dataset)")
            if assign is None:
                warnings.warn(f"File {filename} does not contain state assignments ('assign' dataset)")
            ind = [ i for i in range(len(enr))]
            if 'emin' in kwargs:
                ind = np.where(enr >= kwargs['emin'])
            if 'emax' in kwargs:
                ind = np.where(enr[ind] <= kwargs['emax'])
            if 'sym' in kwargs:
                ind = np.where(np.array(sym)[ind] in kwargs['sym'])
            if ind[0].shape[0] > 0:
                self.enr[J] = np.array(enr)[ind]
                self.id[J] = np.array(id)[ind, :]
                self.sym[J] = np.array(sym)[ind]
                self.assign[J] = np.array(assign)[ind]
                self.J_list.append(J)

        if len(self.J_list) == 0:
            raise Exception(f"State selection filters casted out all molecular states") from None

        J_out = [J for J in J_list if J not in self.J_list]
        if len(J_out) > 0:
            warnings.warn(f"State selection filters casted out all molecular states with J = {J_out}", stacklevel=2)

        # generate basis set indices

        if 'm_list' in kwargs:
            m_list = kwargs['m_list']
        else:
            m_list = [m for m in range(-max(self.J_list), max(self.J_list) + 1)]

        # print J and m quanta
        try:
            if kwargs['verbose'] == True:
                print(f"List of J quanta spanned by the basis: {self.J_list}")
                print(f"Number of molecular states and list of m quanta spanned by the basis for each J:")
                nonzero = False
                for J in self.J_list:
                    mlist = [m for m in range(-J, J + 1) if m in m_list]
                    print(f"    J = {J}, no.states = {len(self.enr[J])}, m = {mlist if len(mlist) > 0 else None}")
        except KeyError:
            pass

        for J in self.J_list:
            for m in (mm for mm in range(-J, J + 1) if mm in m_list):
                for istate in range(len(self.enr[J])):
                    print(J, m, istate, self.enr[J][istate])


class Tensor():
    """Matrix elements of molecular laboratory-frame Cartesian tensor operator
    in field-free basis.

    Args:
        filename : str
            Name of richmol HDF5 file.
        tens_name : str
            String identifying tensor, as stored in the HDF5 file.
        states : States
            Field-free basis.
    """
    def __init__(self, filename, tens_name, states, **kwargs):

        tens = rchm.inspect_tensors(filename)
        try:
            J_pairs = tens[tens_name]
            J_pairs = [(J1, J2) for J1 in states.J_list for J2 in states.J_list \
                        if (J1, J2) in J_pairs or (J2, J1) in J_pairs]
            if len(J_pairs) == 0:
                raise Exception(f"None of pairs of J quanta spanned by the basis set " + \
                    "were found in file {filename} for tensor {tens_name}") from None
            else:
                try:
                    if kwargs['verbose'] == True:
                        print(f"J-pairs for tensor {tens}: {J_pairs}")
                        print(f"selection rules |J-J'|: {set(abs(J1 - J2) for (J1, J2) in J_pairs)}")
                except KeyError:
                    pass

        except KeyError:
            raise KeyError(f"File {filename} does not contain tensor with name {tens_name}\n" + \
                f"list of stored tensors: {elem for elem in tens.keys()}") from None


if __name__ == '__main__':

    filename = '../examples/watie/OCS_energies_j0_j30.h5'
    a = States(filename, [i for i in range(31)], emin=10, emax=100, sym=['A'], m_list=[i for i in range(-10,11)], verbose=True)