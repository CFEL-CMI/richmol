import numpy as np
import sys


def read_states_field(fname):
    """Reads field-dressed states from a file.

    Parameters
    ----------
    fname : str
        The name of file with field-dressed energies.

    Returns
    -------
    states : numpy named array states['name'][istate]
        'name' can take the following values:
            'id' - integer state ID number,
            'enr' - state energy,
            'qstr' - string representing state quantum numbers.
        istate is the state running index.
    """
    with open(filename, "r") as fl:
        lines = fl.readlines()

    nstates = len(lines)
    states = np.zeros(nstates, dtype={'names':('id', 'enr', 'qstr'), 'formats':('i8', 'f8', 'U30')})

    for istate,line in enumerate(lines):
        w = line.split()
        id = np.int64(w[0])
        enr = np.float64(w[1])
        qstr = ' '.join([w[i] for i in range(2,len(w))])
        states['id'][istate] = id
        states['enr'][istate] = enr
        states['qstr'][istate] = qstr

    return states


def read_matelem_field(fname):
    """ Reads matrix elements of an operator in the basis of field-dressed states """
    fl = open(fname, "r")
    id1 = []
    id2 = []
    val = []
    for line in fl:
        w = line.split()
        id1.append(np.int64(w[0]))
        id2.append(np.int64(w[1]))
        val.append([np.float64(ww) for ww in w[2:]])
    fl.close()
    nstates1 = max(id1)
    nstates2 = max(id2)
    val = np.array(val, dtype=np.float64)
    nelem = val.shape[1]
    for ielem in nelem:
        coomat = coo_matrix( (val[:,ielem], (id1, id2)), shape=(nstates1+1,nstates2+1), dtype=np.float64 )
        csrmat.append(coomat.tocsr())



def intens_field(states_fname, matelem_fname):

    states = read_states_field(states_fname)
    matelem = read_matelem_field(matelem_fname)

if __name__ == "__main__":

    gns = {"A1":1.0,"A2":1.0,"B1":3.0,"B2":3.0}
    partfunc = {"296.0":174.5813}
    temp = 296.0

    intens_field()

