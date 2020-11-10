import numpy as np
from scipy.sparse import coo_matrix
import sys


planck = 6.62606896e-27 # Planck constant
avogno = 6.0221415e+23  # Avogadro constant
vellgt = 2.99792458e+10 # Speed of light constant
boltz  = 1.380658e-16   # Boltzmann constant


def read_states_field(fname):
    """Reads field-dressed states from a file.

    Args:
        fname (str): The name of file with field-dressed energies.

    Returns:
        states : numpy named array states['name'][istate]
            'name' can take the following values:
                'id' - integer state ID number,
                'enr' - state energy,
                'qstr' - string representing state quantum numbers.
            istate is the state running index.
    """
    with open(fname, "r") as fl:
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
    csrmat = []
    for ielem in range(nelem):
        coomat = coo_matrix( (val[:,ielem], (id1, id2)), shape=(nstates1+1,nstates2+1), dtype=np.float64 )
        csrmat.append(coomat.tocsr())
    return csrmat



def partition_function(states_fname, temperature, **kwargs):

    boltz_beta = planck * vellgt / (boltz * temperature)

    states = read_states_field(states_fname)

    if "zpe" in kwargs:
        zpe = kwargs["zpe"]
    else:
        zpe = np.min(states['enr'])

    return np.sum( np.exp(-(states['enr']-zpe) * boltz_beta) )



def dipole_intens_field(states_fname, matelem_fname, temperature, **kwargs):

    if "partfunc" in kwargs:
        partfunc = kwargs["partfunc"]
    else:
        partfunc = 1.0

    if "ints_tol" in kwargs:
        ints_tol = kwargs["ints_tol"]
    else:
        ints_tol = 1e-34

    intens_cm_molecule = 8.0e-36*np.pi**3/(3.0*planck*vellgt)
    boltz_beta = planck * vellgt / (boltz * temperature)

    states = read_states_field(states_fname)

    if "zpe" in kwargs:
        zpe = kwargs["zpe"]
    else:
        zpe = np.min(states['enr'])

    linestr = read_matelem_field(matelem_fname)

    if len(linestr)>1:
        sys.exit(f"Matrix elements file {matelem_fname} has more than a single data column, " \
                +f"so this file probably contains something else but not the linestrength" )

    ls = linestr[0]
    nonzero_ind = np.array(ls.nonzero()).T

    elow = np.array([ states['enr'].take(rowcol[0]) if states['enr'].take(rowcol[0]) < states['enr'].take(rowcol[1]) \
                      else states['enr'].take(rowcol[1]) for rowcol in nonzero_ind ])

    freq = np.array([ abs( states['enr'].take(rowcol[0]) - states['enr'].take(rowcol[1]) ) \
                      for rowcol in nonzero_ind ])

    qstr = [ ( states['qstr'].take(rowcol[0]), states['qstr'].take(rowcol[1]) ) \
              if states['enr'].take(rowcol[0]) < states['enr'].take(rowcol[1]) \
              else ( states['qstr'].take(rowcol[1]), states['qstr'].take(rowcol[0]) ) \
              for rowcol in nonzero_ind ]
    print(states['qstr'])

    boltz_fac = np.exp(-(elow-zpe) * boltz_beta) / partfunc

    intens = ls.data * boltz_fac * freq * (1.0-np.exp(-abs(freq)*boltz_beta)) * intens_cm_molecule

    # print(" ".join("%s"%q[0] + "%s"%q[1] + "%16.8f"%nu + "%16.8e"%ints + "\n" \
    #         for q,nu,ints in zip(qstr,freq,intens) if ints>=ints_tol ) )


if __name__ == "__main__":

    partfunc = {"296.0":174.5813}
    temp = 296.0

    states_file = "energy_10000V_F_0_3_M_3_3.txt"
    linestr_file = "line_strength_10000V_F_0_3_M_3_3.txt"

    dipole_intens_field(states_file, linestr_file, temp, partfunc=partfunc[str(temp)])

