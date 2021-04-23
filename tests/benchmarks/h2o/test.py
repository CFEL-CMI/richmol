from richmol import field
from richmol.convert_units import DebyeVm_to_invcm


def test_old_format(filename_states, filename_trans):
    """Reads old-format Richmol files"""
    states = field.CarTens(states=filename_states)

    for elem in dir(states):
        print(elem)

    for J in states.kmat.keys():
        for sym in states.kmat[J].keys():
            for irrep in states.kmat[J][sym].keys():
                print(J, sym, irrep)#, states.kmat[J][sym][0].diagonal())


def stark_energies(states, trans):
    """Computes Stark energies"""
    #h0 = read_states(states, jmin=0, jmax=5)
    #mu = read_trans(h0, trans)
    field = [1,2,3]
    mu.field(field)
    mu.mul(-1.0)
    fac = DebyeVm_to_invcm()
    mu.mul(fac)

if __name__ == '__main__':
    path = "tests/benchmarks/h2o/trove_rchm/"
    states = path + "energies_j0_j40_MARVEL_HITRAN.rchm"
    trans = path + "matelem_MU_j<j1>_j<j2>.rchm"

    test_old_format(states, trans)
