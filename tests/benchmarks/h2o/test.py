from richmol.rot.rchm import read_states, read_trans, add_tensor
from richmol.convert_units import DebyeVm_to_invcm
import json


def test_old_format(filename_states, filename_trans):
    """Reads old-format Richmol files"""
    states = read_states(filename_states, mlist=[-4,-3,-2,-1,1,2,3], jmin=1)
    tens = read_trans(states, filename_trans)
    for elem in dir(tens):
        #print(elem, getattr(tens, elem))
        print(elem)

    add_tensor('random_tensor.h5', tens, replace=True)


def stark_energies(states, trans):
    """Computes Stark energies"""
    h0 = read_states(states, jmin=0, jmax=5)
    mu = read_trans(h0, trans)
    field = [1,2,3]
    mu.field(field)
    mu.mul(-1.0)
    fac = DebyeVm_to_invcm()
    mu.mul(fac)

if __name__ == '__main__':
    path = "tests/benchmarks/h2o/trove_rchm/"
    states = path + "energies_j0_j40_MARVEL_HITRAN.rchm"
    trans = path + "matelem_MU_j<j1>_j<j2>.rchm"

    stark_energies(states, trans)
