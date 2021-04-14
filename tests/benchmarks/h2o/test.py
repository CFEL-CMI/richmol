from richmol.rot.rchm import read_states, read_trans, add_tensor
import json


def test_old_format(filename_states, filename_trans):
    """Reads old-format Richmol files"""
    states = read_states(filename_states, mlist=[-4,-3,-2,-1,1,2,3], jmin=1)
    tens = read_trans(states, filename_trans)
    for elem in dir(tens):
        #print(elem, getattr(tens, elem))
        print(elem)

    add_tensor('random_tensor.h5', tens, replace=True)


if __name__ == '__main__':
    path = "tests/benchmarks/h2o/trove_rchm/"
    states = path + "energies_j0_j40_MARVEL_HITRAN.rchm"
    trans = path + "matelem_MU_j<j1>_j<j2>.rchm"
    test_old_format(states, trans)