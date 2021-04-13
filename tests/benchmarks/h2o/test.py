from richmol.rot.rchm import read_states

def test_old_format(filename):
    """Reads old-format Richmol files"""
    tens = read_states(filename, mlist=[-4,-3,-2,-1,1,2,3], jmin=1)
    for elem in dir(tens):
        print(elem, getattr(tens, elem))


if __name__ == '__main__':
    test_old_format("tests/benchmarks/h2o/trove_rchm/energies_j0_j40_MARVEL_HITRAN.rchm")