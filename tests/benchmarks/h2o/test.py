from richmol.field import CarTens
from richmol.convert_units import DebyeVm_to_invcm


def filter(**kwargs):
    pass_J = True
    pass_enr = True
    pass_m = True
    if "J" in kwargs:
        J = kwargs["J"]
        pass_J = J <= 5
    if "m" in kwargs:
        m = round(float(kwargs["m"]),1)
        pass_m = m == _m
    if "enr" in kwargs:
        enr = kwargs["enr"]
        pass_enr = enr <= 2000
    return pass_J * pass_enr * pass_m


if __name__ == '__main__':

    path = "/home/andrey/projects/richmol/tests/data/h2o_rchm_files_TROVE/"
    # states file
    states_file = path + "energies_j0_j40_MARVEL_HITRAN.rchm"
    # template for generating names of matrix elements files for different bra and ket J quanta
    matelem_file = path + "matelem_MU_j<j1>_j<j2>.rchm"

    fac = -1.0 * DebyeVm_to_invcm()
    _m = 3.0
    h0 = CarTens(states_file, bra=filter, ket=filter)
    mu = CarTens(states_file, matelem=matelem_file, bra=filter, ket=filter)
    mu.mul(fac)
    field = [0, 0, 1e6] # in V/m
    mu.field(field)
    mat = mu.tomat(form='full', sparse='coo_matrix')
    print(mat)
    h0.field(field)
    print(h0.mfmat.keys())
    h = h0 + mu

