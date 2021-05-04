"""Calculate Stark energies of water molecule using the old-format richmol files produced by TROVE
(see "tests/benchmarks/data/h2o_rchm_files_TROVE/"), and compare the Stark shifts and the field-dressed
dipole matrix elements with the data calculated using older version of richmol
(see "tests/benchmarks/data/h2o_rchm_files_TROVE/richmol2_results")
"""
import numpy as np
from richmol.field import CarTens, filter
from richmol.convert_units import DebyeVm_to_invcm
import sys
import matplotlib.pyplot as plt
import gzip


Jmax = 5.0

def filter1(**kwargs):
    """Filter function to select bra states"""
    pass_J = True
    pass_enr = True
    pass_m = True
    if "J" in kwargs:
        J = kwargs["J"]
        pass_J = J <= Jmax
    if "m" in kwargs:
        m = round(float(kwargs["m"]),1)
        pass_m = m == m1
    if "enr" in kwargs:
        enr = kwargs["enr"]
        pass_enr = enr <= 10000
    return pass_J * pass_enr * pass_m


def filter2(**kwargs):
    """Filter function to select ket states"""
    pass_J = True
    pass_enr = True
    pass_m = True
    if "J" in kwargs:
        J = kwargs["J"]
        pass_J = J <= Jmax
    if "m" in kwargs:
        m = round(float(kwargs["m"]),1)
        pass_m = m == m2
    if "enr" in kwargs:
        enr = kwargs["enr"]
        pass_enr = enr <= 10000
    return pass_J * pass_enr * pass_m


if __name__ == "__main__":

    print(__doc__)

    path = "tests/benchmarks/data/h2o_rchm_files_TROVE/"

    # states file
    states_file = path + "energies_j0_j40_MARVEL_HITRAN.rchm"

    # template for generating names of matrix elements files for different bra and ket J quanta
    matelem_file = path + "matelem_MU_j<j1>_j<j2>.rchm"

    # converts dipole[Debye] * field[V/m] into energy[cm^-1]
    fac = DebyeVm_to_invcm()

    enr = {}
    vec = {}

    for m1 in np.linspace(-Jmax, Jmax, int(2*Jmax)+1):

        print(f"Run Stark energy calculations for m = {m1}")

        # stationary states
        h0 = CarTens(states_file, bra=filter1, ket=filter1)

        # matrix elements of dipole moment
        mu = CarTens(states_file, matelem=matelem_file, bra=filter1, ket=filter1)

        # multiply dipole with field
        mu.mul(fac)
        field = [0, 0, 1e6] # in V/m
        mu.field(field)

        # total Hamiltonian
        h = h0 - mu

        # diagonalize Hamiltonian
        mat = h.tomat(form='full')
        print("  matrix is hermitian? :", np.max(np.abs(mat - np.conjugate(mat).T)) < 1e-14)
        e, v = np.linalg.eigh(mat)
        enr[m1] = [elem for elem in e]
        vec[m1] = v

    # read Stark energies obtained with older version of richmol

    filename = path + "richmol2_results/stark_energies_10000_0.0"
    enr0 = {}
    with open(filename, "r") as fl:
        for line in fl:
            w = line.split()
            m = round(float(w[0]),1)
            try:
                enr0[m].append(float(w[-1]))
            except KeyError:
                enr0[m] = [float(w[-1])]

    # compare Stark energies

    print("Print max differences in Stark energies for different m quanta")
    for m in enr.keys():
        e = sorted(enr[m])
        e0 = sorted(enr0[m])
        diff = max([abs(i-j) for i,j in zip(e,e0)])
        print(m, diff)

    input("Press enter to continue to dipole matrix elements")

    filename = "transitions_fz" + str(round(field[2],1)) + "_diag_m.txt"
    print(f"field-dressed transitions will be printed into file {filename}")

    mu0 = CarTens(states_file, matelem=matelem_file, thresh=1e-6)

    with open(filename, 'w') as fl:

        for m1 in np.linspace(-Jmax, Jmax, int(2*Jmax)+1):
            for m2 in np.linspace(-Jmax, Jmax, int(2*Jmax)+1):
                if abs(m1-m2)>1: continue

                print(f"compute dipole matrix elements for (m1, m2) = {(m1, m2)}")

                # matrix elements of dipole moment
                mu = filter(mu0, bra=filter1, ket=filter2, thresh=1e-6)

                # bra and ket state assignments
                assign1, assign2 = mu.assign(form='full')

                mux = mu.tomat(form='full', sparse='csr_matrix', cart='x', thresh=1e-6)
                muy = mu.tomat(form='full', sparse='csr_matrix', cart='y', thresh=1e-6)
                muz = mu.tomat(form='full', sparse='csr_matrix', cart='z', thresh=1e-6)

                print("  mux, muy, muz number of nonzero elements:", mux.nnz, muy.nnz, muz.nnz)

                # transform matrix elements to eigenfunctions representation
                mux = np.dot(np.conjugate(vec[m1]).T, mux.dot(vec[m2]))
                muy = np.dot(np.conjugate(vec[m1]).T, muy.dot(vec[m2]))
                muz = np.dot(np.conjugate(vec[m1]).T, muz.dot(vec[m2]))

                # linestrength (without nuclear spin statistical weights)
                linestr = np.square(np.abs(mux)) + np.square(np.abs(muy)) + np.square(np.abs(muz))

                # super inefficient but simple printout of transitions with linestrengths
                for i in range(linestr.shape[0]):

                    ind = np.argmax(np.abs(vec[m1][:,i]))
                    (k1, e1), sym1, J1 = (assign1[key][ind] for key in ("k", "sym", "J"))

                    for j in range(linestr.shape[1]):

                        if m1 == m2 and j>=i:
                            continue

                        if linestr[i,j] < 1e-12:
                            continue

                        enr1 = enr[m1][i]
                        enr2 = enr[m2][j]

                        if enr2 > enr1:
                            continue

                        ind = np.argmax(np.abs(vec[m2][:,j]))
                        (k2, e2), sym2, J2 = (assign2[key][ind] for key in ("k", "sym", "J"))

                        fl.write(" %6.1f"%J1 + " %6.1f"%m1 + " %12.6f"%enr1 + " %3s"%sym1 + " %s"%k1 + \
                                 " %6.1f"%J2 + " %6.1f"%m2 + " %12.6f"%enr2 + " %3s"%sym2 + " %s"%k2 + \
                                 " %12.6f"%(enr1-enr2) + "   %16.6e"%linestr[i,j] + "\n")

    input("Press enter to compare linestrengths")

    freq1 = []
    ls1 = []
    with open(filename,'r') as f:
        lines = f.readlines()
        for line in lines:
            w = line.split()
            freq1.append(float(w[24]))
            ls1.append(float(w[25]))

    freq2 = []
    ls2 = []
    with gzip.open(path + "richmol2_results/MU_me.gz" ,'r') as f:
        lines = f.readlines()
        for line in lines:
            w = line.split()
            freq2.append(float(w[0]))
            ls2.append(-float(w[1]))

    plt.plot(freq1, ls1, 'o')
    plt.plot(freq2, ls2, 'o')
    plt.show()
