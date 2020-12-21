from richmol.extfield import States, Tensor, Hamiltonian, mu_au_to_Cm, planck, c_vac, NoCouplingError
from richmol import rchm
import numpy as np
import sys

#######################################################################
# example of OCS placed in static electric field,
# here we use richmol data in ../convert_formats obtained by converting
# old formatted richmol files (*.rchm) into hdf5 file
#######################################################################

richmol_file = '../convert_formats/OCS_j0_j30.h5'

# list tensors stored in richmol file
tens = rchm.inspect_tensors(richmol_file)
for key in tens.keys():
    print("\n", key, tens[key])
#sys.exit()

# Z-electric field strengths
field_z = 100000 * 100 # 100 kV/cm in units V/m

enr_list = []
Jmax = 30

# since field is along Z axis, states with different m quanta are not coupled
# and can be treated separately
for m in range(-Jmax, Jmax+1):

    # field-free states
    states = States(richmol_file, 'h0', [J for J in range(Jmax + 1)], emin=0, emax=10000, sym=['A'], \
                    m_list=[m], verbose=True)

    # dipole matrix elements
    try:
        mu = Tensor(richmol_file, 'dipole', states, states, verbose=True)
        mu.mul(-1.0)

        # external field in V/m
        field = [0, 0, field_z]

        # multiply dipole with external field
        mu.field(field)

        # convert dipole[au]*field[V/m] into [cm^-1]
        fac = mu_au_to_Cm / (planck * c_vac) / 100.0
        mu.mul(fac)

        # combine -dipole*field with field-free Hamiltonian
        ham = Hamiltonian(mu=mu, h0=states)

    except NoCouplingError:

        print(f"*** Note, for m = {m} there is no coupling with field due to basis truncation at J = {Jmax}")
        ham = Hamiltonian(h0=states)

    hmat, Jlist = ham.tomat(form='mat')

    enr, vec = np.linalg.eigh(hmat)

    # keep energies and assignments
    assign = states.ind_assign([i for i in range(len(enr))], Jlist) # basis state assignments
    for i, e in enumerate(enr):
        ind = np.argmax(abs(vec[:,i])) # index of a leading basis contribution
        enr_list.append([e, abs(vec[ind,i])**2, *assign[ind]])

# write energies and assignments for all states into a file
enr_list.sort(key=lambda x: x[0])
enr_file = open("stark_energies_fz" + str(round(field_z,1)) + "_conv_formats.txt", 'w')
for state in enr_list:
    enr, c2, J, m, enr0, sym, assign = state
    enr_file.write(" %12.6f"%enr + "  %6.4f"%c2 + " %6.1f"%J + " %6.1f"%m + " %10.4f"%enr0 + \
                   " %3s"%sym + " %s"%assign + "\n")
enr_file.close()

