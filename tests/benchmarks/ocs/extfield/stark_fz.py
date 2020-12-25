from richmol.extfield import States, Tensor, Hamiltonian, mu_au_to_Cm, planck, c_vac, NoCouplingError
from richmol import rchm
import numpy as np
import sys

###################################################################
# Example of calculation of rotational energies and dipole spectrum
# for OCS placed in static electric field.
# Apply static field along Z axis.
###################################################################

richmol_file = '../watie/OCS_j0_j30.h5'

# print out list tensors stored in richmol file
tens = rchm.inspect_tensors(richmol_file)
for key in tens.keys():
    print("\n", key, tens[key])
# sys.exit()

# static electric field along Z axis
field_z = 100000 * 100 # 100 kV/cm in units V/m

# max J spanned by basis (max value is defined by richmol_file)
Jmax = 30

# compute energies and field-dressed wave functions

# field-free states
states = States(richmol_file, 'h0', [J for J in range(Jmax + 1)], emin=0, emax=10000, sym=['A'], verbose=True)

# dipole matrix elements
mu = Tensor(richmol_file, 'mu', states, states, verbose=True)
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

# matrix representation of Hamiltonian
hmat = ham.tomat(form='full')

# eigenvalues and eigenvectors
enr, vec = np.linalg.eigh(hmat)

# assignment of basis states
assign = states.ind_assign([i for i in range(len(enr))])

# assignment of field-dressed sates
state_assign = []
for istate in range(len(enr)):
    ind = np.argmax(abs(vec[:,istate]))
    state_assign.append((enr[istate], abs(vec[ind,istate])**2, *assign[ind]))

# sort and store field-dressed state energies and assignments in file

states_list = [assign for assign in state_assign]
states_list.sort(key=lambda x: x[0])

enr_file = open("stark_energies_fz" + str(round(field_z,1)) + ".txt", 'w')
for state in states_list:
    enr, c2, J, m, enr0, sym, assign = state
    enr_file.write(" %12.6f"%enr + "  %6.4f"%c2 + " %6.1f"%J + " %6.1f"%m + " %10.4f"%enr0 + \
                   " %3s"%sym + " %s"%assign + "\n")
enr_file.close()

# compute matrix elements of dipole moment between field-dressed states
# and print out transition linestrenths

tran_file = open("stark_transitions_fz" + str(round(field_z,1)) + ".txt", 'w')

mu = Tensor(richmol_file, 'mu', states, states)

# primitive matrix elements of mu_X, mu_Y, mu_Z
matx = mu.tomat(form='full', cart='x')
maty = mu.tomat(form='full', cart='y')
matz = mu.tomat(form='full', cart='z')

# transform matrix elements to eigenfunctions representation
matx = np.square(np.abs(np.dot(np.conjugate(vec).T, np.dot(matx, vec))))
maty = np.square(np.abs(np.dot(np.conjugate(vec).T, np.dot(maty, vec))))
matz = np.square(np.abs(np.dot(np.conjugate(vec).T, np.dot(matz, vec))))

# linestrength (without Gns factors)
linestr =  matx + maty + matz

# inefficient but simple printout of transitions with linestrengths
for i in range(linestr.shape[0]):
    for j in range(i):
        if linestr[i,j] > 1e-12:
            enr1, c1, J1, m1, e1, sym1, assign1 = state_assign[i]
            enr2, c2, J2, m2, e2, sym2, assign2 = state_assign[j]
            tran_file.write(" %12.6f"%enr1 + " %6.1f"%J1 + " %6.1f"%m1 + " %3s"%sym1 + " %s"%assign1 + \
                            " %12.6f"%enr2 + " %6.1f"%J2 + " %6.1f"%m2 + " %3s"%sym2 + " %s"%assign2 + \
                            "   %12.6f"%(enr1-enr2) + "   %16.6e"%linestr[i,j] + "\n")
tran_file.close()


