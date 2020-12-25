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

# list tensors stored in richmol file
tens = rchm.inspect_tensors(richmol_file)
for key in tens.keys():
    print("\n", key, tens[key])
#sys.exit()

# Z-electric field strengths
field_z = 100000 * 100 # 100 kV/cm in units V/m

Jmax = 30

# compute field-dressed states
# since field is along Z axis, states with different m quanta are not coupled

enr = {}
vec = {}
assign = {}

for m in range(-Jmax, Jmax+1):

    # field-free states
    states = States(richmol_file, 'h0', [J for J in range(Jmax + 1)], emin=0, emax=10000, sym=['A'], \
                    m_list=[m], verbose=True)

    # dipole matrix elements
    try:
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

    except NoCouplingError:

        print(f"*** Note, for m = {m} there is no coupling with field due to basis truncation at J = {Jmax}")
        ham = Hamiltonian(h0=states)

    hmat = ham.tomat(form='full')

    enr[m], vec[m] = np.linalg.eigh(hmat)

    # assignment of basis functions
    bas_assign = states.ind_assign([i for i in range(len(enr[m]))]) # basis state assignments

    # assignment of field-dressed states
    assign[m] = []
    for i, e in enumerate(enr[m]):
        ind = np.argmax(abs(vec[m][:,i])) # index of a leading basis contribution
        assign[m].append([e, abs(vec[m][ind,i])**2, *bas_assign[ind]])

# write energies and assignments for all states into a file
state_assign = [elem for assgn in assign.values() for elem in assgn]
state_assign.sort(key=lambda x: x[0])
enr_file = open("stark_energies_fz" + str(round(field_z,1)) + "_diag_m.txt", 'w')
for state in state_assign:
    enr, c2, J, m, enr0, sym, assgn = state
    enr_file.write(" %12.6f"%enr + "  %6.4f"%c2 + " %6.1f"%J + " %6.1f"%m + " %10.4f"%enr0 + \
                   " %3s"%sym + " %s"%assgn + "\n")
enr_file.close()

# compute field-dressed transition linestrengths

tran_file = open("stark_transitions_fz" + str(round(field_z,1)) + "_diag_m.txt", 'w')

for m1 in range(-Jmax, Jmax+1):

    states1 = States(richmol_file, 'h0', [J for J in range(Jmax + 1)], emin=0, emax=10000, sym=['A'], m_list=[m1])

    for m2 in range(-Jmax, Jmax+1):
        if abs(m1 - m2) > 1: continue # selection rules

        states2 = States(richmol_file, 'h0', [J for J in range(Jmax + 1)], emin=0, emax=10000, sym=['A'], m_list=[m2])

        try:
            mu = Tensor(richmol_file, 'mu', states1, states2)
        except NoCouplingError:
            continue

        # primitive matrix elements of mu_X, mu_Y, mu_Z
        matx = mu.tomat(form='full', cart='x')
        maty = mu.tomat(form='full', cart='y')
        matz = mu.tomat(form='full', cart='z')

        # transform matrix elements to eigenfunctions representation
        matx = np.square(np.abs(np.dot(np.conjugate(vec[m1]).T, np.dot(matx, vec[m2]))))
        maty = np.square(np.abs(np.dot(np.conjugate(vec[m1]).T, np.dot(maty, vec[m2]))))
        matz = np.square(np.abs(np.dot(np.conjugate(vec[m1]).T, np.dot(matz, vec[m2]))))

        # linestrength (without Gns factors)
        linestr =  matx + maty + matz

        # inefficient but simple printout of transitions with linestrengths
        for i in range(linestr.shape[0]):
            for j in range(linestr.shape[1]):
                if m1 == m2 and j>=i: continue
                if linestr[i,j] > 1e-12:
                    enr1, c1, J1, mm1, e1, sym1, assign1 = assign[m1][i]
                    enr2, c2, J2, mm2, e2, sym2, assign2 = assign[m2][j]
                    if enr2 > enr1: continue
                    tran_file.write(" %12.6f"%enr1 + " %6.1f"%J1 + " %6.1f"%mm1 + " %3s"%sym1 + " %s"%assign1 + \
                                    " %12.6f"%enr2 + " %6.1f"%J2 + " %6.1f"%mm2 + " %3s"%sym2 + " %s"%assign2 + \
                                    "   %12.6f"%(enr1-enr2) + "   %16.6e"%linestr[i,j] + "\n")
tran_file.close()


