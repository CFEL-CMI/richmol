from richmol.tdtools import Psi, Etensor
import numpy as np
from scipy import sparse
import sys

#####################################################
# Example of building the total Hamiltonian matrix
# for OCS molecule placed in static electric field
#####################################################


if __name__ == "__main__":

    # read basis states
    # fname_enr = '../../database/OCS/old_watie/ocs_energies_j0_j30.rchm'
    fname_enr = '../../database/OCS/OCS_energies_j0_j30.rchm'

    fmax = 29
    diag = {}
    vec = {}

    for m in range(-fmax, fmax+1):

        print(f"\n\n Run m = {m}")
        psi = Psi(fname_enr, fmin=abs(m), fmax=30, mmin=m, mmax=m, dm=1, df=1)

        # read tensor matrix elements (e.g., dipole moment)
        # dipole_me = '../../database/OCS/old_watie/ocs_matelem_mu_j<j1>_j<j2>.rchm'
        dipole_me = '../../database/OCS/OCS_mu_j<j1>_j<j2>.rchm'
        dipole = Etensor(dipole_me, psi, psi)

        # static dc field along Z axis in units of V/m
        field = [0,0, 1e4 * 100] # [x,y,z] field components

        # dipole interaction Hamiltonian
        # note that field must be in units V/m
        H = -1.0 * dipole * field

        # compute and diagonalize Hamiltonian 
        hmat = H.hmat(plus_diag=True)
        diag[m], vec[m] = np.linalg.eigh(hmat)

        # print energies and assignments
        for i,e in enumerate(diag[m]):
            v = vec[m][:,i]
            ind = (-abs(v)**2).argsort()[0] # index of largest coefficient
            c2 = abs(v[ind])**2
            f = psi.f[ind]
            _m = psi.m[ind]
            istate = psi.istate[ind]
            quanta = psi.states[f]['qstr'][istate]
            print(i, e, c2, '[', f, _m, quanta, ']')

    # compute matrix elements of dipole

    for m1 in range(-fmax, fmax+1):

        psi1 = Psi(fname_enr, fmin=abs(m1), fmax=30, mmin=m1, mmax=m1, dm=1, df=1)

        for m2 in range(-fmax, fmax+1):

            if abs(m1-m2)>1: continue

            psi2 = Psi(fname_enr, fmin=abs(m2), fmax=30, mmin=m2, mmax=m2, dm=1, df=1)

            print(f"compute pair {m1} / {m2}")

            dipole_me = '../../database/OCS/OCS_mu_j<j1>_j<j2>.rchm'
            dipole = Etensor(dipole_me, psi1, psi2)

            dipole_me = [dipole.mat(ix=ix) for ix in range(dipole.ncart)]

            # transform dipole matrix elements to eigenbasis

            dme = [np.dot(np.conjugate(vec[m1]), np.dot(me, vec[m2])) for me in dipole_me]
