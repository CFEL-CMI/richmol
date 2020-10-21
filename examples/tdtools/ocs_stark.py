from richmol.tdtools import Psi, Etensor
import numpy as np
from scipy import sparse

#####################################################
# Example of building the total Hamiltonian matrix
# for OCS molecule placed in static electric field
#####################################################


def hmatrix(ham, psi):
    """Test routine that returns matrix representation of tensor 'ham' (Etensor)
    in the field-free basis 'psi' (Psi).
    The field-free energies are added to the diagonal of tensor matrix.
    """
    prefac = ham.prefac
    flist = psi.flist
    hmat = {(f1,f2) : np.zeros( (len(psi.quanta[f1]), len(psi.quanta[f2])), dtype=np.complex128 ) \
            for f1 in flist for f2 in flist}
    for fkey in list(set(ham.MF.keys()) & set(ham.K.keys())):
        for mm,kk in zip(ham.MF[fkey], ham.K[fkey]):
            hmat[fkey] += np.kron(mm, kk.todense()) * prefac

    for f in flist:
        hmat[(f,f)] += np.diag(psi.energy[f])

    totmat = np.block([[hmat[(f1,f2)] for f2 in flist] for f1 in flist])
    return totmat


if __name__ == "__main__":

    # read basis states
    # fname_enr = '../../database/OCS/old_watie/ocs_energies_j0_j30.rchm'
    fname_enr = '../../database/OCS/OCS_energies_j0_j30.rchm'
    psi = Psi(fname_enr, fmin=0, fmax=30, mmin=-30, mmax=30, dm=1, df=1)

    # read tensor matrix elements (e.g., dipole moment)
    # dipole_me = '../../database/OCS/old_watie/ocs_matelem_mu_j<j1>_j<j2>.rchm'
    dipole_me = '../../database/OCS/OCS_mu_j<j1>_j<j2>.rchm'
    dipole = Etensor(dipole_me, psi)

    # static dc field along Z axis in units of V/m
    field = [0,0, 1e4 * 100] # [x,y,z] field components

    # dipole interaction Hamiltonian
    # note that field must be in units V/m
    H = -1.0 * dipole * field

    # compute and diagonalize Hamiltonian using the above function
    hmat = hmatrix(H, psi)
    diag, _ = np.linalg.eigh(hmat)

    # compute and diagonalize Hamiltonian using the in-built method
    hmat2 = H.matrix(psi, plus_diag=True)
    diag2, _ = np.linalg.eigh(hmat2)

    # print and compare energies
    for e,e2 in zip(diag,diag2):
        print(e,abs(e-e2))

    # compute matrix elements of dipole in the field-free basis
    dipole_me = [dipole.matrix(psi, ix=ix) for ix in range(dipole.ncart)]
