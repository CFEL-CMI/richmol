import numpy as np
import sys
from richmol.tdtools import Psi, Etensor


if __name__ == "__main__":

    # read basis
    fname_enr = "../../data/richmol_files_camphor/camphor_energies_j0_j20.rchm"
    psi = Psi(fname_enr, fmin=0, fmax=30, mmin=-30, mmax=30, dm=1, df=1, sym=['A','B1','B2','B3'])

    # inital wavepacket (f,m,id,ideg,coef)
    psi.j_m_id = (0, 0, 1, 1, 1.0)

    # read tensor(s)
    fname_tens = "../../data/richmol_files_camphor/camphor_matelem_alpha_j<j1>_j<j2>.rchm"
    alpha = Etensor(fname_tens, psi)

    # time grid for time in ps
    time_grid = np.linspace(0,300,(300)/0.01+1)

    # set up field (must be in units of V/m)
    E0 = 1e+10
    t0 = 100
    sigma = 200
    lightspeed = 299792458e0
    two_pi_c = 2.0 * np.pi * lightspeed * 1e-12 * 1e2 # in cm/ps
    omega = 12500*two_pi_c
    field_func = lambda t: E0*np.exp(-(t-t0)**2/(2*sigma**2))*np.cos(omega*t)
    field = [[0,0,field_func(t)] for t in time_grid]

    for it,t in enumerate(time_grid):
        print(it,t)
        hamiltonian = -0.5 * alpha * field[it] # NOTE: this only accounts for perturbation Hamiltonian
                                               #       the field-free part is in psi.energy[f]
        #psi2 = hamiltonian * psi
        #psi = psi2
        psi2 = hamiltonian.U(0.02, 0.01, psi, method="taylor")
        psi = psi2
        #sys.exit()

