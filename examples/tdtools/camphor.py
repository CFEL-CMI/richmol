import numpy as np
import sys, copy
from richmol.tdtools import Psi, Etensor


if __name__ == "__main__":

    # read basis
    fname_enr = "etc/richmol_files_camphor/camphor_energies_j0_j20.rchm"
    psi = Psi(fname_enr, fmin=0, fmax=30, mmin=-30, mmax=30, dm=1, df=1, sym=['A','B1','B2','B3'])

    # inital wavepacket (f,m,id,ideg,coef)
    psi.j_m_id = (0, 0, 1, 1, 1.0)

    # read tensor(s)
    fname_tens = "etc/richmol_files_camphor/camphor_matelem_alpha_j<j1>_j<j2>.rchm"
    alpha = Etensor(fname_tens, psi)

    # time grid for time in ps
    time_grid = np.linspace(0,0.3,11)

    # set up field (must be in units of V/m)
    E0 = 1e+10
    t0 = 100
    sigma = 200
    lightspeed = 299792458e0
    two_pi_c = 2.0 * np.pi * lightspeed * 1e-12 * 1e2 # in cm/ps
    omega = 12500*two_pi_c
    field_func = lambda t: E0*np.exp(-(t-t0)**2/(2*sigma**2))*np.cos(omega*t)
    field = [[0,0,field_func(t)] for t in time_grid]

    psi_t = copy.deepcopy(psi)
    psi_a = copy.deepcopy(psi)
    psi_l = copy.deepcopy(psi)
    for it in range(len(time_grid) - 1):
        print("{:03d}".format(it) + "   t = " + str(time_grid[it]) + " -> " + str(time_grid[it + 1]) + " picosec")

        t_in, t_out = time_grid[it], time_grid[it + 1]
        hamiltonian = -0.5 * alpha * field[it] # NOTE: this only accounts for perturbation Hamiltonian
                                               #       the field-free part is in psi.energy[f]

        psi_t2 = hamiltonian.U(t_in, t_out, psi_t, method="taylor")
        psi_a2 = hamiltonian.U(t_in, t_out, psi_a, method="arnoldi")
        psi_l2 = hamiltonian.U(t_in, t_out, psi_l, method="lanczos")

        conv = sum([np.sum(np.abs(psi_t2.coefs[f] - psi_a2.coefs[f])**2) for f in psi.coefs.keys()])
        print("   conv_tay_arn = " + str(conv))

        conv = sum([np.sum(np.abs(psi_a2.coefs[f] - psi_l2.coefs[f])**2) for f in psi.coefs.keys()])
        print("   conv_arn_lan = " + str(conv))

        conv = sum([np.sum(np.abs(psi_l2.coefs[f] - psi_t2.coefs[f])**2) for f in psi.coefs.keys()])
        print("   conv_lan_tay = " + str(conv))

        psi_t = psi_t2
        psi_a = psi_a2
        psi_l = psi_l2
