"""An example of calculation of the laser kick alignment of OCS molecule
for different pulse durations and selected initial temperature
"""

from richmol.rot import Molecule, solve, LabTensor
from richmol.tdse import TDSE
from richmol.convert_units import AUpol_x_Vm_to_invcm
import numpy as np
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt


if __name__ == "__main__":


    # Field-free rotational solutions


    ocs = Molecule()

    ocs.XYZ = ("angstrom",
               "C",  0.0,  0.0,  -0.522939783141,
               "O",  0.0,  0.0,  -1.680839357,
               "S",  0.0,  0.0,  1.037160128)

    # molecular-frame dipole moment (in au)
    ocs.dip = [0, 0, -0.31093]

    # molecular-frame polarizability tensor (in au)
    ocs.pol = [[25.5778097, 0, 0], [0, 25.5778097, 0], [0, 0, 52.4651140]]

    Jmax = 10
    sol = solve(ocs, Jmax=Jmax)

    # laboratory-frame dipole moment operator
    dip = LabTensor(ocs.dip, sol)

    # laboratory-frame polarizability tensor
    pol = LabTensor(ocs.pol, sol)

    # field-free Hamiltonian
    h0 = LabTensor(ocs, sol)

    # matrix elements of cos^2(theta)-1/3
    cos2 = LabTensor("cos2theta", sol) 


    # Time-dependent simulations


    # set up 800nm Gaussian pulse
    def field(t, FWHM):
        nm = 1e-9
        omega = 800 * nm # in nm
        omega = 2 * np.pi * speed_of_light / omega * 1e-12  # in 1/ps
        t0 = 2.5 * FWHM / 2
        amp = 1e10 # in V/m
        return [0, 0, amp * np.exp(-4*np.log(2)*(t-t0)**2/FWHM**2) * np.cos(omega*t)]

    # set up interaction Hamiltonian
    # `AUpol_x_Vm_to_invcm` converts polarizability[au] * field[V/m] into [cm^-1]
    H = -1/2 * pol * AUpol_x_Vm_to_invcm()

    # initial rotational temperature in Kelvin
    temp = 0

    # matrix elements of cos^2(theta) in sparse matrix form
    cos2mat = cos2.tomat(form="full", cart="0")

    cos2_expval = {}

    for FWHM in [0.5, 1, 10, 20, 50, 100]:

        print(f"run simulations for FWHM {FWHM}")

        # set up TDSE parameters, initial time `t_start`, terminal time `t_end`,
        # time step `dt`, time units `t_units`, energy units `enr_units`
        tdse = TDSE(t_start=0, t_end=300, dt=0.01, t_units="ps", enr_units="invcm")

        # initial states - Boltzmann-weighted eigenfunctions of `h0`
        # at T=0 K - only ground state
        vecs = tdse.init_state(h0, temp=temp)
        print(f"number of initial state vectors: {len(vecs)}")

        cos2_expval[FWHM] = np.zeros(len(tdse.time_grid()), dtype=np.complex128)

        for i, t in enumerate(tdse.time_grid()):

            # apply field to Hamiltonian
            thresh = 1e3 # thresh for considering field as zero
            H.field(field(t, FWHM), thresh=thresh)

            # update vector
            vecs, t_ = tdse.update(H, H0=h0, vecs=vecs, matvec_lib='scipy')

            # expectation value of cos^2(theta)
            expval = sum( np.dot( np.conj(vecs[i][:]), cos2mat.dot(vecs[i][:]) )
                          for i in range(len(vecs)) ) + 1/3

            cos2_expval[FWHM][i] = expval.real

            if i % 1000 == 0:
                print(f"time {t} ps")

        plt.plot(tdse.time_grid(), cos2_expval[FWHM], linewidth=2, label=f"FWHM {FWHM}")
        plt.title(f"alignment of OCS at T = {temp} K, Gaussian pulse, 800 nm, E$_0$ = 1e10 V/m")
        plt.xlabel("time in ps")
        plt.ylabel("$\cos^2\\theta$")
        plt.legend()
        plt.draw()
        plt.pause(0.0001)

    plt.savefig(f"ocs_alignment_T{temp}.png")
