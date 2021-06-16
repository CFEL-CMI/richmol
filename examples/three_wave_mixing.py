"""Example of a three-wave mixing simulation for chiral molecule 1,2-propanediol

Details are described in original paper https://doi.org/10.1038/nature12150

Authors: @yachmena
"""

from richmol.rot import Molecule, LabTensor, solve
from richmol.convert_units import MHz_to_invcm, Debye_x_Vm_to_MHz
from richmol.tdse import TDSE
import numpy as np
import matplotlib.pyplot as plt


def threeWM(molecule, Jmax=1):
    """Simulates results of three-wave mixing field configuration applied
    to a molecule `molecule`, considering all rotational states with J <= `Jmax`.
    The field configuration used is described in https://doi.org/10.1038/nature12150

    Returns:
        time_grid : array
            Time grid
        field : array (len(time_grid), 3)
            X, Y, Z components of electric field on `time_grid` time grid
        expval : dict
            Expectation values of lab-frame dipole moment on `time_grid` time grid
    """
    # obtain rotational solutions
    sol = solve(molecule, Jmax=Jmax)

    # print energy levels and assignments
    print("\nRotational solutions")
    print("J  sym #    energy      J   k  tau  |leading coef|^2")
    for J in sol.keys():
        for sym in sol[J].keys():
            sol[J][sym].assign_nprim = 2 # print up to two leading contributions (default is one)
            for i in range(sol[J][sym].nstates):
                print(J, "%4s"%sym, i, "%12.6f"%sol[J][sym].enr[i], sol[J][sym].assign[i])

    # matrix elements of lab-frame dipole operator
    dip = LabTensor(molecule.dip, sol, thresh=1e-8)

    # field-free Hamiltonian
    h0 = LabTensor(molecule, sol, thresh=1e-8)

    # Time-dependent simulations

    # oscillating Z field tuned to produce pi/2 transition from the ground
    # (J,k,tau) = (0,0,0) state to excited (1,0,1) state at 12212.15 MHz
    def fz(t):
        t0 = 300 # ns
        FWHM = 2 * t0 / 2.5
        omega = 12212.0 # MHz
        omega = 2 * np.pi * omega * 1e-3 # 1/ns
        amp = 1250  # V/m
        return amp * np.exp(-4*np.log(2)*(t-t0)**2/FWHM**2) * np.cos(omega*t)

    # static X field with a 400 ns switch-off time after 800 ns
    def fx(t):
       t0 = 1000
       return 65 * 100 / (1 + np.exp(1.5e-2*(t-t0))) # 65 V/cm

    # TDSE parameters, initial time `t_start`, terminal time `t_end`,
    # time step `dt`, time units `t_units`, energy units `enr_units`
    tdse = TDSE(t_start=0, t_end=1000, dt=0.01, t_units="ns", enr_units="MHz")

    # interaction Hamiltonian
    H = -1 * dip * Debye_x_Vm_to_MHz() # `Debye_x_Vm_to_MHz()` converts dipole(Debye) * field(V/m) into energy(MHz)

    # choose X dc field dressed states as initial, at T = 0 K
    temp = 0
    H = H * [fx(0), 0, 0]
    vecs = tdse.init_state(h0 + H, temp=temp)
    print(f"\nNumber of initial state vectors: {len(vecs)} (T = {temp})")

    # print state assignments in the full basis (including m-degeneracy)
    # this might be needed to identify absolute basis state index
    # for a state of interest
    assign, _ = h0.assign(form="full")
    print("\nAssignment of states in the full basis")
    for i in range(len(assign['J'])):
        print(f"ind = {i}, J = {assign['J'][i]}, m = {assign['m'][i]}, k = {assign['k'][i]}")

    # dipole matrix elements in sparse matrix form
    dipmat = [dip.tomat(form="full", cart=cart) for cart in ("x", "y", "z")]

    # dict to store temporal expectation values of dipole moment projections
    expval = {"x": [], "y": [], "z": []}

    print("\nStart time propagation")
    for i, t in enumerate(tdse.time_grid()):

        H.field([fx(t), 0, fz(t)], thresh=1e0)

        # update vector
        vecs, t_ = tdse.update(H, H0=h0, vecs=vecs, matvec_lib='scipy')

        # expectation value of dipole moment
        for mat, cart in zip(dipmat, ("x", "y", "z")):
            expv = sum( np.dot( np.conj(vecs[i][:]), mat.dot(vecs[i][:]) )
                         for i in range(len(vecs)) )
            expval[cart].append(expv)

        if i % 1000 == 0:
            print(f"time {round(t_, 4)} ns, [mu_x, mu_y, mu_z] = {[round(expval[cart][-1].real, 6) for cart in expval.keys()]} ")

    field = np.array([[fx(t), 0, fz(t)] for t in tdse.time_grid()])

    return tdse.time_grid(), field, expval


if __name__ == "__main__":

    # R-enantiomer
    Rpropanediol = Molecule()

    # rotational constants in MHz
    Rpropanediol.B = (8572.05, 3640.10, 2790.96)

    # molecular-frame dipole moment in Debye
    Rpropanediol.dip = [1.2, 1.9, 0.36]

    # run three-wave mixing
    times, field, Rdip = threeWM(Rpropanediol, Jmax=1)

    # S-enantiomer
    Spropanediol = Molecule()

    # rotational constants in MHz
    Spropanediol.B = (8572.05, 3640.10, 2790.96)

    # molecular-frame dipole moment in Debye (can change sign of any single component between R and S)
    Spropanediol.dip = [1.2, 1.9, -0.36]

    # run three-wave mixing
    times, field, Sdip = threeWM(Spropanediol, Jmax=1)

    # plot results
    plt.title(f"Three wave mixing, 1,2-propanediol")
    plt.plot(times, field[:,0]/np.max(field[:,0]), label=f"field X / {round(np.max(field[:,0])*1e-2, 1)} V/cm ")
    plt.plot(times, field[:,1]/np.max(field[:,1]), label=f"field Y / {round(np.max(field[:,1])*1e-2, 1)} V/cm ")
    plt.plot(times, field[:,2]/np.max(field[:,2]), label=f"field Z / {round(np.max(field[:,2])*1e-2, 1)} V/cm ")
    # plt.plot(times, Rdip["x"], label="$\\mu_X$, R-enantiomer")
    plt.plot(times, Rdip["y"], label="$\\mu_Y$, R-enantiomer")
    # plt.plot(times, Rdip["z"], label="$\\mu_Z$, R-enantiomer")
    # plt.plot(times, Sdip["x"], label="$\\mu_X$, S-enantiomer")
    plt.plot(times, Sdip["y"], label="$\\mu_Y$, S-enantiomer")
    # plt.plot(times, Sdip["z"], label="$\\mu_Z$, S-enantiomer")
    plt.xlabel("time in ns")
    plt.legend()
    plt.show()
