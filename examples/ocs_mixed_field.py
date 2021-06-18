"""Example of calculation of mixed-field orientation of linear OCS molecule
by a combination of linearly-polarized ac and dc fields

The mixed-field orientation of OCS is described in https://doi.org/10.1002/cphc.201600710

Author: @yachmena
"""
from richmol.rot import Molecule, solve, LabTensor
from richmol.tdse import TDSE
from richmol.convert_units import AUpol_x_Vm_to_invcm, AUdip_x_Vm_to_invcm
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

    # matrix elements of cos(theta)
    cos = LabTensor("costheta", sol) 

    # matrix elements of cos^2(theta)-1/3
    cos2 = LabTensor("cos2theta", sol) 


    # Time-dependent simulations


    # 800nm linearly polarized along Z Gaussian pulse
    # with 700 ps rise and shorter 250 ps fall
    def field(t):
        nm = 1e-9
        omega = 800 * nm # in nm
        omega = 2 * np.pi * speed_of_light / omega * 1e-12  # in 1/ps
        t0 = 700.0 # ps
        FWHM = 2 * 600.0 / 2.5
        FWHM2 = 2 * 250.0 / 2.5
        amp = 1.5e7 * 100 # field amplitude in V/m
        if t <= t0:
            return [0, 0, amp * np.exp(-4*np.log(2)*(t-t0)**2/FWHM**2) * np.cos(omega*t)]
        else:
            return [0, 0, amp * np.exp(-4*np.log(2)*(t-t0)**2/FWHM2**2) * np.cos(omega*t)]

    # static electric field 20.7 kV/cm rotated by an angle `beta` from X in the XZ plane
    dc = 20.7 * 1000 * 100 # in V/m
    beta = 35.0 * np.pi / 180.0
    rot = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    dc_field = np.dot(rot, [dc, 0, 0])

    # TDSE parameters, initial time `t_start`, terminal time `t_end`,
    # time step `dt`, time units `t_units`, energy units `enr_units`
    tdse = TDSE(t_start=0, t_end=1000, dt=0.01, t_units="ps", enr_units="invcm")

    # matrix elements of cos(theta) and cos^2(theta) in sparse matrix format
    cosmat = cos.tomat(form="full", cart="0")
    cos2mat = cos2.tomat(form="full", cart="0")

    # molecule-dc-field interaction potential
    Hdc = -1 * dip * dc_field * AUdip_x_Vm_to_invcm() # `AUdip_x_Vm_to_invcm()` converts dipole(au) * field(V/m) into energy(cm^-1)

    # choose dc-field dressed states at hypothetical T = 0 K as initial states
    temp = 0
    vecs = tdse.init_state(h0 + Hdc, temp=temp)
    print(f"number of initial state vectors: {len(vecs)}")

    # molecule-laser interaction part (without field)
    Hac = -0.5 * pol * AUpol_x_Vm_to_invcm() # `AUpol_x_Vm_to_invcm()` converts polarizability(au) * field(V/m)**2 into energy(cm^-1)

    # lists to store temporal expectation values of cos^2(theta) and cos(theta)
    cos2_expval = []
    cos_expval = []

    # time-evolution loop
    for i, t in enumerate(tdse.time_grid()):

        # apply field to molecule-laser interaction part
        thresh = 1e1 # thresh for considering field as zero
        Hac.field(field(t), thresh=thresh)

        # update vector
        vecs, t_ = tdse.update(Hdc + Hac, H0=h0, vecs=vecs, matvec_lib='scipy')

        # expectation value of cos(theta)
        expval = sum( np.dot( np.conj(vecs[i][:]), cosmat.dot(vecs[i][:]) )
                      for i in range(len(vecs)) )

        # expectation value of cos^2(theta)
        expval2 = sum( np.dot( np.conj(vecs[i][:]), cos2mat.dot(vecs[i][:]) )
                      for i in range(len(vecs)) ) + 1/3

        cos2_expval.append(expval2)
        cos_expval.append(expval)

        if i % 1000 == 0:
            print(f"time {t_} ps")

    # plot results
    f = np.array([field(t)[2] for t in tdse.time_grid()])
    plt.plot(tdse.time_grid(), f/np.max(f), 'gray', linewidth=2, label=f"pulse / {round(np.max(f)/1e2, 1)} V/cm")
    plt.plot(tdse.time_grid(), cos2_expval, linewidth=2, label=f"$\cos^2\\theta$")
    plt.plot(tdse.time_grid(), cos_expval, linewidth=2, label=f"$\cos\\theta$")
    plt.title(f"Mixed-field orientation of OCS at T = {temp} K, dc = {dc/1e5} kV/cm, $\\beta$ = {beta*180.0/np.pi} deg")
    plt.xlabel("time in ps")
    plt.legend()
    plt.draw()
    plt.pause(0.0001)

    plt.savefig(f"ocs_mixed_field_dc{dc/1e5}_beta{beta*180.0/np.pi}_T{temp}.png")
