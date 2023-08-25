"""
This is an example of a simulation of laser-induced one-dimensional alignment of OCS.
Here, we calculate the expectation values of the alignment functions cos^2(theta)
and cos^2(theta_2D) as well as the rotational density.

To compute the expectation values of alignment functions using the rotational density,
see `dens_sampling.py`.
"""
from richmol.rot import Molecule, solve, LabTensor, cos2theta, cos2theta2d
from richmol.tdse import TDSE
from richmol.convert_units import AUpol_x_Vm_to_invcm
from richmol.rotdens import psi_grid
import numpy as np
import h5py
from scipy.constants import speed_of_light


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
    cos2th = LabTensor("cos2theta", sol) 
    costh = LabTensor("costheta", sol) 

    # matrix elements of cos^2(theta) and cos^2(theta_2D)
    #   using Wigner-expansion approach
    cos2th_wig = LabTensor(cos2theta, sol, thresh=1e-12) 
    cos2th2d_wig = LabTensor(cos2theta2d, sol, thresh=1e-12) 


    # Time-dependent simulations


    # set up 800nm Gaussian pulse
    def field(t, omega_nm):
        nm = 1e-9
        fwhm = 10.0 # ps
        omega = omega_nm * nm # in nm
        omega = 2 * np.pi * speed_of_light / omega * 1e-12  # in 1/ps
        t0 = 2.5 * fwhm / 2
        amp = 1e10 # in V/m
        return [0, 0, amp * np.exp(-4*np.log(2)*(t-t0)**2/fwhm**2) * np.cos(omega*t)]

    # set up interaction Hamiltonian
    # `AUpol_x_Vm_to_invcm` converts polarizability[au] * field[V/m] into [cm^-1]
    H = -1/2 * pol * AUpol_x_Vm_to_invcm()

    # initial rotational temperature in Kelvin
    temp = 0

    # matrix elements of cos^2(theta)-1/3 and cos(theta) in sparse matrix form
    cos2th_mat = cos2th.tomat(form="full", cart="0")
    costh_mat = costh.tomat(form="full", cart="0")
    cos2th_wig_mat = cos2th_wig.tomat(form="full", cart="0")
    cos2th2d_wig_mat = cos2th2d_wig.tomat(form="full", cart="0")

    # set up TDSE parameters, initial time `t_start`, terminal time `t_end`,
    # time step `dt`, time units `t_units`, energy units `enr_units`
    tdse = TDSE(t_start=0, t_end=100, dt=0.01, t_units="ps", enr_units="invcm")

    # initial states - Boltzmann-weighted eigenfunctions of `h0`
    # at T=0 K - only ground state
    vecs = tdse.init_state(h0, temp=temp)
    print(f"number of initial state vectors: {len(vecs)}")

    cos2th_ev = []
    costh_ev = []
    cos2th_wig_ev = []
    cos2th2d_wig_ev = []
    wp_vec = []
    times = []

    for i, t in enumerate(tdse.time_grid()):
    
        # apply field to Hamiltonian
        thresh = 1e3 # thresh for considering field as zero
        H.field(field(t, 800), thresh=thresh)
    
        # update vector
        vecs, t_ = tdse.update(H, H0=h0, vecs=vecs, matvec_lib='scipy')
    
        if i % 100 == 0:
            # expectation value of cos^2(theta)
            expval = sum( np.dot( np.conj(vecs[i][:]), cos2th_mat.dot(vecs[i][:]) )
                          for i in range(len(vecs)) ) + 1/3
            cos2th_ev.append(expval.real)

            # expectation value of cos(theta)
            expval = sum( np.dot( np.conj(vecs[i][:]), costh_mat.dot(vecs[i][:]) )
                          for i in range(len(vecs)) )
            costh_ev.append(expval.real)

            # expectation value of cos(theta) using Wigner expansion
            expval = sum( np.dot( np.conj(vecs[i][:]), cos2th_wig_mat.dot(vecs[i][:]) )
                          for i in range(len(vecs)) )
            cos2th_wig_ev.append(expval.real)

            # expectation value of cos^2(theta_2D) using Wigner expansion
            expval = sum( np.dot( np.conj(vecs[i][:]), cos2th2d_wig_mat.dot(vecs[i][:]) )
                          for i in range(len(vecs)) )
            cos2th2d_wig_ev.append(expval.real)

            # keep vectors
            wp_vec.append(vecs.T)
    
            times.append(t)
            print(f"time {t} ps")

    # compute and store rotational density for selected time steps in `wp_vec`
    #   this density is used later to compute expectation values of alignment
    #   functions, see `dens_sampling.py`

    npt = 100
    alpha = np.linspace(0, 2*np.pi, num=npt, endpoint=True)
    beta = np.linspace(0, np.pi, num=npt, endpoint=True)
    gamma = np.linspace(0, 2*np.pi, num=npt, endpoint=True)

    psi = psi_grid(h0, alpha, beta, gamma, form='full')
    wp_grid = np.dot(psi, np.array(wp_vec))

    dens = np.conj(wp_grid) * wp_grid

    with h5py.File("density.h5", 'w') as fl:
        fl.create_dataset("density", data=np.sum(dens, axis=-1))
        fl.create_dataset("alpha", data=alpha)
        fl.create_dataset("beta", data=beta)
        fl.create_dataset("gamma", data=gamma)
        fl.create_dataset("times", data=np.array(times))
        fl.create_dataset("field", data=np.array([field(t, 1e10) for t in times]))

        # for comparison, also store the alignment functions
        fl.create_dataset("cos2theta", data=np.array(cos2th_ev))
        fl.create_dataset("costheta", data=np.array(costh_ev))
        fl.create_dataset("cos2theta_wig", data=np.array(cos2th_wig_ev))
        fl.create_dataset("cos2theta2d_wig", data=np.array(cos2th2d_wig_ev))

