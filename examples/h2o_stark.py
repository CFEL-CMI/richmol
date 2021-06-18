"""An example of calculation of Stark effect for water using accurate
rovibrational data computed by means of the variational approach TROVE

Authors: @yachmena
"""

import urllib.request
import os
import h5py
from richmol.field import CarTens
import numpy as np
from richmol.convert_units import Debye_x_Vm_to_invcm
import matplotlib.pyplot as plt

# Get richmol file from a repository

richmol_file = "h2o_p48_j40_rovib.h5"
if not os.path.exists(richmol_file):
    url = "https://zenodo.org/record/4943807/files/h2o_p48_j40_rovib.h5"
    print(f"download richmol file from {url}")
    urllib.request.urlretrieve(url, "h2o_p48_j40_rovib.h5")
    print("download complete")

# Print available datasets and metadata

fl = h5py.File(richmol_file, "r")
print(f"Available datasets in file '{richmol_file}'")
for key, val in fl.items():
    print(f"'{key}'")                   # dataset name
    print("\t", val.attrs["__doc__"])   # dataset description


# Calculate Stark shifts

# Since the dataset in `richmol_file` contains about 20K field-free rovibrational
# states (not accounting for the m-degeneracy), it is useful to define the so-called
# filter functions used to select a subspace of desired states based on the state
# assignment. Each state is assigned by J and m rotational quantum numbers,
# state symmetry and energy, as well as rovibrational quantum numbers, such as
# k, tau (parity), and vibrational quanta. For each new dataset, the format of
# the rovibrational quanta can be different, please have a look at the dataset
# description to familiarize with the format.

# Below we define two filter functions, selecting (A1, A2) and (B1, B2) state
# symmetries, respectively, and both selecting pure rotational states with J <= 10
# Note: for C2v symmetry molecules, dipole can't couple A1 and A2 with B1 and B2
# symmetry states.

def filter_A1_A2(**kw):
    """Filter to choose states with vibrational quanta v1, v2, v3 = 0, 0, 0
    rotational J <= 10, and total symmetry A1 or A2 (coupled by dipole)
    """
    pass_j = True
    pass_k = True
    pass_sym = True
    if 'k' in kw:
        k = kw['k'] # rovibrational assignment format 'k tau v1 v2 v3 rotsym vibsym repl', see h0.__doc__
        pass_k = all(float(elem) == 0 for elem in k.split()[2:5])
    if 'J' in kw:
        J = kw['J']
        pass_j = J <= 10.0
    if 'sym' in kw:
        sym = kw['sym']
        pass_sym = sym in ("A1", "A2")
    return pass_j * pass_k * pass_sym


def filter_B1_B2(**kw):
    """Filter to choose states with vibrational quanta v1, v2, v3 = 0, 0, 0
    rotational J <= 10, and total symmetry B1 or B2 (coupled by dipole)
    """
    pass_j = True
    pass_k = True
    pass_sym = True
    if 'k' in kw:
        k = kw['k'] # rovibrational assignment 'k tau v1 v2 v3 rotsym vibsym repl', see h0.__doc__
        pass_k = all(float(elem) == 0 for elem in k.split()[2:5])
    if 'J' in kw:
        J = kw['J']
        pass_j = J <= 10.0
    if 'sym' in kw:
        sym = kw['sym']
        pass_sym = sym in ("B1", "B2")
    return pass_j * pass_k * pass_sym

enr = {}
muz = {}

for sym, filter in zip(("A1/A2", "B1/B2"), (filter_A1_A2, filter_B1_B2)):

    # load field-free Hamiltonian and dipole matrix elements
    h0 = CarTens(richmol_file, name="h0", bra=filter, ket=filter)
    dip = CarTens(richmol_file, name="dipole", thresh=1e-6, bra=filter, ket=filter)

    enr[sym] = []
    muz[sym] = []

    fz_grid = np.linspace(1, 100000*100, 10) # field in units V/m (max 100 kV/cm)

    muz0 = dip.tomat(form="full", cart="z") # matrix representation of Z-dipole at zero field

    print(f"symmetry: '{sym}', matrix dimensions: {muz0.shape}")

    for fz in fz_grid:

        field = [0, 0, fz] # X, Y, Z field components

        print(f"run field {field}")

        # Hamiltonian
        h = h0 - dip * field * Debye_x_Vm_to_invcm() # `Debye_x_Vm_to_invcm` converts dipole(Debye) * field(V/m) into cm^-1

        # eigenproblem solution
        e, v = np.linalg.eigh(h.tomat(form='full', repres='dense'))

        # keep field-dressed energies
        enr[sym].append([elem for elem in e])

        # keep field-dressed matrix elements of Z-dipole
        muz[sym].append( np.dot(np.conj(v.T), muz0.dot(v)) )

    # plot energies and dipoles vs field

    enr[sym] = np.array(enr[sym])
    muz[sym] = np.array(muz[sym])


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, constrained_layout=True)
for sym, (a1, a2) in zip(enr.keys(), ((ax1, ax2), (ax3, ax4))):
    a1.set_title(f"{sym}")
    a1.set_ylabel("energy in cm$^{-1}$")
    a1.set_xlabel("field in kV/cm")
    a2.set_title(f"{sym}")
    a2.set_ylabel("$\\mu_Z$ in Debye")
    a2.set_xlabel("field in kV/cm")
    for istate in range(enr[sym].shape[1]):
        a1.plot(fz_grid/100000, enr[sym][:, istate])
        a2.plot(fz_grid/100000, muz[sym][:, istate, istate].real)
plt.show()

