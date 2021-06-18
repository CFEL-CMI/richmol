"""An example of calculation of Stark effect for water using rigid-rotor
approximation

Authors: @yachmena
"""

from richmol.rot import Molecule, solve, LabTensor
import numpy as np
from richmol.convert_units import AUdip_x_Vm_to_invcm, AUpol_x_Vm_to_invcm
import matplotlib.pyplot as plt


# Compute field-free rotational states and matrix elements
# using the data obtained from a quantum chemical calculation


print("compute field-free rotational states")

water = Molecule()

# Cartesian coordinates of atoms
water.XYZ = ("bohr",
             "O",  0.00000000,   0.00000000,   0.12395915,
             "H",  0.00000000,  -1.43102686,  -0.98366080,
             "H",  0.00000000,   1.43102686,  -0.98366080)

# choose frame of principal axes of inertia
water.frame = "ipas"

# molecular-frame dipole moment (au)
water.dip = [0, 0, -0.7288]

# molecular-frame polarizability tensor (au)
water.pol = [[9.1369, 0, 0], [0, 9.8701, 0], [0, 0, 9.4486]]

# symmetry group
water.sym = "D2"

# rotational solutions for J=0..10
Jmax = 10
sol = solve(water, Jmin=0, Jmax=Jmax)

# laboratory-frame dipole moment operator
dip = LabTensor(water.dip, sol, thresh=1e-12) # neglect matrix elements smaller than `thresh`

# laboratory-frame polarizability tensor
pol = LabTensor(water.pol, sol, thresh=1e-12)

# field-free Hamiltonian
h0 = LabTensor(water, sol, thresh=1e-12)

# print rotational energies

print("J  sym #    energy      J   k  tau  |leading coef|^2")
for J in sol.keys():
    for sym in sol[J].keys():
        for i in range(sol[J][sym].nstates):
            print(J, "%4s"%sym, i, "%12.6f"%sol[J][sym].enr[i], sol[J][sym].assign[i])


# Compute Stark effect


print("compute Stark effect")

enr = []
muz = []

fz_grid = np.linspace(1, 1000000*100, 10) # field in units V/m (max field is 1 MV/cm)

muz0 = dip.tomat(form="full", cart="z") # matrix representation of Z-dipole at zero field
print(f"matrix dimensions:", muz0.shape)

# set up molecule-field interaction Hamiltonians
#   permanent dipole
Hdc = -1 * dip * AUdip_x_Vm_to_invcm() # `AUdip_x_Vm_to_invcm` converts dipole(au) * field(V/m) into cm^-1
#   polarizability
Hdc2 = -1 * pol * AUpol_x_Vm_to_invcm() # `AUpol_x_Vm_to_invcm` converts polarizability(au) * field(V/m)**2 into cm^-1

for fz in fz_grid:

    field = [0, 0, fz] # X, Y, Z field components

    print(f"run for field {field}")

    # Hamiltonian
    h = h0 + Hdc * field + Hdc2 * field

    # eigenproblem solution
    e, v = np.linalg.eigh(h.tomat(form='full', repres='dense'))

    # keep field-dressed energies
    enr.append([elem for elem in e])

    # keep field-dressed matrix elements of Z-dipole
    muz.append( np.dot(np.conj(v.T), muz0.dot(v)) )


# plot energies and dipoles vs field


enr = np.array(enr)
muz = np.array(muz)

fig, (ax1, ax2) = plt.subplots(1,2, constrained_layout=True)
ax1.set_ylabel("energy in cm$^{-1}$")
ax1.set_xlabel("field in kV/cm")
ax2.set_ylabel("$\\mu_Z$ in au")
ax2.set_xlabel("field in kV/cm")
for istate in range(enr.shape[1]):
    ax1.plot(fz_grid/1e5, enr[:, istate])
    ax2.plot(fz_grid/1e5, muz[:, istate, istate].real)
plt.show()

