<div align="center">
  <img src="https://github.com/CFEL-CMI/richmol/blob/develop/doc/source/_static/richmol_logo.jpg" height="100px"/>
</div>

# Python-based Simulations of Rovibrational Molecular Dynamics

Richmol was started by theory team members of the Controlled Molecule Imaging group (https://www.controlled-molecule-imaging.org/), Center for Free-Electron Laser Science at Deutsches Elektronen-Synchrotron. We hope to attract a larger community of develoers and researchers working in the field of molecular nuuclear motion dynamics, spectroscopy, and coherent control.

Richmol is intended for computing the rotational and ro-vibrational energy levels, spectra, and field-induced time-dependent dynamics of molecules. It can be interfaced with other similar computer programs, such  as, for exmple, [TROVE](https://github.com/Trovemaster/TROVE) and [Duo](https://github.com/Trovemaster/Duo). We welcome any feedback, feature requests, issues, questions or concerns, please report them in our [discussion forum](https://github.com/CFEL-CMI/richmol/discussions)

## Overview

Richmol is library for simulating the molecular nuclear motion dynamics and related properties.
It includes:
* **Rotational energy levels and spectra** (`richmol.rot`, `richmol.spectrum`): Watson Hamiltonians in *A* and *S* reduction forms, user-built custom effective rotational Hamiltonians, electric dipole, magentic dipole, electric quadrupole, and Raman spectra
* **Rotational field-induced dynamics** (`richmol.field`, `richmol.tdse`): simulations of rotational dynamics and related properties in arbitrary static and time-dependent fields
* **Ro-vibrational dynamics and spectra** (via interface with [TROVE](https://github.com/Trovemaster/TROVE)): simulations of spectra and ro-vibrational dynamics in static and time-dependent fields
* **Non-adiabatic dynamics of diatomic molecules** (via interface with [Duo](https://github.com/Trovemaster/Duo)): field-induced dynamics including non-adiabatic and spin-orbit coupling effects

Coming releases will include:
* **Hyperfine effects** (`richmol.hype`): spectra and dynamics on hyperfine states, obtained from nuclear quadrupole, spin-spin, and spin-rotation interactions
* **VMI observables** (`richmol.vmi`): time-evolutions of 2D projections of probability density functions for selected molecular groups (in axial-recoil approximation)


## Quick install

```
> pip install --upgrade pip
> pip install --upgrade richmol
```

Latest version

```
> pip install --upgrade git+https://github.com/CFEL-CMI/richmol.git
```

## Quick start

### Molecular field-free rotational solutions

Compute rotational energies and matrix elements of dipole moment and polarizability for water molecule using data obtained from a quantum-chemical calculation

```py
from richmol.rot import Molecule, solve, LabTensor
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

# rotational solutions for J=0..5
Jmax = 5
sol = solve(water, Jmin=0, Jmax=Jmax)

# laboratory-frame dipole moment operator
dip = LabTensor(water.dip, sol, thresh=1e-12) # neglect matrix elements smaller than `thresh`

# laboratory-frame polarizability tensor
pol = LabTensor(water.pol, sol, thresh=1e-12)

# field-free Hamiltonian
h0 = LabTensor(water, sol, thresh=1e-12)
```

Print out rotational energies and assignments

```py
print("J  sym #    energy      J   k  tau  |leading coef|^2")
for J in sol.keys():
    for sym in sol[J].keys():
        for i in range(sol[J][sym].nstates):
            print(J, "%4s"%sym, i, "%12.6f"%sol[J][sym].enr[i], sol[J][sym].assign[i])

# prints out
# J  sym #    energy      J   k  tau  |leading coef|^2
# 0    A 0     0.000000 ['0' '0' '0' ' 1.000000']
# 1   B1 0    41.996372 ['1' '0' '1' ' 1.000000']
# 1   B2 0    36.931654 ['1' '1' '0' ' 1.000000']
# ...
```

Print out matrix elements of dipole and polarizability

```py
# X-component of dipole moment
mu_x = dip.tomat(form="full", cart="x")

# XZ-component of polarizability
alpha_xz = pol.tomat(form="full", cart="xz")

# field-free Hamiltonian
h0mat = h0.tomat(form="full", cart="0")

print("matrix dimensions:", h0mat.shape)
print("dipole X:", mu_x)
print("\npolarizability XZ:", alpha_xz)
print("\nfield-free Hamiltonian:", h0mat)
```

Print out the assignment of field-free states

```py
assign_bra, assign_ket = h0.assign(form="full") # assignment of braand ket states, i.e. `assign_bra` and `assign_ket` are equivalent in this case
assign = assign_bra

# print assignment of first 20 states
for i in range(20):
    print(i, "J =", assign["J"][i], ", sym =", assign["sym"][i], ", m =", assign["m"][i], ", k =", assign["k"][i])

# prints out
# 0 J = 0.0 , sym = A , m = 0 , k = ('0 0 0  1.000000', 0.0)
# 1 J = 1.0 , sym = B1 , m = -1 , k = ('1 0 1  1.000000', 41.996371682354464)
# 2 J = 1.0 , sym = B1 , m = 0 , k = ('1 0 1  1.000000', 41.996371682354464)
# ...
```

### Storing and reading matrix elements from HDF5 files

The calculation of field-free energies and matrix elements of interaction tensors can be computationally expensive, especially if one considers the vibrational motions as well. Once computed, the rotational solutions and matrix elements can be stored in an HDF5 format file. The HDF5 files containing rotational, ro-vibrational, and even hyperfine solutions and matrix elements for different molecules can be produced by other programs, such as, for example, [TROVE](https://github.com/Trovemaster/TROVE). A collection of such files for different molecules is available through "Richmol database" section of the main documentation.

Here is how you store matrix elements

```py
dip.store("water.h5", replace=True, comment="dipole moment in au computed using CCSD(T)/AVTZ")
pol.store("water.h5", replace=True, comment="polarizability in au computed using CCSD(T)/AVTZ")
h0.store("water.h5", replace=True, comment="rot solutions from CCSD(T)/AVTZ equilibrium geometry")
```

and how you read them back

```py
from richmol.field import CarTens

dip2 = CarTens(filename="water.h5", name="dip")
print("dipole:", dip2.__doc__)

pol2 = CarTens(filename="water.h5", name="pol")
print("polarizability:", pol2.__doc__)

h02 = CarTens(filename="water.h5", name="h0")
print("field-free H0:", h02.__doc__)

# prints out
# dipole: Cartesian tensor operator, store date: 2021-06-09 14:32:36, comment: dipole moment in au computed using CCSD(T)/AVTZ
# polarizability: Cartesian tensor operator, store date: 2021-06-09 14:32:36, comment: polarizability in au computed using CCSD(T)/AVTZ
# field-free H0: Cartesian tensor operator, store date: 2021-06-09 14:32:37, comment: rot solutions from CCSD(T)/AVTZ equilibrium geometry
```
### Static field simulations

Once the field-free solutions and matrix elements are obtained, the simulations of the field dynamics are straightforward.
Here is an example of simulation of Stark effect for water molecule 

```py
import numpy as np
from richmol.convert_units import AUdip_x_Vm_to_invcm

enr = []
muz = []

fz_grid = np.linspace(1, 1000*100, 10) # field in units V/m

muz0 = dip.tomat(form="full", cart="z") # matrix representation of Z-dipole at zero field
print(f"matrix dimensions:", muz0.shape)

for fz in fz_grid:

    field = [0, 0, fz] # X, Y, Z field components

    # Hamiltonian
    h = h0 - dip * field * AUdip_x_Vm_to_invcm() # `AUdip_x_Vm_to_invcm` converts dipole(au) * field(V/m) into cm^-1

    # eigenproblem solution
    e, v = np.linalg.eigh(h.tomat(form='full', repres='dense'))

    # keep field-dressed energies
    enr.append([elem for elem in e])

    # keep field-dressed matrix elements of Z-dipole
    muz.append( np.dot(np.conj(v.T), muz0.dot(v)) )

# prints out
# matrix dimensions: (286, 286)
```

Plot the results for selected state index

```py
import matplotlib.pyplot as plt

# plot energies and dipoles vs field

enr = np.array(enr)
muz = np.array(muz)

istate = 0 # choose state index

fig, (ax1, ax2) = plt.subplots(1,2, constrained_layout=True)
plt.suptitle(f"state #{istate}")
ax1.set_ylabel("energy in cm$^{-1}$")
ax1.set_xlabel("field in V/m")
ax2.set_ylabel("$\\mu_Z$ in au")
ax2.set_xlabel("field in V/m")
ax1.plot(fz_grid, enr[:, istate])
ax2.plot(fz_grid, muz[:, istate, istate].real)
plt.show()
```

## Citing richmol

To cite this repository

```
@article{richmol2021github,
  author = {Cem Saribal, Guang Yahg, Emil Zak, A. Yachmenev, J. Kuepper},
  title = {{R}ichmol: {P}ython package for variational simulations of molecular nuclear motion dynamics in fields},
  journal = {Comput. Phys. Commun.},
  year = {2021},
  volume = {xx},
  pages = {xx},
  note = {Current version is available from \href{https://github.com/CFEL-CMI/richmol}{GitHub}},
}
```

