<div align="center">
  <img src="https://github.com/CFEL-CMI/richmol/blob/develop/docs/source/_static/richmol_logo.jpg" height="80px"/>
</div>

# Python-based Simulations of Rovibrational Molecular Dynamics

[![Documentation Status](https://readthedocs.org/projects/richmol/badge/?version=latest)](https://richmol.readthedocs.io/en/latest/?badge=latest)


[**Overview**](#overview)
| [**Quick install**](#quick-install)
| [**Quick Start**](#quick-start)
| [**Examples**](examples/)
| [**Citing richmol**](#citing-richmol)
| [**Documentation**](https://richmol.readthedocs.io/)


Richmol is currently being developed and maintained by the Theory team of the [CFEL Controlled Molecule Imaging group](https://www.controlled-molecule-imaging.org) at [Deutsches Elektronen-Synchrotron DESY](https://www.desy.de) and [Universität Hamburg](https://www.uni-hamburg.de). We hope to attract collaborators for a joint development by a larger community of researchers working in fields of molecular nuclear-motion dynamics, spectroscopy, and laser alignment and coherent control of molecules.

Richmol is intended for computing molecular rotational and ro-vibrational energy levels, spectra, and field-induced dynamics. It can be interfaced with other similar variational codes, such  as, for example, [TROVE](https://github.com/Trovemaster/TROVE) and [Duo](https://github.com/Trovemaster/Duo). We welcome any feedback, feature requests, issues, questions or concerns, please report them in our [discussion forum](https://github.com/CFEL-CMI/richmol/discussions) or the [issue tracker](https://github.com/CFEL-CMI/richmol/issues).

## Overview

Richmol is a library for simulating molecular nuclear motion dynamics and related properties.
It is based on Python, some of its parts, for example, time-dependent propagation, can run on GPUs using `numba` and `cupy` libraries.
Richmol can be used for predicting:
* **Rotational energy levels and spectra** (`richmol.rot`, `richmol.spectrum`): Watson Hamiltonians in *A* and *S* reduction forms, user-built custom effective rotational Hamiltonians, electric dipole, magnetic dipole, electric quadrupole, and Raman spectra
* **Rotational field-induced dynamics** (`richmol.field`, `richmol.tdse`): simulations of rotational dynamics and related properties of molecules placed in static and time-dependent fields
* **Ro-vibrational dynamics and spectra** (currently via interface with [TROVE](https://github.com/Trovemaster/TROVE)): simulations of spectra and ro-vibrational dynamics of molecules in static and time-dependent fields
* **Non-adiabatic dynamics of diatomic molecules** (via interface with [Duo](https://github.com/Trovemaster/Duo)): field-induced dynamics of diatomic molecules including non-adiabatic and spin-orbit coupling effects

We plan to include following new features in the next release:
* **Hyperfine effects** (`richmol.hype`): spectra and dynamics on hyperfine states, including nuclear quadrupole, spin-spin, and spin-rotation interactions
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

Some of modules (e.g. `spectrum`) require MPI libraries to be installed on your system, you can install them using

```
> apt install libopenmpi-dev
> pip install mpi4py 
```
alternatively
```
> brew install mpich
> conda install mpi4py 
```


## Quick start

In Richmol, calculations of rotational (and generally ro-vibrational) dynamics proceeds in two steps.
At the first step, molecular energy levels and matrix elements of molecule-field interation tensors
(multipoles) are computed.
These are then used in the second step for calculation of molecular dynamics in static or time-dependent
external electric and/or magnetic fields.

The calculations of field-free molecular rotational or ro-vibrational energy levels and wave functions can be extremely
tedious and can, in principle, be done using other variational codes.
One of such programs is [TROVE](https://github.com/Trovemaster/TROVE), that can be used to generate
accurate ro-vibrational energies, wave functions, and matrix elements of various molecule-field interaction tensors
for small and medium size molecules. These can be stored in an HDF5 file and later
used by Richmol for simulations of molecular spectra or dynamics.
A collection of such HDF5 data files for different molecules is available through
[Richmol molecular database](https://richmol.readthedocs.io/en/latest/richmol_database.html) section of the main documentation.

Below, we show few simple examples of simulations using pure rotational states.
For more examples and tutorials, please see our [examples](examples/) folder and main [Documentation](https://richmol.readthedocs.io/)

### Molecular field-free rotational solutions

Compute rotational energies and matrix elements of dipole moment and polarizability
for water molecule using data obtained from a quantum-chemical calculation

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

### Storing and reading matrix elements from HDF5 files

Once computed, rotational solutions and matrix elements can be stored in an HDF5 file.
A collection of pre-computed HDF5 files containing rotational, ro-vibrational,
and even hyperfine solutions and matrix elements for different molecules are available
through [Richmol molecular database](https://richmol.readthedocs.io/en/latest/richmol_database.html) section of the main [Documentation](https://richmol.readthedocs.io/).
Here is how the matrix elements can be stored in HDF5 file

```py
dip.store("water.h5", replace=True, comment="dipole moment in au computed using CCSD(T)/AVTZ")
pol.store("water.h5", replace=True, comment="polarizability in au computed using CCSD(T)/AVTZ")
h0.store("water.h5", replace=True, comment="rot solutions from CCSD(T)/AVTZ equilibrium geometry")
```

and how they can be read

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

Once the field-free molecular solutions and matrix elements are obtained, the simulations of field-induced dynamics
are straightforward.
Here is an example of the simulation of Stark effect for water molecule 

```py
import numpy as np
from richmol.convert_units import AUdip_x_Vm_to_invcm
import matplotlib.pyplot as plt

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

# plot the results for selected state index

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

<div align="left">
  <img src="https://github.com/CFEL-CMI/richmol/blob/develop/docs/source/_static/readme_water_stark.png" height="300px"/>
</div>

### Time-dependent simulations

Here is an example of the simulation of 'truncated-pulse' alignment for linear OCS molecule.
To begin, we compute the field-free energies, matrix elements of polarizability interaction tensor,
and matrix elements of cos<sup>2</sup>&theta;, that is used to quantify the degree of alignment

```py
from richmol.rot import Molecule, solve, LabTensor
from richmol.tdse import TDSE
from richmol.convert_units import AUpol_x_Vm_to_invcm
from richmol.field import filter
import numpy as np
import matplotlib.pyplot as plt

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

# matrix elements of cos^2(theta)
cos2 = LabTensor("cos2theta", sol) # NOTE: you need to add a constant factor 1/3 to get the true values
```

Now, we define the external electric field. Here, it is loaded from file [trunc_pulse.txt](https://github.com/CFEL-CMI/richmol/tree/develop/docs/source/notebooks/trunc_pulse.txt).
The field in units V/cm has a single *Z* component and is defined on a time grid ranging
from 0 to 300 picoseconds

```py
# truncated-pulse field
with open("trunc_pulse.txt", "r") as fl:
    field = np.array([[float(elem) for elem in line.split()[1:]] for line in fl]) # X, Y, Z field's components
    fl.seek(0)
    times = [float(line.split()[0]) for line in fl] # time grid

# convert field from V/cm to V/m
field *= 1e2

# plot Z component
plt.plot(times, field[:, 2], label="Z component")
plt.xlim([0,70]) # plot first 70 ps
plt.xlabel("time in ps")
plt.ylabel("field in V/m")
plt.legend()
plt.show()
```
<div align="left">
  <img src="https://github.com/CFEL-CMI/richmol/blob/develop/docs/source/_static/readme_trunc_pulse.png" height="300px"/>
</div>

For the initial state distribution we assume a hypothetical temperature of *T* = 0 Kelvin
and use eigenfunctions of the field-free operator `h0` as the initial state vectors.
Run dynamics from time zero to 200 ps with a time step of 10 fs,
plot the expectation values of cos<sup>2</sup>&theta;
and compare them with [reference values](https://github.com/CFEL-CMI/richmol/tree/develop/docs/source/notebooks/trunc_pulse_cos2theta.txt)

```py
tdse = TDSE(t_start=0, t_end=200, dt=0.01, t_units="ps", enr_units="invcm")

# initial states - Boltzmann-weighted eigenfunctions of `h0`, at T=0 K - only ground state
vecs = tdse.init_state(h0, temp=0)

# interaction Hamiltonian
H = -1/2 * pol * AUpol_x_Vm_to_invcm() # `AUpol_x_Vm_to_invcm` converts pol[au]*field[V/m] into [cm^-1]

# matrix elements of cos^2(theta)
cos2mat = cos2.tomat(form="full", cart="0")

cos2_expval = []

for i, t in enumerate(tdse.time_grid()):

    # apply field to Hamiltonian
    H.field([0, 0, field[i, 2]])

    # update vector
    vecs, t_ = tdse.update(H, H0=h0, vecs=vecs, matvec_lib='scipy')

    # expectation value of cos^2(theta)-1/3
    expval = sum(np.dot(np.conj(vecs[i][:]), cos2mat.dot(vecs[i][:])) for i in range(len(vecs)))
    cos2_expval.append(expval)

    if i % 1000 == 0:
        print(t, expval+1/3)

# compare with reference results

plt.plot([t for t in tdse.time_grid()], [elem.real + 1/3 for elem in cos2_expval], 'b', linewidth=4, label="present")

with open("trunc_pulse_cos2theta.txt", "r") as fl:
    cos2_expval_ref = np.array([float(line.split()[1]) for line in fl])
    fl.seek(0)
    times_ref = np.array([float(line.split()[0]) for line in fl])

plt.plot(times_ref, cos2_expval_ref, 'r--', linewidth=2, label="reference")
plt.xlabel("time in ps")
plt.ylabel("$\cos^2\\theta$")
plt.legend()
plt.show()

```

<div align="left">
  <img src="https://github.com/CFEL-CMI/richmol/blob/develop/docs/source/_static/readme_ocs_alignment.png" height="300px"/>
</div>

For more examples and tutorials, please see our [examples](examples/) folder
and [Documentation](https://richmol.readthedocs.io/)

## Citing richmol

To refer to this project in scientic work please cite the following manuscript:

Cem Saribal, Guang Yang, Emil Zak, Yahya Saleh, Jannik Eggers, Vishnu Sanjay, Andrey Yachmenev, and Jochen Küpper: "Richmol: Python package for variational simulations of molecular nuclear motion dynamics in fields", *Comput. Phys. Commun.*, in preparation (2021); the current version of the software is available on [GitHub](https://github.com/CFEL-CMI/richmol)

## Bibtex entry

```
@article{richmol2021github,
  author = {Cem Saribal and Guang Yang and Emil Zak and Yahya Saleh Jannik Eggers and Vishnu Sanjay and Andrey Yachmenev and Jochen Küpper},
  title = {{R}ichmol: {P}ython package for variational simulations of molecular nuclear motion dynamics in fields},
  journal = {Comput. Phys. Commun.},
  year = {2021},
  volume = {in preparation},
  note = {The current version of the software is available on \href{https://github.com/CFEL-CMI/richmol}{GitHub}},
}
```

