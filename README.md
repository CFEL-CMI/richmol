<div align="center">
  <img src="https://github.com/CFEL-CMI/richmol/blob/develop/doc/source/_static/richmol_logo.jpg" height="70px"/>
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

Compute rotational energies and matrix elements of dipole moment and polarizability for water molecule using data obtained from a quantum-chemical calculation

```python
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

