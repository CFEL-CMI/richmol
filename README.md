<div align="left">
  <img src="https://github.com/CFEL-CMI/richmol/blob/develop/doc/source/_static/richmol_logo.jpg" height="100px"/>
</div>

# Python-based Simulations of Rovibrational Molecular Dynamics

Richmol was started by theory team members of the Controlled Molecule Imaging group (https://www.controlled-molecule-imaging.org/), Center for Free-Electron Laser Science at Deutsches Elektronen-Synchrotron. We hope to attract a larger community of develoers and researchers working in the field of molecular nuuclear motion dynamics, spectroscopy, and coherent control.

Richmol is intended for computing the rotational and ro-vibrational energy levels, spectra, and field-induced time-dependent dynamics of molecules. It can be interfaced with other similar computer programs, such  as, for exmaple, [TROVE](https://github.com/Trovemaster/TROVE) and [Duo](https://github.com/Trovemaster/Duo). We welcome any feedback, feature requests, issues, questions or concerns, please report them in our [discussion forum](https://github.com/CFEL-CMI/richmol/discussions)

## Overview



## Quick install

```
> pip install --upgrade pip
> pip install --upgrade richmol
```

Latest version

```
> pip install --upgrade git+https://github.com/CFEL-CMI/richmol.git
```

## Planned features & improvements

* Link our Fortran-based HyFor code for computing quadrupole, spin-spin, and spin-rotation hyperfine effects
* Computation of time-dependent probability density functions on grids (simulations of VMI observables)


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

