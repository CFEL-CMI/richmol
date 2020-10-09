An overview of Richmol
**********************
Richmol development was initiated as part of ERC Marie Skłodowska-Curie project
"Rotationally-Induced Chirality in Molecules", where one of the goals was the development
of a general molecule program for simulations of time-dependent rotational-vibrational
dynamics of molecules in external electric fields. (..a bit more details later)

The package provides a wide range of tools to support variational calculations
of the rotational-vibrational energy levels, spectra, field-induced dynamics
and related properties of a general molecule, calculations based on rotational
effective-Hamiltonian models, and calculations including the nuclear spin hyperfine effects.

How to cite
===========
Bibtex entry::

        @article{Richmol,
          author       = {Alec Owens and Andrey Yachmenev},
          title        = {{RichMol}: A general variational approach for rovibrational molecular dynamics in
                          external electric fields},
          year         = 2018,
          volume       = 148,
          number       = 12,
          pages        = 124102,
          journal      = {The Journal of Chemical Physcis}
          doi          = {10.1063/1.5023874},
          url          = {https://doi.org/10.1063/1.5023874},
          archiveprefix= {arXiv},
          eprint       = {1802.07603},
        }

Features
========

* Calculation of rotational energies and spectra of molecules using the effective-Hamiltonian
  models, e.g., rigid rotor, Watson A and S Hamiltonians. In principle, arbitrary user-defined
  Hamiltonian expressed in terms of the angular momentum operators :math:`\hat{J}`, :math:`\hat{J}_z`,
  :math:`\hat{J}_\pm`, and their powers can be set up.
 
* Calculation of nuclear spin hyperfine effects, such as nuclear quadrupole, spin-rotational,
  and spin-spin interactions.

* Calculation of time-dependent wavepacket dynamics of molecules subject to external electric
  or/and magnetic fields. The type of calculation here depends on the input field-free basis,
  which can be the pure rotational states obtained by watie module, the vibrational or
  rotational-vibrational states calculated with external variational programs, such as TROVE,
  or even the hyperfine rotational or rotational-vibrational states.

Design
======
write about the hierarchy of various interactions based on richmol-format files

