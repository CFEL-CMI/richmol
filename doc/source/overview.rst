An overview of Richmol
**********************
Richmol program development was initiated in 2015 as part of ERC Marie Skłodowska-Curie project
"Rotationally-Induced Chirality in Molecules", where one of the goals was development
of a general variational approach for simulations of time-dependent rotational-vibrational
dynamics of molecules in external electric fields.

Initially, Richmol was designed to compute only the solution of the time-dependent Schrödinger
equation, using as a basis field-free rotational-vibrational wave functions calculated externally,
for example, with the general-molecule variational approach
`TROVE <https://github.com/Trovemaster/TROVE>`_.
This initial setting determined the current structure of the program, which consists of
different modules each providing a solution for a specific :math:`H_0+H'` problem in the basis of :math:`H_0`
eigenstates.
For example, ``hyfor`` module adds the nuclear spin hyperfine Hamiltonian (:math:`H'=H_{\rm hf}`)
to the rotational-vibrational Hamiltonian (:math:`H_0=H_{\rm rv}`) and expands hyperfine wave functions
in the basis of rotational-vibrational state functions.
``tdtools`` module in turn can combine a time-dependent field-induced potential :math:`H'(t)`
with the total hyperfine Hamiltonian (:math:`H_0\leftarrow H_{\rm rv}+H_{\rm hf}`)
and expand time-dependent wave function in the basis of hyperfine state functions.
Different modules are communicating via :ref:`Richmol data files`.

The package provides a range of tools to support variational calculations
of molecular rotational-vibrational energy levels, spectra, field-induced dynamics,
and related properties, including the nuclear spin hyperfine effects.

* Interface to `TROVE <https://github.com/Trovemaster/TROVE>`_ general variational
  approach for computing field-free rotational-vibrational states of small molecules
  with high accuracy.

* Interface to `DUO <https://github.com/Trovemaster/Duo>`_ general variational approach
  for computing field-free rotational-vibrational
  states of diatomic molecules, including the non-adiabatic and spin-orbit coupling effects.


How to cite
===========
General Richmol bibtex entry::

        @article{Richmol,
          author       = {Alec Owens and Andrey Yachmenev},
          title        = {{RichMol}: A general variational approach for rovibrational molecular
                          dynamics in external electric fields},
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

Bibtex entries for hyperfine calculations using module ``hyfor``::

        @article{Hyfor1,
          author       = {Andrey Yachmenev and Jochen Küpper},
          title        = {Communication: General variational approach to nuclear-quadrupole
                          coupling in rovibrational spectra of polyatomic molecules},
          journal      = {The Journal of Chemical Physcis},
          volume       = 147,
          year         = 2017,
          number       = 14,
          pages        = 141101,
          doi          = {10.1063/1.5002533},
          url          = {https://doi.org/10.1063/1.5002533},
          archiveprefix= {arXiv},
          eprint       = {1709.08558},
          primaryclass = {physics},
        }

        @article{Hyfor2,
          author       = {Yachmenev, Andrey and Thesing, Linda V. and Küpper, Jochen},
          title        = {Laser-induced dynamics of molecules with strong nuclear quadrupole
                          coupling},
          journal      = {The Journal of Chemical Physcis},
          volume       = 151,
          year         = 2019,
          number       = 24,
          pages        = 244118,
          doi          = {10.1063/1.5133837},
          url          = {https://doi.org/10.1063/1.5133837},
          archiveprefix= {arXiv},
          eprint       = {1910.13275},
          primaryclass = {physics},
        }

        @article{Hyfor3,
          author       = {Andrey Yachmenev and Sergey Yurchenko and Guang Yang and Emil Zak and
                          Jochen Küpper},
          title        = {Theoretical line list for water molecule with hyperfine resolution},
          journal      = {The Journal of Chemical Physcis},
          volume       = xx,
          year         = 2021,
          number       = xx,
          pages        = xxx,
          doi          = {},
          url          = {},
          archiveprefix= {arXiv},
          eprint       = {},
          primaryclass = {physics},
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
  or/and magnetic fields. The type of dynamics here depends on the input field-free basis,
  which can be the pure rotational states obtained by watie module, the vibrational or
  rotational-vibrational states, calculated with external variational programs, such as TROVE,
  or even the hyperfine rotational or rotational-vibrational states, calculated by hyfor module.

Design
======

Richmol Data Files
==================
