Theoretical backgrounds
***********************
A basic knowledge of the underlying theory of molecular nuclear motion dynamics is necessary
to be able to use Richmol.
Here we describe some theoretical aspects and outline mathematical steps behind computing
the rotational-vibrational energies, spectra, and field dynamics of molecules.

Molecular rotations
===================
The rotational energy levels, wave functions, spectra, and matrix elements of multipole-moment
Cartesian tensor operators (e.g., dipole moment, polarizability, quadrupole moment, etc.)
are computed by the module ``watie``.

Principal axes system
---------------------
The molecular rotational Hamiltonian can be set up from the molecular rotational constants
:math:`B_x, B_y, B_z` as

.. math::

        \hat{H} = B_x\hat{J}_x^2 + B_y\hat{J}_y^2 + B_z\hat{J}_z^2.

The rotational constants can be computed form the equilibrium geometry of the molecule, i.e.,
Cartesian coordinates of atoms, or, when available, taken from experimental measurements.
This form of Hamiltonian assumes that the molecule-fixed frame is oriented within molecule such that
the axes :math:`x,y,z` coincide with the moments of inertia :math:`I_x, I_y, I_z`.
Put differently, the molecular inertia tensor is diagonal, i.e.,
:math:`I_{\alpha\beta}=I_\alpha\delta_{\alpha\beta}~(\alpha,\beta=x,y,z)`.
Such coordinate system is called the Principal Axes System (PAS).

In order to change the molecular coordinate system to PAS, one computes and diagonalizes
the moment of inertia tensor :math:`\mathbf I`, i.e.,

.. math::

        \mathbf{I} = \mathbf{V} \mathbf{I}^{\rm diag} \mathbf{V}^{-1},

and rotates with the eigenvector matrix :math:`\mathbf{V}^{-1}`
all Cartesian vectors, which include Cartesian coordinates of atoms
:math:`\mathbf{r}_{\rm atom}` and all molecular Cartesian tensors, such  as, for example,
dipole moment :math:`\boldsymbol{\mu}` or polarizability :math:`\boldsymbol{\alpha}`, i.e.,

.. math::

        \mathbf{r}_{\rm atom}^{\rm(PAS)} &= \mathbf{V}^{-1}\mathbf{r}_{\rm atom} \\
        \boldsymbol{\mu}^{\rm(PAS)} &= \mathbf{V}^{-1}\boldsymbol{\mu} \\
        \boldsymbol{\alpha}^{\rm(PAS)} &= \mathbf{V}^{-1}\boldsymbol{\alpha}\mathbf{V}.

The rotational constants are computed as

 .. math::

        B_\alpha = \frac{\hbar^2}{2hc I_\alpha^{\rm diag}}~~(\alpha=x,y,z).


.. note::

        The `rotation` of all Cartesian quantities in ``watie`` module is done automatically
        following the user's choice of molecular frame, which is done by setting
        :func:`watie.RigidMolecule.frame` to "pas", for example.
        There is no need to do the above rotations of Cartesian coordinates and tensors manually
        (in place).

Quantization axis
-----------------
The eigenvalues of inertia tensor after diagonalization are sorted such that
:math:`I_x^{\rm diag}\leq I_y^{\rm diag} \leq I_z^{\rm diag}`, accordingly
:math:`B_x\geq B_y\geq B_z`.
In spectroscopic literature rotational constants sorted in descending order are labelled
as :math:`A\geq B\geq C` and the corresponding PAS axes as :math:`a,b,c`.

One is free to permute the :math:`x,y,z` axes without any effect on the final results,
such as energies, wave functions, or time-evolution dynamics in fields.
However, a particular choice of the axes can change complexity of the rotational wave
functions and as a result their spectroscopic assignments.
Sometimes it is beneficial to choose certain PAS axis as the :math:`z` axis in order to simplify
calculations of observables, such as, for example, expectation values of :math:`\cos^2\theta`
(where :math:`\theta` is the Euler angle between the laboratory :math:`Z` and molecular :math:`z`
axes).

.. note::

        Computed observables, such as energies, spectra, and time-evolution dynamics must not depend
        on the choice of the coordinate system, i.e., the orientation of :math:`x,y,z` axes in molecule
        as well as the choice of quantization axis, i.e., whether the Hamiltonian is
        :math:`\hat{H} = A\hat{J}_x^2 + B\hat{J}_y^2 + C\hat{J}_z^2`, or
        :math:`\hat{H} = C\hat{J}_x^2 + A\hat{J}_y^2 + B\hat{J}_z^2`, or
        :math:`\hat{H} = B\hat{J}_x^2 + C\hat{J}_y^2 + A\hat{J}_z^2`, etc.

There are few special types of molecules with two out of three or all three rotational constants
being equal, i.e., `spherical top` molecules with :math:`A=B=C`, `prolate symmetric top` molecules
with :math:`A>B=C`, and `oblate symmetric top` molecules with :math:`A=B>C`.
Linear molecules have only two rotational constants :math:`A=B`.
For these special cases the Schrödinger equation with rotational Hamiltonian can be solved
exactly provided the :math:`x,y,z` axes are chosen in a certain way.
For `prolate symmetric top` molecules the :math:`z` axis must be aligned along the smallest
inertia :math:`a` axis.
For `oblate symmetric top` molecules the :math:`z` axis must be aligned along the largest inertia
:math:`c` axis.

The solutions of `spherical top` and `symmetric top` problems are symmetric-top functions
:math:`|J,k,m\rangle`. For more general problem of rotation of `asymmetric top` molecules
(molecules with all three rotational constants different :math:`A>B>C`) the wave function
is build as a linear combination of symmetric-top functions, so the choice of the :math:`x,y,z`
axes in an `asymmetric top` molecule will affect the linear combination coefficients,
however not the energies. When a molecule is close to either of the special types, i.e.,
being `near prolate top` with :math:`A>B\approx C` or `near oblate top` with :math:`A\approx B>C`,
it might be beneficial, mainly for the state assignment purposes, to choose the :math:`z` in accord
with the corresponding type of `symmetric top`.
To tell which of `symmetric top` types a given molecule is close to, one can compute the so-called
asymmetry parameter :math:`\kappa=(2B-A-C)/(A-C)`, which is equal to :math:`-1` for a `prolate
symmetric top` and :math:`+1` for an `oblate symmetric top`.

.. note::

        A standard spectroscopic convention is the following: when :math:`\kappa\simeq -1`,
        permute the :math:`x,y,z` axes such that :math:`B_z` becomes the largest
        rotational constant, and when :math:`\kappa\simeq +1`, permute
        the :math:`x,y,z` such that :math:`B_z` becomes the smallest rotational
        constant. The Hamiltonian is
        :math:`\hat{H} = B_x\hat{J}_x^2 + B_y\hat{J}_y^2 + B_z\hat{J}_z^2`.
        In ``watie`` the axes permutations can be set by a user using the molecular frame function
        :func:`watie.RigidMolecule.frame()`.

Other choices of axes
---------------------
Another popular choice of the molecular axes is along the principal moments of molecular
polarizability tensor. In general, one follows the same procedure as for the PAS, with the molecular
polarizability tensor put in place of the moment of inertia tensor.
The rotational Hamiltonian is then constructed as

 .. math::

        \hat{H} = \frac{1}{2}\sum_{\alpha,\beta=x,y,z}G_{\alpha,\beta}\hat{J}_\alpha\hat{J}_\beta,

where :math:`G_{\alpha,\beta}` is the rotational kinetic energy matrix.


Molecule-field interaction
==========================


Time-dependent Schrödinger equation
===================================
