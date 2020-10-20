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
The molecular rotational Hamiltonian can be set up using molecular rotational constants
:math:`B_x, B_y, B_z` as

.. math::

        \hat{H}_{mol} = B_x\hat{J}_x^2 + B_y\hat{J}_y^2 + B_z\hat{J}_z^2

for non-linear molecules, and as

.. math::

        \hat{H}_{mol} = B\hat{J}^2

for linear (:math:`B=B_x=B_y,~B_z=\infty`) and spherical-top (:math:`B=B_x=B_y=B_z`) molecules.
The rotational constants can be computed form the equilibrium geometry of a molecule, i.e.,
Cartesian coordinates of atoms, or, when available, taken from experimental measurements.
This form of Hamiltonian assumes that the molecule-fixed frame is oriented within a molecule such that
the :math:`x,y,z` axes coincide with the principal axes of inertia :math:`I_x, I_y, I_z`.
Put differently, the molecular inertia tensor is diagonal, i.e.,
:math:`I_{\alpha\beta}=I_\alpha\delta_{\alpha\beta}~(\alpha,\beta=x,y,z)`.
Such coordinate system is called the Principal Axes System (PAS).

In order to change molecular coordinate system to PAS, we need to compute and diagonalize
the moment of inertia tensor :math:`\mathbf I`, i.e.,

.. math::

        \mathbf{I} = \mathbf{V} \mathbf{I}^{\rm diag} \mathbf{V}^{-1}.

Afterwards, rotate with the eigenvector matrix :math:`\mathbf{V}^{-1}`
the Cartesian coordinates of atoms
:math:`\mathbf{r}_{\rm atom}` as well as all molecular multipole moment tensors, such  as, for example,
dipole moment :math:`\boldsymbol{\mu}` or polarizability :math:`\boldsymbol{\alpha}`, i.e.,

.. math::

        \mathbf{r}_{\rm atom}^{\rm(PAS)} &= \mathbf{V}^{-1}\mathbf{r}_{\rm atom} \\
        \boldsymbol{\mu}^{\rm(PAS)} &= \mathbf{V}^{-1}\boldsymbol{\mu} \\
        \boldsymbol{\alpha}^{\rm(PAS)} &= \mathbf{V}^{-1}\boldsymbol{\alpha}\mathbf{V}.

The rotational constants are computed as

 .. math::

        B_\alpha = \frac{\hbar^2}{2hc I_\alpha^{\rm diag}}~~(\alpha=x,y,z).

where :math:`c` is the speed of light and :math:`h` is the Planck's constant.

Quantization axis
-----------------
The eigenvalues of the inertia tensor after diagonalization are sorted such that
:math:`I_x^{\rm diag}\leq I_y^{\rm diag} \leq I_z^{\rm diag}`, accordingly
:math:`B_x\geq B_y\geq B_z`.
In spectroscopic literature, rotational constants sorted in the descending order are labelled
as :math:`A\geq B\geq C` and the corresponding PAS axes as :math:`a,b,c`.

One is free to permute the :math:`x,y,z` axes without any effect on final results,
such as energies, wave functions, or time-evolution dynamics in fields.
However, the change in the axes system also changes the representation to the rotational wave
functions, so that the spectroscopic assignments can vary.
It is often beneficial to choose certain PAS axis as the :math:`z` axis in order to simplify
calculations of the observables, such as, for example, expectation values of :math:`\cos^2\theta`
(where :math:`\theta` is the Euler angle between laboratory :math:`Z` and molecular :math:`z`
axes).

.. note::

        Computed observables, such as energies, spectra, and time-evolution dynamics must not depend
        on the choice of the coordinate system, i.e., the orientation of :math:`x,y,z` axes in molecule
        as well as the choice of quantization axis, i.e., whether the Hamiltonian is
        :math:`\hat{H}_{mol} = A\hat{J}_x^2 + B\hat{J}_y^2 + C\hat{J}_z^2`, or
        :math:`\hat{H}_{mol} = C\hat{J}_x^2 + A\hat{J}_y^2 + B\hat{J}_z^2`, or
        :math:`\hat{H}_{mol} = B\hat{J}_x^2 + C\hat{J}_y^2 + A\hat{J}_z^2`, etc.

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

The solutions of the `spherical top` and `symmetric top` problems are symmetric-top functions
:math:`|J,k,m\rangle`, which form a complete basis set. For a more general problem of rotation of `asymmetric top` molecules
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
        constant.
        The expression for the Hamiltonian is then identical for all cases, i.e.,
        :math:`\hat{H}_{mol} = B_x\hat{J}_x^2 + B_y\hat{J}_y^2 + B_z\hat{J}_z^2`.

Other choices of axes
---------------------
Another popular choice of the molecular axes is along the principal moments of molecular
polarizability tensor. In general, in order to change the coordinate system,
one follows the same procedure as for the PAS, with the molecular
polarizability tensor put in place of the moment of inertia tensor.
The rotational Hamiltonian is then constructed as

 .. math::

        \hat{H}_{mol} = \frac{1}{2}\sum_{\alpha,\beta=x,y,z}G_{\alpha,\beta}\hat{J}_\alpha\hat{J}_\beta,

where :math:`G_{\alpha,\beta}` is the rotational kinetic energy matrix.

Typical computational protocol using ``watie`` module
-----------------------------------------------------

* Input molecular parameters

  * Cartesian coordinates of atoms for the equilibrium configuration, using :func:`RigidMolecule.XYZ`,
    together with the values of dipole moment vector, polarizability tensor, etc., using
    :func:`RigidMolecule.tensor`.
    These can be computed `ab initio` or taken (inferred) from spectroscopic experiment.

  * Experimental values of rotational constants, if available.

* Choose coordinate system, using :func:`RigidMolecule.frame`, as PAS, principal polarizability
  frame, etc. If necessary, permute :math:`x,y,z` axes, using :func:`RigidMolecule.frame`.

* For selected values of quantum number of the total angular momentum :math:`J`:

  * Set up basis of symmetric-top functions, using :func:`SymtopBasis`.

    * If necessary, symmetrize basis functions for a selected point-symmetry or rotation symmetry group
      :math:`D_2`, using :func:`symmetrize`.

  * Set up rotational Hamiltonian using rotational constants (in case of PAS) or kinetic :math:`G` matrix (in
    case of other frame choices). For this, use :func:`JJ` (:math:`\hat{J}^2`), :func:`Jxx`
    (:math:`\hat{J}_x^2`), :func:`Jxy` (:math:`\hat{J}_{x}\hat{J}_{y}`), ..., :func:`Jzz` (:math:`\hat{J}_z^2`)
    operators. The kinetic matrix :math:`G` is computed by :func:`RigidMolecule.gmat`.

  * Compute matrix elements of Hamiltonian in the basis of symmetric-top functions, using
    :func:`SymtopBasis.overlap`

  * Compute eigenvalues and eigenvectors of Hamiltonian using, e.g., :func:`numpy.linalg.eigh`.

  * If necessary, transform the initial basis of symmetric-top functions to the eigenfunction representation
    of rotational Hamiltonian, using :func:`SymtopBasis.rotate`.


Molecule-field interaction
==========================
The molecule-field interaction is represented in the dipole approximation as

 .. math::

        \hat{H}_{int} = -\boldsymbol{\mu}\mathbf{E}-\frac{1}{2}\mathbf{E}^T\boldsymbol{\alpha}\mathbf{E}

where :math:`\boldsymbol{\mu}` is the dipole moment operator, :math:`\boldsymbol{\alpha}` is the electronic polarisability operator (induced dipole moment) and :math:`\mathbf{E}` is time-dependent electric field vector.
The laboratory-fixed dipole moment :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\alpha}` are transformed into the molecular-frame,
to make use of the molecule-fixed polarisability and dipole moment tensor elements which are obtained from experiment or can be calculated with electronic structure packages.
To do so, we first transform all laboratory-fixed Cartesian tensors into appropriate irreducible spherical tensor form.
A general transformation of the Cartesian tensor :math:`\mathbf{T}^{(cart)}` of rank-l to its irreducible spherical tensor form :math:`\mathbf{T}^{(l)}_q` can be written as

.. math::

       \mathbf{T}^{(l)}_q = \sum_{i_1,i_2,...,i_l} \mathbf{U}^{(l)}_{q,i_1,i_2,...,i_l} \mathbf{T}^{(cart)}_{i_1,i_2,...,i_l}

where :math:`q=-l,-l+1,...,l-1,l` label the components of the irreducible spherical tensor operator.
Spherical tensor representation carries a number of advantages. First and foremost, the spherical-tensor objects have identical transformation properties to symmetric-top basis functions.
This leads to particularly elegant formulas for the matrix elements of spherical tensors in the symmetric-top basis. Secondly, the spherical tensor form allows one to directly
derive selection rules in the symmetric-top basis, as they span identical representation.
The laboratory-fixed cartesian dipole moment and polarisability tensors  :math:`\mu_{i}` and :math:`\alpha_{ij}`, :math:`i,j=X,Y,Z`  are transformed into their spherical tensor forms as follows

.. math::

       \boldsymbol{\alpha}^{(2)}_q = \sum_{A,B=X,Y,Z} \mathbf{U}^{(2)}_{q,AB} \boldsymbol{\alpha}_{AB}\\
       \boldsymbol{\mu}^{(1)}_q = \sum_{A=X,Y,Z} \mathbf{U}^{(1)}_{q,A} \boldsymbol{\mu}_{A}

where

.. math::

      \mathbf{U}^{(1)} = \begin{bmatrix}
                \frac{1}{\sqrt{2}} &-\frac{i}{\sqrt{2}}  & 0 \\
                0 & 0  & 1 \\
                -\frac{1}{\sqrt{2}} &-\frac{i}{\sqrt{2}}  & 0 \\
            \end{bmatrix}

Higher rank irreducible spherical tensor representations can be conveniently constructed from lower rank representations with the following formulas

.. math::
          \mathbf{T}^{(l)}_q = \sum_{q_1} \langle l_1 q_1 l_2 q_2 | l q \rangle \mathbf{T}^{(l_1)}_{q_1} \mathbf{T}^{(l_2)}_{q_2}

where :math:`l_1+l_2 = l` and :math:`\langle l_1 q_1 l_2 q_2 | l q \rangle` is the Clebsh-Gordan coefficient. The irreducible spherical tensor representation for the rank-2 polarisability tensor
can be therefore written as

.. math::

      \mathbf{U}^{(2)}_{q,i_1i_2} = \sum_{\sigma_1}  \langle 1 \sigma_1 1 q-\sigma_1 | 2 q \rangle \mathbf{U}^{(1)}_{\sigma_1,i_1} \mathbf{U}^{(1)}_{q-\sigma_1,i_2}

Time-dependent Schrödinger equation
===================================
