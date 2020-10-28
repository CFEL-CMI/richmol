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


Rotational-vibrbational wave function
=====================================

In Richmol the rotational-virbational wavefunction is represented as as sum-of-products of rotational and vibrbational basis functions

.. math::

    |\Psi_{J,M,h} \rangle = \sum_{v}\sum_{\tau = 0,1} \sum_{K= \tau }^J c^{(Jh)}_{vK\tau} |v\rangle |J,K,M,\tau \rangle

where :math:`J` is the total angular momentum quantum number, :math:`M` is the projection of the total angular momentum onto laboratory-fixed :math:`Z`-axis
:math:`K` is the absolute value of the projection of the total angular momentum onto molecule-fixed :math:`z`-axis and :math:`\tau` is the basis function parity. The vibrational basis functions
are solutions to field-free Schrodinger equation at :math:`J=0` and are generally denoted as :math:`|v\rangle`.
Here :math:`|J,K,M,\tau\rangle` is the symmetric-top basis function in the parity-operator representation, which is related by a unitary transformation to bare symmetric-top functions as follows


.. math::

    |J,K,M,\tau\rangle=d_{K, \tau}|J,K,M\rangle + d_{-K, \tau}|J,-K,M\rangle

with coefficients :math:`d_{K, \tau} = \frac{1}{\sqrt{2}}` and :math:`d_{-K, \tau} = \frac{(-1)^{\tau}}{\sqrt{2}}`, such that the rotational parity is given by :math:`(-1)^{\tau}`. The bare symmetric-top functions
are defined as :math:`|J,K,M\rangle = \left(\frac{2J+1}{8\pi^2}\right)^{\frac{1}{2}}D^{(J)*}_{MK}(\phi,\theta,\chi)`, where :math:`\phi,\theta,\chi` are three Euler angles given ZYZ' convention relating the molecule-fixed frame
with laboratory-fixed frame.



Molecule-field interaction
==========================

In general we shall denote laboratory-frame cartesian tensor of rank :math:`\Omega` with :math:`T^{(\Omega,LF)}_A`, where :math:`A=(i_1,i_2,...,i_{\Omega})` is the tensor's covariant multi-index,
such that, for example, the laboratory-frame electronic polarisability is tensor of rank-2: :math:`\alpha_{ij} \equiv T^{(\Omega=2,LF)}_{i_1,i_2}`.
A general form of the the interaction Hamiltonian matrix elements can be written as

.. math::

    \langle \Psi_{J,M,h}| \hat{H}_{int}|\Psi_{J',M',h'}\rangle = \sum_{\xi} v_{\xi}  \sum_A \langle \Psi_{J,M,h}| T^{(\Omega_{\xi},LF)}_{A_{\xi}} |\Psi_{J',M',h'}\rangle E_{A_{\xi}}

where :math:`E_A = E_{i_1}\cdot E_{i_2}\cdot ... \cdot E_{i_{\Omega_{\xi}}}` denotes the time-dependent electric field tensor
and tensor contraction is carrier out over all indices in the multi-index :math:`A`. :math:`\xi` labels interaction terms of
rank-:math:`\Omega_{\xi}`, such as
dipole, polarisability and hyperpolarisailities and :math:`v_{\xi}` are prefactors standing before the interaction terms:
:math:`v_{1}=-1, v_{2}=-\frac{1}{2}, v_{3}=-\frac{1}{6}` etc.

.. note::

  For example Watie carries an option to calculate and store matrix elements of cartesian tensors in the laboratory-frame,
  which can be used in solving the time-dependent Schrodinger equation.
  When one is interested in simulating
  rigid molecule dynamics in external electric fields, the molecule-field interaction is most commonly written as

  .. math::

        \hat{H}_{int} = -\boldsymbol{\mu}\mathbf{E}-\frac{1}{2}\mathbf{E}^T\boldsymbol{\alpha}\mathbf{E}

  where :math:`\boldsymbol{\mu}` is the dipole moment operator, :math:`\boldsymbol{\alpha}` is the electronic polarisability operator (induced dipole moment) and :math:`\mathbf{E}` is time-dependent electric field vector.


.. note::

  The laboratory-frame matrix elements of cartesian tensors :math:`\langle \Psi_{J,M,h}| T^{(\Omega,LF)}_A|\Psi_{J',M',h'}\rangle` can be calculated with watie's function me().


Watie uses a convenient method for calculating the laboratory-frame matrix elements of cartesian tensors :math:`\langle \Psi_{J,M,h}| T^{(\Omega,LF)}_A|\Psi_{J',M',h'}\rangle` utilizing the
properties of irreducible spherical tensor operators.

In the first step the laboratory-frame cartesian tensor is transformed into the laboratory-frame irreducible spherical tensor form by the following linear transformation:

.. math::

      T^{(\Omega,LF)}_A = \sum_{\omega=0}^{\Omega} \sum_{\sigma = -\omega}^{\omega} \mathbf{U}^{(\Omega)\dagger}_{\omega \sigma,A}   T^{(\Omega,LF)}_{\omega \sigma}

where is a collection of irreducible spherical tensor operators :math:`T^{(\Omega,LF)}_{\omega \sigma}` where :math:`\omega = 0,1,...,\Omega` and :math:`\sigma = -\omega,...,\omega`.
Note that :math:`\sum_{\omega=0}^{\Omega} \sum_{\sigma = -\omega}^{\omega} = \Omega^2`, such that the number of elements in :math:`T^{(\Omega,LF)}_{\omega \sigma}` and :math:`T^{(\Omega,LF)}_A` is equal.
The cartesian-to-spherical transformation matrix :math:`\mathbf{U}^{(\Omega)\dagger}_{\omega \sigma,A}` can be directly evaluated with the aid of spherical tensor composition formulas:


.. math::

      \mathbf{U}^{(\Omega)}_{\omega \sigma,A}= \sum_{\sigma_1 = -\omega_1} ^{\omega_1}\sum_{\sigma_2 = -\omega_2} ^{\omega_2}   \langle \omega_1 \sigma_1 \omega_2 \sigma_2 | \Omega \sigma \rangle \mathbf{U}^{(\Omega_1)}_{\omega_1\sigma_1,\tilde{A}_1} \mathbf{U}^{(\Omega_2)}_{\omega_2\sigma_2,\tilde{A}_2}

where :math:`\mathbf{U}^{(\Omega_i)}_{\omega_i\sigma_i,\tilde{A}_i}, i=1,2` are lower-rank spherical tensor representation matrices and :math:`\langle \omega_1 \sigma_1 \omega_2 \sigma_2 | \Omega \sigma \rangle` is the Clebsch-Gordan coefficient.
The combined ranks of composite representations  :math:`\Omega_1+\Omega_2` must equal the rank of the output representation :math:`\Omega`. Also the indices must satisfy the relation :math:`A= \tilde{A}_1 \bigcup \tilde{A}_2`.
It is therefore sufficient only to know the lowest rank (rank-1) cartesian-to-spherical transformation matrix, which is given as


.. math::

      \mathbf{U}^{(1)} = \begin{bmatrix}
                \frac{1}{\sqrt{2}} &-\frac{i}{\sqrt{2}}  & 0 \\
                0 & 0  & 1 \\
                -\frac{1}{\sqrt{2}} &-\frac{i}{\sqrt{2}}  & 0 \\
            \end{bmatrix}



Spherical tensor representation carries a number of advantages. First and foremost, the spherical-tensor objects have identical transformation properties to symmetric-top basis functions.
This leads to particularly elegant formulas for the matrix elements of spherical tensors in the symmetric-top basis. Secondly, the spherical tensor form allows one to directly
derive selection rules in the symmetric-top basis, as the tensors and the basis functions span identical representation.

The laboratory-fixed fram spherical tensors :math:`T^{(\Omega,LF)}_{\omega \sigma}` are rotated to the molecule-fixed frame with the following transformation:

.. math::

    T^{(\Omega,LF)}_{\omega \sigma} = \sum_{\sigma'=-\omega}^{\omega} D^{(\omega)*}_{\sigma \sigma'} T^{(\Omega,MF)}_{\omega \sigma'}

where :math:`D^{(\omega)*}_{\sigma \sigma'}` are elements of the Wigner-D matrix representation for the total angular momentum :math:`\omega`. In watie the Wigner-D matrices are parametrized with the
Euler angles :math:`\theta,\phi,\chi` in the ZYZ' convention. Finally, the molecule-fixed spherical tensor form :math:`T^{(\Omega,MF)}_{\omega \sigma'}` can be transformed back to the cartesian form
by means of the relation

.. math::

      T^{(\Omega,LF)}_{\omega \sigma}  = \sum_{A} \mathbf{U}^{(\Omega)}_{\omega \sigma,A} T^{(\Omega,LF)}_A

such that the laboratory-fixed and molecule-fixed cartesian tensor operators are related by the following linear transformation

.. math::

      T^{(\Omega,LF)}_A = \sum_{A'} W_{AA'}^{(\Omega)} T^{(\Omega,MF)}_{A'}

where :math:`W_{AA'}^{(\Omega)} =  \sum_{\omega=0}^{\Omega} \sum_{\sigma = -\omega}^{\omega}  \sum_{\sigma' = -\omega}^{\omega} \mathbf{U}^{(\Omega)\dagger}_{\omega \sigma,A} D^{(\omega)*}_{\sigma \sigma'} \mathbf{U}^{(\Omega)}_{\omega \sigma',A'}`.
The purpose of transforming the laboratory-fixed frame cartesian tensors to the molecule-fixed frame is two-fold: 1) the experimentally available data on electronic polarisabilities, dipole moments etc. is given in the molecular frame; 2) the rotational wavefunctions
is a function of Euler angles which link laboratory and molecular frames - so that the integration over the Euler angles must invovle a molecule-fixed operator.


The cartesian tensor matrix elements can be now calculated

.. math::

    \langle \Psi_{J,M,h}| T^{(\Omega,LF)}_A|\Psi_{J',M',h'}\rangle =  \sum_{A'} \langle \Psi_{J,M,h}| W_{AA'}^{(\Omega)} T^{(\Omega,MF)}_{A'}|\Psi_{J',M',h'}\rangle =  \sum_{\omega=0}^{\Omega} M^{(JMJ'M')}_{\omega A} K^{(JhJ'h')}_{\omega}

where the :math:`\textit{M-tensor}` is defined as

.. math::

    M^{(JMJ'M')}_{\omega A} = \sqrt{(2J+1)(2J'+1)}(-1)^M \sum_{\sigma=-\omega}^{\omega} \mathbf{U}^{(\Omega)\dagger}_{\omega \sigma,A} \begin{pmatrix}
              J' & \omega  & J \\
              M' & \sigma  & -M
          \end{pmatrix}

where :math:`\begin{pmatrix} J' & \omega  & J \\ M' & \sigma  & -M \end{pmatrix}` is the 3-j symbol. The :math:`\textit{K-tensor}` is defined as

.. math::

 K^{(JhJ'h')}_{\omega} =\sum_{v K \tau} \sum_{v' K' \tau'} c_{vK\tau}^{(Jh)*}c_{v'K'\tau'}^{(J'h')} \sum_{\sigma'=-\omega}^{\omega} F_{KK'\tau \tau' \omega \sigma'}^{(JJ')} \sum_{A'} \mathbf{U}^{(\Omega)}_{\omega \sigma',A'} \langle v | T^{(\Omega,MF)}_{A'} | v' \rangle_Q

where

.. math::

  F_{KK'\tau \tau' \omega \sigma'}^{(JJ')} = \sum_{s,s'=0}^1 d_{(-1)^sK, \tau}d_{(-1)^{s'}K', \tau'}(-1)^K \begin{pmatrix}
            J' & \omega  & J \\
            (-1)^{s'}K' & \sigma'  & (-1)^{s+1}K
        \end{pmatrix}


In elements :math:`\langle v | T^{(\Omega,MF)}_{A'} | v' \rangle_Q` the integration is carried over all :math:`D` internal coordinates of the system, denoted with :math:`Q = (Q_1,Q_2,...,Q_D)`.
The molecule-fixed cartesian elements of molecule-field interaction tensors :math:`T^{(\Omega,MF)}_{A'}` are functions of internal coordinates and are routinely evaluated by fitting a predefined functional form
to a set of point-calculations at various geometries of the system.

.. note::

  The elements of the :math:`\textit{K-tensor}` carry information about the molecule-fixed properties of the rotational-vibrational wavefunctions involved in the transition, whereas
  the :math:`\textit{M-tensor}` refers to laboratory-fixed properties. Both tensors are stored in richmol-format files <molecule_name>_<tensor_name>_j<J>_j<J'>.rchm.

Finally the matrix elements of the general field-matter interaction Hamiltonian ca be written in a compact form as


.. math::

  \langle \Psi_{J,M,h}| \hat{H}_{int}(t)|\Psi_{J',M',h'}\rangle = \sum_{\xi} v_{\xi} \sum_{\omega=0}^{\Omega_{\xi}} K^{(JhJ'h')}_{\omega,\xi}
  \tilde{M}^{(JMJ'M')}_{\omega,\xi}(t)

where

.. math::

  \tilde{M}^{(JMJ'M')}_{\omega,\xi}(t) = \sum_{A_{\xi}} M^{(JMJ'M')}_{\omega A_{\xi},\xi}(t) E_{A_{\xi}}

Time-dependent Schrödinger equation
===================================


The rotational-vibrational wavefunction :math:`|\Phi(t)\rangle` is propagated in time using the time-evolution operators

.. math::

  |\Phi(t)\rangle = U(t,t_0)|\Phi(t_0)\rangle

with :math:`U(t_0,t_0)=1`, where split-operator method was used to approximate :math:`U(t,t_0)` as

.. math::

  U(t,t_0) = \exp\left[-i\frac{\Delta t}{2\hbar}\hat{H}_{mol}\right] \exp\left[-i\frac{\Delta t}{\hbar}\hat{H}_{int}\left(\frac{t+t_0}{2}\right)\right] \exp\left[-i\frac{\Delta t}{2\hbar}\hat{H}_{mol}\right] +\mathcal{O}\left((\Delta t)^3\right)

where :math:`\Delta t = t-t_0` is the time-step, which is user-specified and must be sufficiently small for a given molecule-field interaction Hamiltonian. The time-dependent wavefunction is represented in the
spectral basis of the molecular Hamiltonian :math:`\hat{H}_{mol}`:

.. math::

  |\Phi(t_0)\rangle = \sum_{J=J_{min}}^{J_{max}}\sum_{M=M_{min}}^{M_{max}}\sum_{h} C_{JMh}(t)|\Psi_{J,M,h}\rangle

where :math:`J_{min},J_{max}` and :math:`M_{min},M_{max}` are specified by the user. Evaluation of exponents containing the molecular Hamiltonian is straightforward:

.. math::

  U(t,t_0) = \exp\left[-i\frac{\Delta t}{2\hbar}\hat{H}_{mol}\right] |\Phi(t_0)\rangle  = \sum_{J=J_{min}}^{J_{max}}\sum_{M=M_{min}}^{M_{max}}\sum_{h} C_{JMh}(t_0)\exp\left[-i\frac{\Delta t}{2\hbar}E_{J,h}\right]|\Psi_{J,M,h}\rangle

The matrix representation of the interaction Hamiltonian part in the split-time-evolution operator is calculated with a Krylov subspace method.

.. math::

  \langle \Psi_{J',M',h'}|\exp\left[-i\frac{\Delta t}{\hbar}\hat{H}_{int}\left(\frac{t+t_0}{2}\right)\right]|\Psi_{J,M,h}\rangle\approx
  \mathbf{A}_p\mathbf{Z}_p^*e^{-i\mathbf{D}_ph}\mathbf{Z}_p\mathbf{A}_p^*

where :math:`\mathbf{A}_p` is the projection matrix from the full field-free basis :math:`|\Psi_{J,M,h}\rangle` onto the Krylov sub-space of size
:math:`p`.  :math:`e^{-i\mathbf{D}_ph}` and  :math:`\mathbf{Z}_p`  are the diagonal matrix of eigenvalues of the interaction Hamiltonian exponents
and the diagonalizing transformation, respectively. The scheme of the Krylov method is displayed in the figure below:

.. image:: krylov.pdf


The matrix-vector products (MVP) of the interaction Hamiltonian with the time-dependent coefficients vector :math:`C_{JMh}(t)` is used
to construct the Krylov sub-space basis. In Richmol, these MVPs (denoted with :math:`\mathbf{Hc}` in the figure above)


 .. math::

  y_{J,M,h}(t) = \sum_{J',M',h'} \langle \Psi_{J,M,h} | \hat{H}_{int}(t))| \Psi_{J',M',h'}\rangle \cdot
  C_{J',M',h'}(t))

are calculated sequentially in the following way

.. math::

 y^{(1)}_{J,M,h,J',M',\xi}(t) = \sum_{h'}  K^{(JhJ'h')}_{\omega,\xi} C_{J',M',h'}(t)\\



   y_{J,M,h}(t) = \sum_{J',M',\xi} v_{\xi} y^{(1)}_{J,M,h,J',M',\xi}(t) \tilde{M}^{(JMJ'M')}_{\omega,\xi}(t)
