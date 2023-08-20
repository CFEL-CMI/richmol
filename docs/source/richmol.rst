.. currentmodule:: richmol

Public API: richmol package
***************************

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   richmol.rot

Unit conversion (:code:`convert_units`)
---------------------------------------
We use `scipy.constants <https://docs.scipy.org/doc/scipy/reference/constants.html>`_
for the reference values of all physcial constants

.. autosummary::

   richmol.convert_units.MHz_to_invcm
   richmol.convert_units.J_to_invcm
   richmol.convert_units.Debye_to_si
   richmol.convert_units.Debye_to_au
   richmol.convert_units.Debye_to_sqrt_erg_x_sqrt_cm3
   richmol.convert_units.Debye_x_Vm_to_invcm
   richmol.convert_units.Debye_x_Vm_to_MHz
   richmol.convert_units.Buckingham_to_si
   richmol.convert_units.Buckingham_to_au
   richmol.convert_units.Buckingham_to_sqrt_erg_x_sqrt_cm5
   richmol.convert_units.AUdip_x_Vm_to_invcm
   richmol.convert_units.AUpol_x_Vm_to_invcm


Molecule-field interaction potential (:code:`field`)
----------------------------------------------------

.. autosummary::

   richmol.field.CarTens
   richmol.field.filter

.. autoclass:: richmol.field.CarTens

    .. automethod:: filter
    .. automethod:: tomat
    .. automethod:: assign
    .. automethod:: full_form
    .. automethod:: block_form
    .. automethod:: mul
    .. automethod:: field
    .. automethod:: vec
    .. automethod:: __mul__
    .. automethod:: __add__
    .. automethod:: __sub__
    .. automethod:: store
    .. automethod:: read


.. autofunction:: richmol.field.filter


Rotational density (:code:`rotdens`)
------------------------------------

.. autosummary::

   richmol.rotdens.dens_grid
   richmol.rotdens.psi_grid


TROVE bindings (:code:`trove`)
------------------------------

.. autosummary::

   richmol.trove.CarTensTrove

.. autoclass:: richmol.trove.CarTensTrove

    .. automethod:: read_states
    .. automethod:: read_trans


Time-dependent solutions (:code:`tdse`)
---------------------------------------

.. autosummary::

   richmol.tdse.TDSE

.. autoclass:: richmol.tdse.TDSE

    .. automethod:: time_grid
    .. automethod:: init_state
    .. automethod:: update


Spectra (:code:`spectrum`)
--------------------------

.. autosummary::

   richmol.spectrum.FieldFreeSpec

.. autoclass:: richmol.spectrum.FieldFreeSpec

    .. automethod:: linestr
    .. automethod:: abs_intens
    .. automethod:: totxt


