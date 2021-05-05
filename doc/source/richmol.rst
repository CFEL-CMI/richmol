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
   richmol.convert_units.Debye_to_au
   richmol.convert_units.Debye_x_Vm_to_invcm
   richmol.convert_units.AUdip_x_Vm_to_invcm
   richmol.convert_units.AUpol_x_Vm_to_invcm


Modelling field-dependent problems (:code:`field`)
--------------------------------------------------

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
    .. automethod:: read_states
    .. automethod:: read_trans

.. autofunction:: richmol.field.filter

