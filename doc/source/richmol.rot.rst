richmol.rot package
*******************

.. currentmodule:: richmol.rot

.. autosummary::

   Molecule
   solve
   Solution
   mol_tensor
   LabTensor

Rigid molecule description
--------------------------

.. autoclass:: Molecule

    .. automethod:: XYZ
    .. automethod:: frame
    .. automethod:: ABC
    .. automethod:: B
    .. automethod:: sym
    .. automethod:: store_xyz
    .. automethod:: imom
    .. automethod:: gmat
    .. automethod:: linear
    .. automethod:: kappa
    .. automethod:: ABC_geom
    .. automethod:: B_geom
    .. automethod:: store
    .. automethod:: read

.. autofunction:: mol_tensor

Rotational solutions
--------------------

.. autofunction:: solve
.. autoclass:: Solution

    .. automethod:: store
    .. automethod:: read

Matrix elements
---------------

.. autoclass:: LabTensor

