richmol.rot package
*******************

.. autosummary::

   richmol.rot.Molecule
   richmol.rot.solve
   richmol.rot.Solution
   richmol.rot.mol_tensor
   richmol.rot.LabTensor

Rigid molecule description
--------------------------

.. autoclass:: richmol.rot.Molecule

    .. automethod:: XYZ
    .. automethod:: frame
    .. automethod:: B
    .. automethod:: B_geom
    .. automethod:: sym
    .. automethod:: store_xyz
    .. automethod:: imom
    .. automethod:: gmat
    .. automethod:: linear
    .. automethod:: kappa
    .. automethod:: store
    .. automethod:: read

.. autofunction:: mol_tensor

Rotational solutions
--------------------

.. autofunction:: richmol.rot.solve
.. autoclass:: richmol.rot.Solution

    .. automethod:: store
    .. automethod:: read

Matrix elements
---------------

.. autoclass:: richmol.rot.LabTensor

