import urllib.request
import os
import h5py
from richmol.field import CarTens
import numpy as np

"""
This example shows how to convert '*.rchm' files produced by TROVE or HyFor into
richmol database format.
"""

richmol_file = "h2o_p48_j40_hyperfine.h5"

# dictionary describing where energies and matrix elements files (*.rchm files) are stored locally
# key = dataset name, value = (path to *.rchm files, comment string)

new_data = {
    "h0" : ("hyfor_energies_f0.0_f39.0.chk",
            "Authors: Andrey Yachmenev (CFEL/DESY) and Sergei Yurchenko (UCL). " + \
            "TROVE/HyFor variationally calculated hyperfine energies and transitions " + \
            "for H_2^16O. Basis: p48, J: 0..40, PES: [10.1098/rsta.2017.0149]. " + \
            "Calculated ro-vibrational energies substituted by empirical values " + \
            "[10.1016/j.jqsrt.2012.10.002] where available. Hyperfine interactions " + \
            "included spin-spin (magnetic dipole-dipole) and spin-rotation coupling. " + \
            "The rovibrational energies and matrix elements of hyperfine tensor " + \
            "operators as well as dipole and quadrupole tensors can be found on " + \
            "Zenodo [10.5281/zenodo.4986069]"),
    "dipole" : ("DMS_QUAD_hyperfine/matelem_MU_ns_f<f1>_f<f2>.rchm",
                "hyperfine electric dipole moment in Debye, DMS: [10.1063/1.5043545]"),
    "quad" : ("DMS_QUAD_hyperfine/matelem_QUAD_ns_f<f1>_f<f2>.rchm",
              "hyperfine electric quadrupole moment in atomic units, ab initio surface: " + \
              "CCSD(T)/aug-cc-pwCVQZ using CFOUR"),
}

for key in new_data.keys():
    print(f"read dataset '{key}' from '{new_data[key][0]}'")
    if key == "h0":
        states_file = new_data[key][0]
        data = CarTens(new_data[key][0])
    else:
        data = CarTens(states_file, new_data[key][0])

    print(f"add dataset '{key}' to '{richmol_file}'")
    data.store(richmol_file, name=key, comment=new_data[key][1], replace=True)

