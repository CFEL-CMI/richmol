import urllib.request
import os
import h5py
from richmol.field import CarTens
import numpy as np

"""
In this example, we add new operator matrix elements (e.g., spin-rotation tensor)
into the existing richmol database, which is stored on Zenodo.
We also perform consistency checks for the data that already exist in the database.
The new matrix element data was calculated using TROVE and stored in '*.rchm' file format.
This example also shows how to convert '*.rchm' files produced by TROVE (or HyFor) into
richmol format.
"""

# existing database with rovibrational matrix elements for water molecule

richmol_file = "h2o_p48_j40_rovib.h5"
url = "https://zenodo.org/record/4943807/files/h2o_p48_j40_rovib.h5"


# dictionary describing where new datasets (*.rchm files) are stored locally
# key = dataset name, value = (path to *.rchm files, comment string)

new_data = {
    "h0" : ("energies_j0_j40_MARVEL_HITRAN.rchm", ""),   # don't add comments since we assume this data already exists in richmol_file
    "dipole" : ("DMS/matelem_MU_j<j1>_j<j2>.rchm", ""),
    "quad" : ("QUAD/matelem_QUAD_j<j1>_j<j2>.rchm", ""),
    "spin-spin H1-H2" : ("SS/matelem_SR_j<j1>_j<j2>.rchm",
                         "spin-spin (magnetic dipole-dipole interaction) tensor for H1-H2 in kHz"),
    "spin-rot H1" : ("SR1_TZ/matelem_SR_j<j1>_j<j2>.rchm",
                     "spin-rotation tensor for H1 in kHz, surface: CCSD(T)/aug-cc-pwCVTZ using CFOUR"),
    "spin-rot H2" : ("SR2_TZ/matelem_SR_j<j1>_j<j2>.rchm",
                     "spin-rotation tensor for H2 in kHz, surface: CCSD(T)/aug-cc-pwCVTZ using CFOUR"),
}

diff_tol = 1e-16

# download existing h5 database file from Zenodo

if not os.path.exists(richmol_file):
    print(f"download richmol file from {url}")
    urllib.request.urlretrieve(url, "h2o_p48_j40_rovib.h5")
    print("download complete")

# read available data from h5 file

old_data = {}
with h5py.File(richmol_file, "r") as fl:
    print(f"Available datasets in '{richmol_file}'")
    for key, val in fl.items():
        print(f"'{key}'")                   # dataset name
        print("\t", val.attrs["__doc__"])   # dataset description
        old_data[key] = None

for key in old_data.keys():
    print(f"read dataset '{key}' from '{richmol_file}'")
    old_data[key] = CarTens(richmol_file, name=key)

# read new data

for key in new_data.keys():
    print(f"read dataset '{key}' from '{new_data[key][0]}'")
    if key == "h0":
        states_file = new_data[key][0]
        data = CarTens(new_data[key][0])
    else:
        data = CarTens(states_file, new_data[key][0])

    # compare two existing datasets to match
    if key in old_data:
        print(f"'{key}' dataset already exists in '{richmol_file}' database, check if they're the same ...")
        nerr = 0
        maxdiff = 0
        for cart in data.cart:
            h0 = old_data[key].tomat(cart=cart)
            h1 = data.tomat(cart=cart)
            for Jpair in h0.keys():
                for sym_pair in h0[Jpair].keys():
                    m0 = h0[Jpair][sym_pair]
                    m1 = h1[Jpair][sym_pair]
                    diff = np.max(np.abs(m0 - m1))
                    maxdiff = max([maxdiff, diff])
                    if diff > diff_tol:
                        nerr += 1
                        print(f"\tmax difference = {diff} > {diff_tol} for " + \
                              f"J = {Jpair}, sym = {sym_pair}, cart = {cart}")
        if nerr > 0:
            print(f"found discrepancies in {nerr} blocks")
        else:
            print(f"\t... matrix elements perfectly match (max difference: {maxdiff})")

    else:
        print(f"add dataset '{key}' to '{richmol_file}'")
        data.store(richmol_file, name=key, comment=new_data[key][1])

