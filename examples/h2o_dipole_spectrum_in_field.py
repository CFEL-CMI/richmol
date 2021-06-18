"""An example of calculation of the field-dressed rovibrational dipole
spectrum of water

Authors: @saribalc, @yangguang888 (@yachmena - data in Zenodo repository)
"""

import urllib.request
import os
import h5py
from richmol.field import CarTens
# from richmol import spectrum

# get richmol file

richmol_file = "h2o_p48_j40_rovib.h5"
if not os.path.exists(richmol_file):
    url = "https://zenodo.org/record/4943807/files/h2o_p48_j40_rovib.h5"
    print(f"download richmol file from {url}")
    urllib.request.urlretrieve(url, "h2o_p48_j40_rovib.h5")
    print("download complete")

# print available datasets and their description

fl = h5py.File(richmol_file, "r")
print("Available datasets")
for key, val in fl.items():
    print(f"'{key}'")                   # dataset name
    print("\t", val.attrs["__doc__"])   # dataset description

# ... do spectrum 
