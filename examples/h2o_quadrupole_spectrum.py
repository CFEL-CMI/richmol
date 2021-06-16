"""An example of calculation of the quadrupole spectrum of water

Authors: @saribalc (@yachmena - data in Zenodo repository)
"""

import urllib.request
import os
import h5py
from richmol.field import CarTens
from richmol.spectrum import FieldFreeSpec

# get richmol file

richmol_file = "h2o_p48_j40_rovib.h5"
if not os.path.exists(richmol_file):
    url = "https://zenodo.org/record/4943807/files/h2o_p48_j40_rovib.h5"
    print(f"download richmol file from {url}")
    urllib.request.urlretrieve(url, "h2o_p48_j40_rovib.h5")
    print("download complete")

# print available datasets and their description

with h5py.File(richmol_file, "r") as fl:
    print("Available datasets")
    for key, val in fl.items():
        print(f"'{key}'")                   # dataset name
        print("\t", val.attrs["__doc__"])   # dataset description

# ... do spectrum

# initialization
spec = FieldFreeSpec(
    richmol_file,
    names = ['h0', 'quad'],
    j_max = 30,
    type = 'elec',
    order = 'quad',
    units = 'a.u.',
    e_max = 1e4
)

# linestrengths
spec.linestr(thresh=0)

# absorption intensities
temp, part_sum = 296.0, 174.5813
spec.abs_intens(temp, part_sum, abun=1, thresh=1e-36)
