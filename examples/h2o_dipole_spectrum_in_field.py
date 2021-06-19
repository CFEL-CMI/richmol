"""An example of calculation of the field-dressed rovibrational dipole
spectrum of water

Authors: @saribalc (@yachmena - data in Zenodo repository)
"""

import urllib.request
import os
import h5py
import numpy as np
import scipy as sp
import scipy.constants as const
import matplotlib.pyplot as plt
from richmol.field import CarTens
from richmol.convert_units import (
    Debye_x_Vm_to_invcm, Debye_to_sqrt_erg_x_sqrt_cm3
)
import sys


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


# cartesian tensor operators
def filt(**kwargs):
    pass_J, pass_enr = True, True
    if 'J' in kwargs:
        pass_J = kwargs['J'] <= 5
    if 'enr' in kwargs:
        pass_enr = kwargs['enr'] <= 1e4
    return pass_J and pass_enr
h0 = CarTens(richmol_file, name='h0', bra=filt, ket=filt) # (1/cm)
mu = CarTens(richmol_file, name='dipole', bra=filt, ket=filt) # (D)

# field-dressed eigenstates
field = [0, 0, 1e3] # (V/m)
h = h0 - mu * field * Debye_x_Vm_to_invcm() # (1/cm)
enrs, vecs = np.linalg.eigh(h.tomat(form='full', repres='dense')) # (1/cm)

# field-dressed linestrength
linestr = []
for cart in mu.cart:
    linestr.append(
        abs(
            vecs.conjugate().T.dot(
                mu.tomat(form='full', repres='dense', cart=cart).dot(vecs)
            )
        )**2
    )
linestr = sp.sparse.coo_matrix(np.triu(sum(linestr), 1)) # (D^2)
linestr *= Debye_to_sqrt_erg_x_sqrt_cm3()**2 # (erg*cm^3)

# absorption intensities
abun = 1.0 # natural terrestrial isotopic abundance
temp = 296.0 # temperature (K)
part_sum = 174.5813 # total internal partition sum
c_1 = const.h * 1e2 * const.c / (const.k * temp) # (cm)
c_2 = 8 * sp.pi**3 / (3 * 1.0e9 * const.h * const.c * part_sum) # (1/erg/cm)
abs_intens_func = lambda R, E_low, v : c_2 * abun * v * R \
    * np.exp(-c_1 * E_low) * (1 - np.exp(-c_1 * v))
enrs -= np.amin(enrs)
E_low = enrs[linestr.row]
freq = enrs[linestr.col] - E_low
abs_intens = abs_intens_func(linestr.data, E_low, freq)

# figure
plt.scatter(freq, abs_intens, s=1, label='all transitions')
plt.xlabel(r'frequency ($cm^{-1}$)')
plt.ylabel(r'intensity ($cm^{-1}$/($molecule \cdot cm^{-2}$))')
plt.yscale('log')
plt.ylim(bottom=1e-36)
plt.xlim(left = 0)
plt.legend(loc='best')
plt.title('absorption intensities')
plt.tight_layout()
plt.savefig('spectrum.png', dpi=500, format='png')
plt.close()

# assignments txt
_, assign_ket = h0.assign(form='full')
leadings_inds = np.argmax(vecs, axis=0)
assign = []
for ind, leading_ind in enumerate(leadings_inds):
    qstr, _ = assign_ket['k'][leading_ind]
    sym = assign_ket['sym'][leading_ind]
    assign.append((enrs[ind],  abs(vecs[leading_ind, ind])**2, sym, qstr))
assign = np.array(
    assign,
    dtype = [('enr', 'f8'), ('coef', 'f8'), ('sym', 'S2'), ('qstr', '<S50')]
)
np.savetxt('assign.txt', assign, fmt='%12.6f %6.4f %3s %s')

# spectrum txt
spec = np.empty(
    abs_intens.shape,
    dtype = [ ('bra ind', 'i4'), ('ket ind', 'i4'),
              ('freq', 'f8'), ('linestr', 'f8'), ('abs intens', 'f8') ]
)
spec['bra ind'] = linestr.row
spec['ket ind'] = linestr.col
spec['freq'] = freq
spec['linestr'] = linestr.data
spec['abs intens'] = abs_intens
np.savetxt('spectrum.txt', spec, fmt='%5.0f %5.0f %12.6f %16.6e %16.6e')
