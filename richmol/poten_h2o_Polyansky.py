""" Potential energy surface of water molecule from our friend Oleg Polyansky
see http://rsta.royalsocietypublishing.org/content/376/2115/20170149
"""
import numpy as np
import os.path
from ctypes import CDLL, c_int, RTLD_GLOBAL

dll_path = os.path.join(os.path.dirname(__file__), 'poten_h2o_Polyansky')
dll = np.ctypeslib.load_library('rsta20170149supp1', dll_path)

def poten(coords):
    r1 = np.asfortranarray(coords[:,0])
    r2 = np.asfortranarray(coords[:,1])
    alpha = np.asfortranarray(coords[:,2])
    npoints = coords.shape[0]
    npoints_c = c_int(npoints)
    v = np.asfortranarray(np.zeros(npoints, dtype=np.float64))
    dll.water_poten.argtypes = [ \
            c_int,
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='F') ]
    dll.water_poten.restype = None
    dll.water_poten(npoints_c, r1, r2, np.cos(alpha), v)
    # the output value 'v' is in Hartree, convert it to cm^-1
    cm = 219474.624
    return v*cm
