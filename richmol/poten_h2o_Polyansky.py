""" Potential energy surface of water molecule from our friend Oleg Polyansky
see http://rsta.royalsocietypublishing.org/content/376/2115/20170149
"""
import numpy as np
import os.path
from ctypes import CDLL, c_int, RTLD_GLOBAL

dll_name = os.path.dirname(os.path.abspath(__file__)) + '/poten_h2o_Polyansky/rsta20170149supp1.so'
dll = CDLL(dll_name, mode=RTLD_GLOBAL)

def poten(coords):
    r1 = np.asfortranarray(coords[:,0])
    r2 = np.asfortranarray(coords[:,1])
    alpha = np.asfortranarray(coords[:,2])
    npoints = coords.shape[0]
    npoints_c = c_int(npoints)
    v = np.asfortranarray(np.zeros(npoints, dtype=np.float64))
    dll.poten.argtypes = [ \
            c_int,
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='F'), \
            np.ctypeslib.ndpointer(np.float64, ndim=1, flags='F') ]
    dll.poten.restype = None
    dll.poten(npoints_c, r1, r2, np.cos(alpha), v)
    return v

