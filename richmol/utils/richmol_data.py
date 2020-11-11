""" Set of tools to work with Richmol database files """
import numpy as np
import sys


def read_coefs(filename, coef_thresh=1.0e-16):
    """ Reads Richmol coefficients file """
    states = []
    fl = open(filename, "r")
    for line in fl:
        w = line.split()
        jrot = int(w[0])
        id = int(w[1])
        ideg = int(w[2])
        enr = np.float64(w[3])
        nelem = int(w[4])
        coef = []
        vib = []
        krot = []
        for ielem in range(nelem):
            c = np.float64(w[5+ielem*4])
            im = int(w[6+ielem*4])
            if abs(c)**2<=coef_thresh: continue
            coef.append(c*{0:1,1:1j}[im])
            vib.append(int(w[7+ielem*4]))
            krot.append(int(w[8+ielem*4]))
        states.append({"j":jrot,"id":id,"ideg":ideg,"coef":coef,"v":vib,"k":krot,"enr":enr})
    fl.close()
    return states


def read_vibme(filename):
    """ Reads Richmol file with vibrational matrix elements """
    fl = open(filename, "r")
    dim = 0
    for line in fl:
        w = line.split()
        dim = max([dim,int(w[0]),int(w[1])])
        n = len(w[2:])
        try:
            assert (n==nelem), f"Inconsistent number of tensor elements across different transition states in file {filename}"
        except NameError:
            nelem = n
    vibme = np.zeros((nelem,dim,dim), dtype=np.float64)
    fl.seek(0)
    for line in fl:
        w = line.split()
        i = int(w[0])
        j = int(w[1])
        vibme[:,i,j] = [np.float64(ww) for ww in w[2:]]
    fl.close()
    return vibme

