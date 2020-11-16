""" Tools to work with Richmol database files """
import numpy as np
import gzip


def read_coefs(filename, coef_thresh=1.0e-16):
    """ Reads Richmol coefficients file """
    if filename.endswith('.gz'):
        fl = gzip.open(filename, "r")
    else:
        fl = open(filename, "r")
    max_nelem = 0
    nstates = 0
    for line in fl:
        w = line.split()
        max_nelem = max([int(w[5]), max_nelem])
        nstates+=1

    dt = [('J', 'i4'), ('id', 'i8'), ('ideg', 'i4'), ('sym', 'U10'), ('enr', 'f8'), \
          ('n', 'i8'), ('c', np.complex128, [max_nelem]), ('v', 'i8', [max_nelem]), \
          ('k', 'i4', [max_nelem])]
    coefs = np.zeros(nstates, dtype=dt)

    fl.seek(0)
    istate = 0
    for line in fl:
        w = line.split()
        J = int(w[0])
        id = int(w[1])
        sym = w[2]
        ideg = int(w[3])
        enr = np.float64(w[4])
        nelem = int(w[5])
        coefs['J'][istate] = J
        coefs['id'][istate] = id
        coefs['sym'][istate] = sym
        coefs['ideg'][istate] = ideg
        coefs['enr'][istate] = enr
        n = 0
        for ielem in range(nelem):
            c = np.float64(w[6+ielem*4])
            im = int(w[7+ielem*4])
            if abs(c)**2<=coef_thresh: continue
            coefs['c'][istate,n] = c*{0:1,1:1j}[im]
            coefs['v'][istate,n] = int(w[8+ielem*4])
            coefs['k'][istate,n] = int(w[9+ielem*4])
            n+=1
        coefs['n'][istate] = n
        istate+=1
    fl.close()
    return coefs


def read_vibme(filename):
    """ Reads Richmol file with vibrational matrix elements """
    if filename.endswith('.gz'):
        fl = gzip.open(filename, "r")
    else:
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
        vibme[:,i-1,j-1] = [np.float64(ww) for ww in w[2:]]
    fl.close()
    return vibme
