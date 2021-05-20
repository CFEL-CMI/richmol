import numpy as np
from numpy.polynomial.legendre import legval, legder
from numpy.polynomial.hermite import hermval, hermder
import functools
import operator
import itertools
import math
import opt_einsum


def potme(bra, ket, poten, weights, nmax=None, w=None):
    """Matrix elements of potential function in product basis"""
    if isinstance(bra, (tuple, list)):
        fbra = prod2(*bra, nmax=nmax, w=w)
    else:
        fbra = bra
    if isinstance (ket, (tuple, list)):
        fket = prod2(*ket, nmax=nmax, w=w)
    else:
        fket = ket
    return opt_einsum.contract('kg,lg,g,g->kl', np.conj(fbra), fket, poten, weights)


def vibme(bra, ket, dbra, dket, gmat, weights, nmax=None, w=None):
    """Matrix elements of vibrational kinetic energy in product basis"""
    # assert (gmat.shape[1] == gmat.shape[2]), f"bogus shape of G-matrix: {gmat.shape}, dimensions 2 and 3 must be equal"
    # assert (gmat.shape[0] == len(weights)), f"dimension 0 of G-atrix = {gmat."

    nq = gmat.shape[1]

    # products of bra basis sets
    fbra = []
    if isinstance(bra, (tuple, list)):
        assert (len(bra) == len(dbra)), f"length of `bra` (={len(bra)}) != length of `dbra` (={len(dbra)})"
        for i in range(nq):
            print(i)
            bra_ = [elem if ielem != i else dbra[i] for ielem,elem in enumerate(bra)]
            fbra.append(prod2(*bra_, nmax=nmax, w=w))
    else:
        assert (nq == 1), f"`gmat` dimensions 1 and 2 (={gmat.shape[1]}) != length of `bra` = {len(bra)}"
        for i in range(nq):
            fbra.append(dbra)

    # products of ket basis sets
    fket = []
    if isinstance(ket, (tuple, list)):
        assert (len(ket) == len(dket)), f"length of `ket` (={len(ket)}) != length of `dket` (={len(ket)})"
        for i in range(nq):
            print(i)
            ket_ = [elem if ielem != i else dket[i] for ielem,elem in enumerate(ket)]
            fket.append(prod2(*ket_, nmax=nmax, w=w))
    else:
        assert (nq == 1), f"`gmat` dimensions 1 and 2 (={gmat.shape[1]}) != length of `ket` = {len(ket)}"
        for i in range(nq):
            fket.append(dket)

    me = np.zeros((fbra[0].shape[0], fket[0].shape[0]), dtype=np.float64)

    for i in range(gmat.shape[1]):
        for j in range(gmat.shape[2]):
            print(i,j)
            me = me + opt_einsum.contract('kg,lg,g,g->kl', np.conj(fbra[i]), fket[j], gmat[:,i,j], weights)
            # me = me + np.einsum('kg,lg,g,g->kl', np.conj(fbra[i]), fket[j], gmat[:,i,j], weights)
    return me


def prod2(*funcs, nmax=None, w=None):
    """Product basis set using einsum"""
    npts = funcs[0].shape[1]
    assert (all(f.shape[1] == npts for f in funcs)), f"input arrays in `funcs` have different second dimensions"
    if nmax is None:
        nmax = max([f.shape[0] for f in funcs])
    if w is None:
        w = [1 for i in range(len(funcs))]

    psi = funcs[0]

    n = np.einsum('i,j->ij', [i for i in range(len(funcs[0]))], np.ones(len(funcs[1])))
    nsum = n * w[0]

    for ifunc in range(1, len(funcs)):
        psi = np.einsum('kg,lg->klg', psi, funcs[ifunc])

        n2 = np.einsum('i,j->ij', np.ones(len(psi)), [i for i in range(len(funcs[ifunc]))])
        nsum = nsum + n2 * w[ifunc]

        ind = np.where(nsum <= nmax)
        psi = psi[ind]

        if ifunc <= len(funcs)-2:
            nsum = np.einsum('i,j->ij', nsum[ind], np.ones(funcs[ifunc+1].shape[0]))
    return psi


def prod(*funcs, nmax=None, w=None):
    """Product basis set using itertools"""
    npts = funcs[0].shape[1]
    assert (all(f.shape[1] == npts for f in funcs)), f"input arrays in `funcs` have different second dimensions"
    if nmax is None:
        nmax = max([f.shape[0] for f in funcs])
    if w is None:
        w = [1 for i in range(len(funcs))]
    n = ([i for i in range(f.shape[0])] for f in funcs)
    combs = [elem for elem in itertools.product(*n) if sum(nn*ww for nn,ww in zip(elem,w))<=nmax ]
    res = np.zeros((len(combs), npts), dtype=funcs[0].dtype)
    for icomb, comb in enumerate(combs):
        res[icomb, :] = functools.reduce(operator.mul, [f[n, :] for f, n in zip(funcs, comb)])
    return res


def hermite(nmax, r, r0, alpha):
    """Normalized Hermite functions and derivatives

    f(r) = 1/(pi^(1/4) 2^(n/2) sqrt(n!)) exp(-1/2 x^2) Hn(x)
    df(r)/dr = 1/(pi^(1/4) 2^(n/2) sqrt(n!)) exp(-1/2 x^2) (-Hn(x) x + dHn(x)/dx) * alpha
    where x = (r - r0) alpha

    NOTE: returns f(r) and df(r) without weight factor exp(-1/2 x^2)
    """
    x = (r - r0) * alpha
    sqsqpi = np.sqrt(np.sqrt(np.pi))
    c = np.diag([1.0 / np.sqrt(2.0**n * math.factorial(n)) / sqsqpi for n in range(nmax+1)])
    f = hermval(x, c)
    df = (hermval(x, hermder(c, m=1)) - f * x) * alpha
    return f, df


def legendre(nmax, r, a, b):
    """Normalized Legendre functions and derivatives

    NOTE: NEED TO CHECK THE COORDINATE SCALING !!!!!

    f(r) = Ln(x) * (b-a)/2
    df(r)/dr = dLn(x)/dx * ((b-a)/2)^2
    """
    x = 0.5 * (b - a) * r + 0.5 * (a + b)
    fac = 0.5 * (b - a)
    c = np.diag([np.sqrt((2.0 * n + 1) * 0.5) * fac for v in range(nmax+1)])
    f = legval(x, c)[:,:,0]
    df = legval(x, legder(c, m=1))[:,:,0] * fac
    return f, df
