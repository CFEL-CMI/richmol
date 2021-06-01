import numpy as np
from numpy.polynomial.legendre import legval, legder
from numpy.polynomial.hermite import hermval, hermder
from orthnet import Legendre_Normalized, Hermite2
import functools
import operator
import itertools
import math
import opt_einsum
import torch
from torch import nn
import sys

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
    return opt_einsum.contract('kg,lg,g,g->kl', torch.conj(fbra), fket, poten, weights)


def vibme(bra, ket, dbra, dket, gmat, weights, nmax=None, w=None):
    """Matrix elements of vibrational kinetic energy in product basis"""

    assert (gmat.shape[1] == gmat.shape[2]), f"bad shape of G-matrix: {gmat.shape}, dimensions 2 and 3 must be equal"

    nq = gmat.shape[1]

    # products of bra basis sets
    fbra = []
    if isinstance(bra, (tuple, list)):
        assert (len(bra) == len(dbra)), f"length of `bra` (={len(bra)}) != length of `dbra` (={len(dbra)})"
        for i in range(nq):
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
            ket_ = [elem if ielem != i else dket[i] for ielem,elem in enumerate(ket)]
            fket.append(prod2(*ket_, nmax=nmax, w=w))
    else:
        assert (nq == 1), f"`gmat` dimensions 1 and 2 (={gmat.shape[1]}) != length of `ket` = {len(ket)}"
        for i in range(nq):
            fket.append(dket)

    me = torch.zeros((fbra[0].shape[0], fket[0].shape[0]), dtype=torch.float64)

    for i in range(gmat.shape[1]):
        for j in range(gmat.shape[2]):
            me = me + opt_einsum.contract('kg,lg,g,g->kl', torch.conj(fbra[i]), fket[j], gmat[:,i,j], weights)
    return me


def prod2(*fn, nmax=None, w=None):
    """Product basis set using einsum

    Args:
        fn : list of arrays (no_funcs, no_points)
            Primitive basis sets, each containing `no_funcs` functions on quadrature grid of `no_points`
            points, note that all basis sets must be defined with respect to the same quadrature grid
        nmax : int
            Maximal value of quantum number in the product basis, defined as
            nmax >= w[0] * n[0] + w[1] * n[1] + w[2] * n[2] ... , where n[0], n[1], n[2] ... are indices
            of functions in each basis set, i.e., n[i] in range(fn[i].shape[0]), and w are weights
        w : list
            Weights for computing quantum number in the product basis

    Returns:
        psi : array (no_funcs, no_points)
            Product basis, containing `no_funcs` product functions on quadrature grid of `no_points` points
    """

    npts = fn[0].shape[1]
    assert (all(f.shape[1] == npts for f in fn)), f"input arrays in `*fn` have different second dimensions: {[f.shape for f in fn]}"
    if nmax is None:
        nmax = torch.max([f.shape[0] for f in fn])
    if w is None:
        w = torch.from_numpy(np.array([1 for i in range(len(fn))]))

    psi = fn[0]

    n = opt_einsum.contract('i,j->ij', torch.from_numpy(np.array([i for i in range(len(fn[0]))])), torch.ones(len(fn[1])))
    nsum = n * w[0]

    for ifn in range(1, len(fn)):
        psi = opt_einsum.contract('kg,lg->klg', psi, fn[ifn])

        n2 = opt_einsum.contract('i,j->ij', torch.ones(len(psi)), torch.from_numpy(np.array([i for i in range(len(fn[ifn]))])))
        nsum = nsum + n2 * w[ifn]

        ind = torch.where(nsum <= nmax)
        psi = psi[ind]

        if ifn <= len(fn)-2:
            nsum = opt_einsum.contract('i,j->ij', nsum[ind], torch.ones(fn[ifn+1].shape[0]))
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


def hermite_numpy(nmax, r, r0, alpha): #deprecated, can be deleted, see hermite (in pytorch) below
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

def hermite(nmax, r, r0, alpha):
    """Normalized Hermite functions and derivatives

    f(r) = 1/(pi^(1/4) 2^(n/2) sqrt(n!)) exp(-1/2 x^2) Hn(x)
    df(r)/dr = 1/(pi^(1/4) 2^(n/2) sqrt(n!)) exp(-1/2 x^2) (-Hn(x) x + dHn(x)/dx) * alpha
    where x = (r - r0) alpha

    NOTE: returns f(r) and df(r) without weight factor exp(-1/2 x^2)
    """
    x = torch.from_numpy((r - r0) * alpha)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    sqsqpi = torch.sqrt(torch.sqrt(torch.tensor(np.pi)))
    fac = torch.tensor([1.0 / np.sqrt(2.0**n * math.factorial(n)) / sqsqpi for n in range(nmax+1)])
    f = Hermite2(x, nmax).tensor
    f = f * fac[None, :]
    # numerical derivative, maybe implement analytical derivative
    dh = 0.001
    fdh = Hermite2(x+dh, nmax).tensor
    fdh = fdh * fac[None, :]
    df = (fdh - f)/dh
    df = (df - f * x) * alpha
    return f.transpose(0, 1), df.transpose(0, 1)

def legendre(nmax, r, a, b, r0):
    """Normalized Legendre functions and derivatives

    NOTE: NEED TO CHECK THE COORDINATE SCALING !!!!!

    f(r) = Ln(x) * (b-a)/2
    df(r)/dr = dLn(x)/dx * ((b-a)/2)^2
    """
    r = torch.from_numpy(r)
    x = 0.5 * (b - a) * (r - r0)
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    fac = 0.5 * (b - a)
    #Andrey multiplies in the original code by fac, but I don't think
    # this is correct so I won't for now
    f = Legendre_Normalized(x, nmax).tensor
    # compute derivative using finite differences
    dh = 0.001
    fdh = Legendre_Normalized(x+dh, nmax).tensor
    df = fac*(fdh-f)/dh
    return f.transpose(0,1), df.transpose(0,1)


class InvResnet(nn.Module):
    """an Invertible Resnet: current implementation is to be used to augment
    Legendre 3D basis
    """
    def __init__(self, n_input=3, n_hidden_units=10):
        """
        Args:
            n_input: number of inputs to the NN
            n_hidden_units: number of hidden units in each layer
        """
        super().__init__()
        self.n_input = n_input
        self.n_hidden_units = n_hidden_units
        self.InvBlock = nn.Sequential(nn.Linear(n_input, n_hidden_units), LipSwish(),
                                      nn.Linear(n_hidden_units, n_hidden_units), LipSwish(),
                                      nn.Linear(n_hidden_units, n_input), LipSwish(),
                                      )
    def forward(self, x):
        y = InvTanh()(x)
        y = y + self.InvBlock(y)
        return nn.Tanh()(y)

    def Inverse(self, y):
        """Computes the inverse of the NN model
        """
        x1 = InvTanh()(y)
        x2 = x1
        for i in range(20):
            x2 = x1 - self.InvBlock(x2)
        return nn.Tanh()(x2)

class InvTanh(nn.Module):
    """Implements the inverse of Tanh(x)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5*torch.log((1+x)/(1-x))

class LipSwish(nn.Module):
    """Implements Lipswish activation function: Lipswish(x) = (x/1.1) sigmoid(x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x/1.1)*nn.Sigmoid()(x)

if __name__ == "__main__":
    #Make sure Inverse is computed correctly
    x = torch.Tensor([0.3,0.2,0.9])
    print(x)
    NF = InvResnet()
    y = NF(x)
    print(NF.Inverse(y))
