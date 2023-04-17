from richmol.rotdens import _stateEulerGrid_basis, _stateEulerGrid_rotdens
from richmol.hyperfine import Hyperfine
import numpy as np
from typing import Callable
from numpy.typing import NDArray
from collections import defaultdict


def spinDensity(h: Hyperfine,
                grid: NDArray[np.float_],
                diag: bool = False,
                state_filter: Callable = lambda **_: True,
                c2_thresh: float = 1e-08):

    mydict = lambda: defaultdict(mydict)

    # apply input state filter function to select hyperfine states
    # and determine indices of rovibrational states with significant contributions

    rv_states = mydict()
    for f, vec_f in h.eigvec.items():
        for sym, vec_sym in vec_f.items():
            for istate in range(vec_sym.shape[-1]):
                if state_filter(J=f, sym=sym, ind=istate):
                    ind, = np.where(abs(vec_sym[:, istate])**2 >= c2_thresh)
                    spin, j, rv_sym, k, rv_ind = h.quantaRovib[f][sym][ind]
                    rv_states[(j, rv_sym)].append(*rv_ind)

    rv_states = [(j, sym, ind) for (j, sym), val in rv_states.items() for ind in list(set(val))]

    # rovibrational state filter for `rotdens._stateEulerGrid_...` function

    def rovib_state_filter(**kw):
        j = int(kw['J'])
        sym = kw['sym'].lower()
        ind = int(kw['ind'])
        return (j, sym, ind) in rv_states

    # compute rovibrational wavefunctions

    if hasattr(h, 'rotdens') and len(list(h.rotdens.keys())) > 0:
        psi = _stateEulerGrid_rotdens(h, grid, m_val=None, state_filter=rovib_state_filter) 
    elif hasattr(h, 'basis'):
        psi = _stateEulerGrid_basis(h, grid, m_val=None, state_filter=rovib_state_filter)
    else:
        raise AttributeError(f"input parameter 'h' has neither 'rotdens' nor 'basis' " +\
                             f"attributes which are necessary to compute rotational density")

    # --------------------------- CONTINUE FROM HERE --------------------------
    # h.quantaSpinJSym = (spin, j, sym, dim)
    # h.quanta = (enr, (spin, j, sum, k, rvInd))

    # jmax = max(list(psi.keys()))
    # im = [m for m in range(-jmax, jmax+1)].index(m)  # only for m_val=None in _stateEulerGrid_...

    for j1, psi_j1 in psi.items():
            for sym1, psi_sym1 in psi_j1.items():
                    for (ind1, v1, psi1) in psi_sym1:

                            # vboth = list(set(v2) & set(v1))
                            # vind1 = [v1.index(v) for v in vboth]
                            # vind2 = [v2.index(v) for v in vboth]
                            # d = np.einsum('vg,vg,g->g', np.conj(psi1[vind1, im, :]),
                            #               psi2[vind2, im, :], np.sin(grid[:, 1]))

    vec1 = self.eigvec[f1][sym1][:, ind1]
    vec2 = self.eigvec[f2][sym2][:, ind2]
    nz_ind1 = np.where(abs(vec1)**2 >= c2_thresh)
    nz_ind2 = np.where(abs(vec2)**2 >= c2_thresh)

    states1 = [(j, sym, rvInd) for (spin, j, sym, k, rvInd), enr in self.quanta_k1]
    states2 = [(j, sym, rvInd) for (spin, j, sym, k, rvInd), enr in self.quanta_k2]

    def state_filter1(**kw):
        j = int(kw['J'])
        sym = kw['sym'].lower()
        ind = int(kw['ind'])
        return (j, sym, ind) in states1

    def state_filter2(**kw):
        j = int(kw['J'])
        sym = kw['sym'].lower()
        ind = int(kw['ind'])
        return (j, sym, ind) in states2

    psi1 = _stateEulerGrid(self, grid, state_filter=state_filter1)
    psi2 = _stateEulerGrid(self, grid, state_filter=state_filter2)

    # for i in nz_ind1:
    #     (spin, j, sym, k, rvInd), enr = self.quanta_k1[i]
    #     mi = [int(m*2) for m in np.arange(-spin[-1], spin[-1]+1)]
    #     mj = [int(m*2) for m in np.arange(-j, j+1)]
    #     mij = np.array([(m1, m2) for m1 in mi for m2 in mj])
    #     n = len(mij)
    #     threej = py3nj.wigner3j([int(f1*2)]*n, [int(spin[-1]*2)]*n, [int(j*2)]*n,
    #                             [-int(mf1*2)]*n, mij[:, 0], mj[:, 1])

    # spin, j, sym, k = self.quanta_k1[f1][sym1][ind1][0]

    # for i, (spin, j, rvSym, dim) in enumerate(self.quantaSpinJSym[f1][sym1]):
    #     mi = [int(m*2) for m in np.arange(-spin[-1], spin[-1]+1)]
    #     mj = [int(m*2) for m in np.arange(-j, j+1)]
    #     mij = np.array([(m1, m2) for m1 in mi for m2 in mj])
    #     n = len(mij)
    #     threej = py3nj.wigner3j([int(f*2)]*n, [int(spin[-1]*2)]*n, [int(j*2)]*n,
    #                             [-int(mF*2)]*n, mij[:, 0], mj[:, 1])
    #     threej = threej.reshape(len(mi), len(mj))
    #     func = np.array([elem[1] for elem in psi[j][rvSym]]) # (l, v, mj, rg)
    #     np.einsum('ij,lvjg->ilvg', threej, func) # (mi, mj) * (l, v, mj, rg)
    #     threej, psi[j][rvSym][irv][1][v, m, g]

