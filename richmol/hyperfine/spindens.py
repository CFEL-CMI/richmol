from richmol.rotdens import _stateEulerGrid_basis, _stateEulerGrid_rotdens
from richmol.hyperfine import Hyperfine
from richmol.hyperfine.reduced_me import spinMe
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import py3nj
import sys


# Cartesian-to-spherical rank-1 tensor transformation: (x, y, z) -> (-1, 0, 1)
Us = np.array([[np.sqrt(2)/2, -np.sqrt(2)*1j/2, 0],
               [0, 0, 1],
               [-np.sqrt(2)/2, -np.sqrt(2)*1j/2, 0]], dtype=np.complex128),

# spherical-to-Cartesian rank-1 tensor transformation: (-1, 0, 1) -> (x, y z)
Ux = np.linalg.pinv(Us)


def spinDensity(h: Hyperfine,
                grid: NDArray[np.float_],
                m_val: int,
                diag: bool = False,
                state_filter: Callable = lambda **_: True,
                c2_thresh: float = 1e-08):

    # contracts rovibrational functions with three-j symbol
    # (-1)^(F + m_F) \sum_{m_J}  (   F    I    J)  |J,k,v>(chi, theta, phi)
    #                            (-m_F  m_I  m_J)

    def psi_threej(rv_quanta, spin_quanta, F):
        filter = lambda **kw: (int(kw['J']), kw['sym'], int(kw['ind'])) in rv_quanta

        if hasattr(h, 'rotdens') and len(list(h.rotdens.keys())) > 0:
            rv_psi = _stateEulerGrid_rotdens(h, grid, m_val=None, state_filter=filter) 
        elif hasattr(h, 'basis'):
            rv_psi = _stateEulerGrid_basis(h, grid, m_val=None, state_filter=filter)
        else:
            raise AttributeError(f"input parameter 'h' has neither 'rotdens' nor 'basis' " +\
                                 f"attributes which are necessary to compute rotational density")

        psi_3j = []
        for (J, rv_sym, rv_ind), I in zip(rv_quanta, spin_quanta):
            vib, psi = rv_psi[J][rv_sym][rv_ind]

            mi = [int(m*2) for m in np.linspace(-I, I+1, 2*I+1)]
            mj = [int(m*2) for m in np.linspace(-J, J+1, 2*J+1)]
            mij = np.array([(m1, m2) for m1 in mi for m2 in mj])
            n = len(mij)
            threej = py3nj.wigner3j([int(F*2)]*n, [int(I*2)]*n, [int(J*2)]*n,
                        [-int(m_val*2)]*n, mij[:, 0], mij[:, 1])
            threej = threej.reshape(len(mi), len(mj))

            psi = np.dot(threej, psi) # (mi, mj) * (v, mj, ipoint) -> (mi, v, ipoint)
            psi_3j.append((vib, rv_psi))

        return psi_3j

    # computes spin matrix elements <I', m_I' | I_a(n) | I,m_I>, a = x, y, z

    def spin_me(I1, I2):
        spins = h.spins
        me = np.zeros((3, len(spins), 2*I1+1, 2*I2+1), dtype=np.complex128)
        reduced_me = spinMe([I1], [I2], spins, 1, 'spin')
        for im1, m1 in enumerate(np.linspace(-I1, I1+1, 2*I1+1)):
            for im2, m2 in enumerate(np.linspace(-I2, I2+1, 2*I2+1)):
                pow = I1 - m1
                ipow = int(pow)
                assert (pow - ipow < 1e-6), f"I1 - m1 = {I1} - {m1} is non-integer"
                threej = py3nj.wigner3j([int(I1*2)]*3, [2]*3, [int(I2*2)]*3,
                                        [-int(m1*2)]*3, [-2, 0, 2], [int(m2*3)]*3)
                me[:, :, im1, im2] = (-1)**ipow * np.dot(Ux, threej)[:, None] * reduced_me[:, 0, 0]
        return me


    for f, vec_f in h.eigvec.items():
        for sym, vec_sym in vec_f.items():
            for istate in range(vec_sym.shape[-1]):
                if state_filter(F=f, sym=sym, ind=istate):

                    ind = np.where(abs(vec_sym[:, istate])**2 >= c2_thresh)[0].tolist()
                    quanta = [h.quantaRovib[f][sym][i] for i in ind]
                    rv_quanta = [(j, rv_sym, rv_ind) for (sp, j, rv_sym, k, rv_ind) in quanta]
                    spin_quanta = [sp[-1] for (sp, j, rv_sym, k, rv_ind) in quanta]
                    psi_3j = psi_threej(rv_quanta, spin_quanta, f)


