from richmol.rotdens import _stateEulerGrid_basis, _stateEulerGrid_rotdens
from richmol.hyperfine import Hyperfine
from richmol.hyperfine.reduced_me import spinMe
from collections import defaultdict
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import py3nj
import sys


# Cartesian-to-spherical rank-1 tensor transformation: (x, y, z) -> (-1, 0, 1)
Us = np.array([[np.sqrt(2)/2, -np.sqrt(2)*1j/2, 0],
               [0, 0, 1],
               [-np.sqrt(2)/2, -np.sqrt(2)*1j/2, 0]], dtype=np.complex128)

# spherical-to-Cartesian rank-1 tensor transformation: (-1, 0, 1) -> (x, y z)
Ux = np.linalg.pinv(Us)


def spinDensity(h: Hyperfine,
                grid: NDArray[np.float_],
                mF: float,
                diag: bool = False,
                state_filter: Callable = lambda **_: True,
                c2_thresh: float = 1e-08):
    """Computes vibrationally-averaged spin probability density matrix
    on a 3D grid of Euler angles

    Args:
        h : :py:class:`richmol.hyperfine.Hyperfine`
            Hyperfine (spin-rovibrational or spin-rotational) Hamiltonian
        grid : array (:, 3)
            Grid of Euler angle values (in radians), phi, theta, chi = grid(i, :)
        mF : int
            Value of fixed m_F quantum number
        diag : bool
            If True, only diagonal elements of probability density matrix will
            be computed
        state_filter : function
            State filter function to select a desired subspace of states spanned
            by the Hamiltonian `h`, see, for example, `bra` and `ket` parameters
            of :py:class:`richmol.field.CarTens`
        c2_thresh : float
            Threshold for to select only those spin-rovibrational
            contributions to a hyperfine state that have the corresponding
            coefficient c_i such that |c_i|^2 > `c2_thresh`

    Returns:
        dens : nested dict
            Spin probability density matrix for different states, pairs of bra
            and ket J quanta, and symmetries
            Example of printing out the density's Cartesian component 'a'
            (a = 0, 1, 2 for 'x', 'y', 'z') for the nucleus 'n':

            .. code-block:: python

                for (j1, j2), dens_j in dens.items():
                    for (sym1, sym2), dens_sym in dens_j.items():
                        for (i, j), den in dens_sym.items():
                            print(f"density for state pair {(j1, sym1, i)}, {(j2, sym2, j)}")
                            for (phi, theta, chi), d in zip(grid, den[a, n]):
                                print(phi, theta, chi, d)
    """
    # contracts rovibrational functions with three-j symbol
    # (-1)^(F + m_F) \sqrt{2F +1} \sum_{m_J}  (   F    I    J)  |J,k,v>(chi, theta, phi)
    #                                         (-m_F  m_I  m_J)
    def psi_threej(F, quanta, coefs):
        # quanta = [(I, J, rv_sym, rv_ind), (...), ...] set of spin-rovibrational
        #   basis quanta contributing to a hyperfine state
        # coefs = [c1, c2, ..] set of corresponding coefficients

        # rovibrational density
        rv_quanta = [(J, sym, ind) for (In, J, sym, ind) in quanta]
        filter = lambda **kw: (int(kw['J']), kw['sym'], int(kw['ind'])) in rv_quanta
        if hasattr(h, 'rotdens') and len(list(h.rotdens.keys())) > 0:
            rv_psi = _stateEulerGrid_rotdens(h, grid, m_val=None, state_filter=filter) 
        elif hasattr(h, 'basis'):
            rv_psi = _stateEulerGrid_basis(h, grid, m_val=None, state_filter=filter)
        else:
            raise AttributeError(f"input parameter 'h' has neither 'rotdens' nor 'basis' " +\
                                 f"attributes which are necessary to compute rotational density")

        psi_3j = []
        for (In, J, rv_sym, rv_ind), coef in zip(quanta, coefs):
            vib, psi = rv_psi[J][rv_sym][rv_ind]

            I = In[-1]
            mi = [int(m*2) for m in np.linspace(-I, I, int(2*I+1))]
            mj = [int(m*2) for m in np.linspace(-J, J, int(2*J+1))]
            mij = np.array([(m1, m2) for m1 in mi for m2 in mj])
            n = len(mij)
            threej = py3nj.wigner3j([int(F*2)]*n, [int(I*2)]*n, [int(J*2)]*n,
                                    [-int(mF*2)]*n, mij[:, 0], mij[:, 1])
            threej = threej.reshape(len(mi), len(mj))

            fac = (-1)**(F+mF) * np.sqrt(2*F+1) * coef
            psi = fac * np.dot(threej, psi) # (mi, mj) * (v, mj, ipoint) -> (mi, v, ipoint)
            psi = np.transpose(psi, (1, 0, 2)) # (v, mi, ipoint)
            psi_3j.append((In, vib, psi))

        return psi_3j

    # computes matrix elements of spin operator <I',m_I'| I_a(n) |I,m_I>, a = x, y, z
    def spin_me(In1, In2):
        spins = h.spins
        I1 = In1[-1]
        I2 = In2[-1]
        me = np.zeros((3, len(spins), int(2*I1+1), int(2*I2+1)), dtype=np.complex128)
        reduced_me = spinMe([In1], [In2], spins, 1, 'spin')
        for im1, m1 in enumerate(np.linspace(-I1, I1, int(2*I1+1))):
            for im2, m2 in enumerate(np.linspace(-I2, I2, int(2*I2+1))):
                pow = I1 - m1
                ipow = int(pow)
                assert (pow - ipow < 1e-6), f"I1 - m1 = {I1} - {m1} is non-integer"
                threej = py3nj.wigner3j([int(I1*2)]*3, [2]*3, [int(I2*2)]*3,
                                        [-int(m1*2)]*3, [-2, 0, 2], [int(m2*2)]*3)
                me[:, :, im1, im2] = (-1)**ipow * np.dot(Ux, threej)[:, None] * reduced_me[:, 0, 0]
        return me # (a, ispin, m_I, m_J), where a=0..2 (x, y, z), ispin=0..no.spins


    # compute contractions of spin-rovibrational basis functions with three-j symbols
    # for selected hyperfine states
    psi_3j = []
    for F, vec_f in h.eigvec.items():
        for sym, vec_sym in vec_f.items():
            for istate in range(vec_sym.shape[-1]):
                if state_filter(F=F, sym=sym, ind=istate):
                    ind = np.where(abs(vec_sym[:, istate])**2 >= c2_thresh)
                    coefs = vec_sym[ind, istate]
                    q = [h.quantaRovib[F][sym][i] for i in ind[0]]
                    quanta = [(In, J, rv_sym, rv_ind) for (In, J, rv_sym, k, rv_ind) in q]
                    psi = psi_threej(F, quanta, coefs)
                    psi_3j.append((psi, F, sym, istate))


    mydict = lambda: defaultdict(mydict)
    dens = mydict()

    for i, (psi1, F1, sym1, istate1) in enumerate(psi_3j):
        for j, (psi2, F2, sym2, istate2) in enumerate(psi_3j):
            if diag and i != j:
                continue
            d = np.zeros((3, len(h.spins), len(grid)), dtype=np.complex128)
            for (I1n, vib1, psi1_3j) in psi1:
                for (I2n, vib2, psi2_3j) in psi2:
                    vboth = list(set(vib1) & set(vib2))
                    vind1 = [vib1.index(v) for v in vboth]
                    vind2 = [vib2.index(v) for v in vboth]
                    me = spin_me(I1n, I2n)
                    d += np.einsum('vig,vjg,anij->ang',
                                   psi1_3j[vind1, :, :],
                                   psi2_3j[vind2, :, :],
                                   me,
                                   optimize='optimal')
            dens[(F1, F2)][(sym1, sym2)][(istate1, istate2)] = d

    return dens

