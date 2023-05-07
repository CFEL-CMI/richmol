import numpy as np
import spherical
import quaternionic
from collections import defaultdict
from typing import Callable


def _stateEulerGrid_basis(h, grid, m_val=None, state_filter: Callable = lambda **kw: True):
    """Computes wave functions on a 3D grid of Euler angles.
    Uses `h.basis[J][sym]` dictionary containing as values rotational solutions
    `richmol.rot.basis.PsiTableMK`, with keys representing different
    J quanta and symmetries.
    """
    if not hasattr(h, 'basis'):
        raise AttributeError(f"Parameter 'h' has no attribute 'basis'") from None

    bas = h.basis

    # evaluate set of spherical-top functions on grid

    state_list = [(int(j), sym, istate)
                    for j in bas.keys()
                    for sym in bas[j].keys()
                    for istate in range(bas[j][sym].k.table['c'].shape[-1])
                    if state_filter(J=j, sym=sym, ind=istate)]

    j_sym_list = list(set([(j, sym) for (j, sym, _) in state_list]))
    j_list = list(set([j for (j, _, _) in state_list]))

    k_list = list(set([ int(k) for (j, sym) in j_sym_list
                               for (_, k) in bas[j][sym].k.table['prim'] ]))
    if m_val is None:
        mlist = [m for m in range(-max(j_list), max(j_list)+1)]
    else:
        mlist = [m_val]

    wigner = spherical.Wigner(max(j_list))
    R = quaternionic.array.from_euler_angles(grid)
    wigD = wigner.D(R)
    jkm = {j : np.array([[np.sqrt((2*j+1) / (8*np.pi**2)) \
                          * np.conj(wigD[:, wigner.Dindex(j, m, k)])
                          for m in mlist] for k in k_list]) for j in j_list}

    # evaluate states on grid

    mydict = lambda: defaultdict(mydict)
    psi = mydict()

    for (j, sym, istate) in state_list:

        k_val = bas[j][sym].k.table['prim'].T[1]
        coefs = bas[j][sym].k.table['c']

        k_ind = [k_list.index(k) for k in k_val]
        sym_top = jkm[j][k_ind, :, :]

        if len(psi[j][sym]) == 0:
            psi[j][sym] = []
        c = coefs[:, istate]
        func = np.sum(sym_top[:, :, :] * c[:, None, None], axis=0)
        psi[j][sym].append(
            (istate, [0], np.array([func]))
        )  # shape = (v, m=-Jmax:Jmax, ipoint=0:npoints)

    return psi


def _stateEulerGrid_rotdens(h, grid, m_val=None, state_filter: Callable = lambda **kw: True):
    """Computes wave functions on a 3D grid of Euler angles.
    Uses `h.rotdens[j][sym]` and `h.rotdens_kv[j][sym] dictionaries containing
    as values state eigenvector coefficients and sets of (k, v) quanta
    (k - rotational and v - vibrational), respectively,
    with keys representing different J quanta and symmetries.
    """
    if not hasattr(h, 'rotdens'):
        raise AttributeError(f"Parameter 'h' has no attribute 'rotdens'") from None

    # evaluate set of spherical-top functions on grid
    # for all k quanta appering in h.rotdens_kv[j][sym][(k, v)]

    state_list = [(int(j), sym, istate)
                    for j in h.rotdens.keys()
                    for sym in h.rotdens_kv[j].keys()
                    for istate in range(h.rotdens[j][sym].shape[-1])
                    if state_filter(J=j, sym=sym, ind=istate)]

    j_sym_list = list(set([(j, sym) for (j, sym, _) in state_list]))
    j_list = list(set([j for (j, _, _) in state_list]))

    k_list = list(set([ int(k) for (j, sym) in j_sym_list
                               for (k, v) in h.rotdens_kv[j][sym] ]))

    if m_val is None:
        mlist = [m for m in range(-max(j_list), max(j_list)+1)]
    else:
        mlist = [m_val]

    wigner = spherical.Wigner(max(j_list))
    R = quaternionic.array.from_euler_angles(grid)
    wigD = wigner.D(R)
    jkm = {j : np.array([[np.sqrt((2*j+1) / (8*np.pi**2)) \
                          * np.conj(wigD[:, wigner.Dindex(j, m, k)])
                          for m in mlist] for k in k_list]) for j in j_list}

    # evaluate states on grid

    mydict = lambda: defaultdict(mydict)
    psi = mydict()

    for (j, sym, istate) in state_list:

        kv = h.rotdens_kv[j][sym]
        coefs = h.rotdens[j][sym]

        k_ind = [k_list.index(k) for (k, v) in kv]
        sym_top = jkm[j][k_ind, :, :]

        vlist = list(set([v for (k, v) in kv]))
        k_, v_ = np.array(kv).T
        v_ind = [np.where(v_==v) for v in vlist]

        if len(psi[j][sym]) == 0:
            psi[j][sym] = []
        c = np.array(coefs[:, istate].todense())[:, 0]
        func = sym_top[:, :, :] * c[:, None, None]
        psi[j][sym].append(
            (istate, vlist, np.array([np.sum(func[i], axis=0) for i in v_ind]))
        )  # shape = (v, m=-Jmax:Jmax, ipoint=0:npoints)

    return psi


def densityEulerGrid(h, grid, m_val, diag=False, state_filter=lambda **kw: True):
    """Computes vibrationally-averaged probability density matrix
    on a 3D grid of Euler angles

    Args:
        h : :py:class:`richmol.field.CarTens`
            Rotational or rovibrational Hamiltonian
        grid : array (:, 3)
            Grid of Euler angle values (in radians), phi, theta, chi = grid(i, :)
        m_val : int
            Value of fixed 'm' quantum number
        diag : bool
            If True, only diagonal elements of probability density matrix will
            be computed
        state_filter : 
            State filter function to select a desired subspace of states spanned
            by the Hamiltonian `h`, see, for example, `bra` and `ket` parameters
            of :py:class:`richmol.field.CarTens`

    Returns:
        dens : nested dict
            Probability density matrix for different pairs of bra and ket J
            quanta and symmetries
            Example:

            .. code-block:: python

                for (j1, j2), dens_j in dens.items():
                    for (sym1, sym2), dens_sym in dens_j.items():
                        for (i, j), den in dens_sym.items():
                            print(f"density for state pair {(j1, sym1, i)}, {(j2, sym2, j)}")
                            for (phi, theta, chi), d in zip(grid, den):
                                print(phi, theta, chi, d)
    """
    # NOTE: in principle can run for m_val=None, i.e. m=[-J..J], it will not cause much overhead
    if hasattr(h, 'rotdens') and len(list(h.rotdens.keys())) > 0: # rotdens attribute is always initialised as empty dict when reading tensors from file, hence here check it length
        psi = _stateEulerGrid_rotdens(h, grid, m_val=m_val, state_filter=state_filter) 
    elif hasattr(h, 'basis'):
        psi = _stateEulerGrid_basis(h, grid, m_val=m_val, state_filter=state_filter)
    else:
        raise AttributeError(f"input parameter 'h' has neither 'rotdens' nor 'basis' attributes which are necessary to compute rotational density")

    # jmax = max(list(psi.keys()))
    # im = [m for m in range(-jmax, jmax+1)].index(m)  # only for m_val=None in _stateEulerGrid_...
    im = 0  # only for m_val=m_val in _stateEulerGrid_...

    mydict = lambda: defaultdict(mydict)
    dens = mydict()

    for j1, psi_j1 in psi.items():
        for j2, psi_j2 in psi.items():

            if diag and j1 != j2: continue

            for sym1, psi_sym1 in psi_j1.items():
                for sym2, psi_sym2 in psi_j2.items():

                    if diag and sym1 != sym2: continue

                    for (ind1, v1, psi1) in psi_sym1:
                        for (ind2, v2, psi2) in psi_sym2:

                            if diag and ind1 != ind2: continue

                            vboth = list(set(v2) & set(v1))
                            vind1 = [v1.index(v) for v in vboth]
                            vind2 = [v2.index(v) for v in vboth]
                            d = np.einsum('vg,vg,g->g', np.conj(psi1[vind1, im, :]),
                                          psi2[vind2, im, :], np.sin(grid[:, 1]))

                            dens[(j1, j2)][(sym1, sym2)][(ind1, ind2)] = d
    return dens

