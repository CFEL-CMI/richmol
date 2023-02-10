import numpy as np
import spherical
import quaternionic
from collections import defaultdict


def _stateEulerGrid(h, grid, m_val=None, state_filter=lambda **kw: True):
    """Computes wave functions on a 3D grid of Euler angles
    """
    if not hasattr(h, 'rotdens'):
        raise AttributeError(f"Parameter 'h' has no attribute 'rotdens'") from None

    # evaluate set of spherical-top functions on grid
    # for all k quanta appering in h.rotdens_kv[j][sym][(k, v)]

    jlist = [int(j) for j in h.rotdens.keys() if state_filter(J=j)]
    klist = list(set([ int(k) for j in jlist
                              for sym in h.rotdens_kv[j].keys()
                              for (k, v) in h.rotdens_kv[j][sym] ]))
    if m_val is None:
        mlist = [m for m in range(-max(jlist), max(jlist)+1)]
    else:
        mlist = [m_val]

    wigner = spherical.Wigner(max(jlist))
    R = quaternionic.array.from_euler_angles(grid)
    wigD = wigner.D(R)
    jkm = {j : np.array([[np.sqrt((2*j+1) / (8*np.pi**2)) \
                          * np.conj(wigD[:, wigner.Dindex(j, m, k)])
                          for m in mlist] for k in klist]) for j in jlist}

    # evaluate states on grid

    mydict = lambda: defaultdict(mydict)
    psi = mydict()

    for j in h.rotdens.keys():
        for sym in h.rotdens[j].keys():

            kv = h.rotdens_kv[j][sym]
            coefs = h.rotdens[j][sym]

            k_ind = [klist.index(k) for (k, v) in kv]
            sym_top = jkm[j][k_ind, :, :]

            vlist = list(set([v for (k, v) in kv]))
            k_, v_ = np.array(kv).T
            v_ind = [np.where(v_==v) for v in vlist]

            for istate in range(coefs.shape[-1]):
                if state_filter(J=j, sym=sym, ind=istate):  # apply state filters
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
    psi = _stateEulerGrid(h, grid, m_val=m_val, state_filter=state_filter) # NOTE: in principle can run for m_val=None, i.e. m=[-J..J], it will not cause much overhead

    # jmax = max(list(psi.keys()))
    # im = [m for m in range(-jmax, jmax+1)].index(m)  # only for m_val=None in _stateEulerGrid
    im = 0  # only for m_val=m_val in _stateEulerGrid

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
