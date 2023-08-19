from richmol.field import CarTens
from richmol.rot.wig import jy_eig
import numpy as np
import spherical
import quaternionic
from collections import defaultdict
from typing import Callable, Union, Any, Dict



def psi_grid(h: CarTens,
             alpha: np.ndarray,
             beta: np.ndarray,
             gamma: np.ndarray,
             form: str = 'block') -> Union[Dict[Any, Any], np.ndarray]:
    """
    Computes wave functions on a grid of Euler angles (`alpha`, `beta`, `gamma`).

    Args:
        h (CarTens): Cartesian tensor representing rotational or rovibrational 
                     field-free solutions.
        alpha (np.ndarray): Values of the Euler alpha angle in radians.
        beta (np.ndarray): Values of the Euler beta angle in radians.
        gamma (np.ndarray): Values of the Euler gamma angle in radians.
        form (str, optional): Determines the format of the output. 
            - 'block': Returns a dictionary with wave functions grouped by J and symmetry.
            - 'full': Returns wave functions concatenated across different J and symmetry.
              Default is 'block'.

    Returns:
        Union[Dict[Any, Any], np.ndarray]: Wave functions on the grid of Euler angles. 
        If `form` is 'block', it returns a dictionary. For 'full', it returns a 2D array.

        Example for 'block':

        .. code-block:: python

            for J in psi.keys():
                for sym in psi[J].keys():
                    for istate, wf in enumerate(psi[J][sym]):
                        print(J, sym, istate, wf[ia, ib, ic])
                        # where `ia`, `ib`, and `ic` are indices of Euler angles

        Example for 'full':

        .. code-block:: python

            for i, wf in enumerate(psi):
                print(istate, wf[ia, ib, ic])
    """

    assert (form in ('full', 'block')), f"Unknown value of parameter 'form' = {form}"

    try:
        bas = h.symtop_basis
    except AttributeError:
        raise AttributeError(f"Input parameter 'h' has no attribute 'symtop_basis', " + \
                             f"which is necessary to compute wave functions") from None

    mydict = lambda: defaultdict(mydict)
    psi = mydict()

    na = len(alpha)
    nb = len(beta)
    ng = len(gamma)

    for J in bas.keys():
        mu = np.linspace(-int(J), int(J), int(2*J)+1)
        wig = jy_eig(int(J), trid=True)
        for sym in bas[J].keys():
            kbas = bas[J][sym]['k']
            k = np.array([int(_k) for (_, _k) in kbas['prim']]) # ============ NOTE: might be (j, k, v) ==============
            k_ind = np.where(k[:, None] == mu)[1]
            egamma = np.exp(1j * k[:, None] * gamma[None, :])
            wig_k = np.einsum('kp,pi,pg->kig', kbas['c'].toarray().T, wig[k_ind, :],
                              egamma, optimize='optimal')

            mbas = bas[J][sym]['m']
            m = np.array([int(_m) for (_, _m) in mbas['prim']])
            m_ind = np.where(m[:, None] == mu)[1]
            ealpha = np.exp(1j * m[:, None] * alpha[None, :])
            wig_m = np.einsum('mp,pi,pa->mia', mbas['c'].toarray().T, np.conj(wig)[m_ind, :],
                              ealpha, optimize='optimal')

            ebeta = np.exp(-1j * mu[:, None] * beta[None, :])
            res = np.einsum('mia,ib,kig->mkabg', wig_m, ebeta, wig_k, optimize='optimal')
            psi[J][sym] = res.reshape(-1, len(alpha), len(beta), len(gamma))

    if form == 'full':
        psi = np.concatenate(
            [ psi[J][sym]
                if J in psi.keys()
                    and sym in psi[J].keys()
                else np.zeros((h.dim1[J][sym], na, nb, ng))
                for J in h.Jlist1 for sym in h.symlist1[J]
            ],
            axis=0
        )

    return psi


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
        m_list = [m for m in range(-max(j_list), max(j_list)+1)]
    else:
        m_list = [m_val]

    wigner = spherical.Wigner(max(j_list))
    R = quaternionic.array.from_euler_angles(grid)
    wigD = wigner.D(R)
    jkm = {j : np.array([[np.sqrt((2*j+1) / (8*np.pi**2)) \
                          * np.conj(wigD[:, wigner.Dindex(j, m, k)])
                          for m in m_list] for k in k_list]) for j in j_list}

    # evaluate states on grid

    mydict = lambda: defaultdict(mydict)
    psi = mydict()

    for (j, sym, istate) in state_list:

        k_val = bas[j][sym].k.table['prim'].T[1]
        coefs = bas[j][sym].k.table['c']

        k_ind = [k_list.index(k) for k in k_val]
        sym_top = jkm[j][k_ind, :, :]

        if len(psi[j][sym][istate]) == 0:
            psi[j][sym][istate] = []
        c = coefs[:, istate]
        func = np.sum(sym_top[:, :, :] * c[:, None, None], axis=0)
        psi[j][sym][istate] = ([0], np.array([func])) # shape = (v, m=-Jmax:Jmax, ipoint=0:npoints)

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
        m_list = [m for m in range(-max(j_list), max(j_list)+1)]
    else:
        m_list = [m_val]

    wigner = spherical.Wigner(max(j_list))
    R = quaternionic.array.from_euler_angles(grid)
    wigD = wigner.D(R)
    jkm = {j : np.array([[np.sqrt((2*j+1) / (8*np.pi**2)) \
                          * np.conj(wigD[:, wigner.Dindex(j, m, k)])
                          for m in m_list] for k in k_list]) for j in j_list}

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

        if len(psi[j][sym][istate]) == 0:
            psi[j][sym][istate] = []
        c = np.array(coefs[:, istate].todense())[:, 0]
        func = sym_top[:, :, :] * c[:, None, None]
        psi[j][sym][istate] = (vlist, np.array([np.sum(func[i], axis=0) for i in v_ind])) # shape = (v, m=-Jmax:Jmax, ipoint=0:npoints)

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

                    for ind1, (v1, psi1) in psi_sym1.items():
                        for ind2, (v2, psi2) in psi_sym2.items():

                            if diag and ind1 != ind2: continue

                            vboth = list(set(v2) & set(v1))
                            vind1 = [v1.index(v) for v in vboth]
                            vind2 = [v2.index(v) for v in vboth]
                            d = np.einsum('vg,vg,g->g', np.conj(psi1[vind1, im, :]),
                                          psi2[vind2, im, :], np.sin(grid[:, 1]))

                            dens[(j1, j2)][(sym1, sym2)][(ind1, ind2)] = d
    return dens

