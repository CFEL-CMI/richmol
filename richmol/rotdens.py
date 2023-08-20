from richmol.field import CarTens
from richmol.rot.wig import jy_eig
import numpy as np
import spherical
import quaternionic
from collections import defaultdict
from typing import Callable, Union, Any, Dict


def dens_grid(h: CarTens,
              alpha: np.ndarray,
              beta: np.ndarray,
              gamma: np.ndarray,
              diag_only: bool = False,
              form: str = 'block') -> Union[Dict[Any, Any], np.ndarray]:
    """
    Computes rotational densities |Ψ*Ψ'| on a grid of Euler angles (α, β, γ).

    Args:
        h (CarTens): Cartesian tensor representing rotational or rovibrational
            field-free solutions.
        alpha (np.ndarray): Values of the Euler α angle in radians.
        beta (np.ndarray): Values of the Euler β angle in radians.
        gamma (np.ndarray): Values of the Euler γ angle in radians.
        form (str, optional): Determines the format of the output. 
            - 'block': Returns a dictionary with densities grouped by J and symmetry.
            - 'full': Returns densities concatenated across different J and symmetry.
              Default is 'block'.
        diag_only (bool, optional): If set to True, only diagonal densities |Ψ*Ψ|
            are calculated. If False, all diagonal and non-diagonal elements |Ψ*Ψ'|
            are calculated.
            Default is False.

    Returns:
        Union[Dict[Any, Any], np.ndarray]: Densitties on the grid of Euler angles. 
        If `form` is 'block', it returns a dictionary. For 'full', it returns a 5D array
        if `diag_only` is False, and a 4D array if `diag_only` is True.
        See examples below.

        Example for 'block':

        .. code-block:: python

            dens = dens_grid(h, alpha, beta, gamma, form='block')
            assign1, assign2 = h.assign(form='block') # bra and ket state assignments
            for (J1, J2), dens_J in dens.items():
                for (sym1, sym2), dens_sym in dens_J.items():
                    assgn1 = assign1[J1][sym1]
                    assgn2 = assign2[J2][sym2]
                    for istate in range(dens_sym.shape[-2]):
                        for jstate in range(dens_sym.shape[-1]):
                            print(J1, sym1, istate, assgn1['m'][istate], assgn1['k'][istate], "  |  ",
                                  J2, sym2, jstate, assgn2['m'][jstate], assgn2['k'][jstate], "   ",
                                  dens_sym[ia, ib, ig, istate, jstate])
                                  # where `ia`, `ib`, and `ig` are indices of Euler angles α, β, γ

            # for diagonal-only contributions

            dens = dens_grid(h, alpha, beta, gamma, form='block', diag_only=True)
            assign1, assign2 = h.assign(form='block')
            for (J1, J2), dens_J in dens.items():
                for (sym1, sym2), dens_sym in dens_J.items():
                    assgn1 = assign1[J1][sym1]
                    assgn2 = assign2[J2][sym2]
                    for istate in range(dens_sym.shape[-1]):
                        print(J1, sym1, istate, assgn1['m'][istate], assgn1['k'][istate], "  |  ",
                              J2, sym2, istate, assgn2['m'][istate], assgn2['k'][istate], "   ",
                              dens_sym[ia, ib, ig, istate])
                              # where `ia`, `ib`, and `ig` are indices of Euler angles α, β, γ

        Example for 'full':

        .. code-block:: python

            assign, _ = h.assign(form='full')
            J, sym, m, k = [assign[key] for key in ('J', 'sym', 'm', 'k')]

            dens = dens_grid(h, alpha, beta, gamma, form='full')
            for istate in range(dens.shape[-2]):
                for jstate in range(dens.shape[-1]):
                    print(istate, J[istate], sym[istate], m[istate], k[istate], "  |  ",
                          jstate, J[jstate], sym[jstate], m[jstate], k[jstate], "   ",
                          dens[ia, ib, ig, istate, jstate])
                          # where `ia`, `ib`, and `ig` are indices of Euler angles α, β, γ

            # for diagonal-only contributions

            dens = dens_grid(h, alpha, beta, gamma, form='full', diag_only=True)
            for istate in range(dens.shape[-1]):
                print(istate, J[istate], sym[istate], m[istate], k[istate],
                      dens[ia, ib, ig, istate])
                      # where `ia`, `ib`, and `ig` are indices of Euler angles α, β, γ
    """

    psi = psi_grid(h, alpha, beta, gamma, form='block', sum_over_vib=False)

    vol_elem = np.sin(beta)

    dens = defaultdict(dict)

    for J1, psi_J1 in psi.items():
        for sym1, (psi_sym1, v1) in psi_J1.items():
            for J2, psi_J2 in psi.items():
                for sym2, (psi_sym2, v2) in psi_J2.items():

                    if diag_only and (J1 != J2 or sym1 != sym2):
                        continue

                    vboth = list(set(v2) & set(v1))
                    vind1 = [v1.index(v) for v in vboth]
                    vind2 = [v2.index(v) for v in vboth]

                    if diag_only:
                        d = np.einsum('vabgk,vabgk,b->abgk',
                                      np.conj(psi_sym1[vind1]), 
                                      psi_sym2[vind2],
                                      vol_elem)
                    else:
                        d = np.einsum('vabgk,vabgl,b->abgkl',
                                      np.conj(psi_sym1[vind1]), 
                                      psi_sym2[vind2],
                                      vol_elem)

                    dens[(J1, J2)][(sym1, sym2)] = d

    if form == 'full':
        na, nb, ng = len(alpha), len(beta), len(gamma)
        if diag_only:
            dens = np.block(
                [ dens[(J1, J1)][(sym1, sym1)]
                    if (J1, J1) in dens.keys() \
                        and (sym1, sym1) in dens[(J1, J1)].keys()
                    else np.zeros((na, nb, ng, h.dim1[J1][sym1]))
                    for J1 in h.Jlist1 for sym1 in h.symlist1[J1] ]
            )
        else:
            dens = np.block(
                [ [ dens[(J1, J2)][(sym1, sym2)]
                    if (J1, J2) in dens.keys() \
                        and (sym1, sym2) in dens[(J1, J2)].keys()
                    else np.zeros((na, nb, ng, h.dim1[J1][sym1], h.dim2[J2][sym2]))
                    for J2 in h.Jlist2 for sym2 in h.symlist2[J2] ]
                    for J1 in h.Jlist1 for sym1 in h.symlist1[J1] ]
            )

    return dens


def psi_grid(h: CarTens,
             alpha: np.ndarray,
             beta: np.ndarray,
             gamma: np.ndarray,
             form: str = 'block',
             sum_over_vib: bool = True) -> Union[Dict[Any, Any], np.ndarray]:
    """
    Computes wave functions on a grid of Euler angles (α, β, γ).

    Args:
        h (CarTens): Cartesian tensor representing rotational or rovibrational
            field-free solutions.
        alpha (np.ndarray): Values of the Euler α angle in radians.
        beta (np.ndarray): Values of the Euler β angle in radians.
        gamma (np.ndarray): Values of the Euler γ angle in radians.
        form (str, optional): Determines the format of the output. 
            - 'block': Returns a dictionary with wave functions grouped by J and symmetry.
            - 'full': Returns wave functions concatenated across different J and symmetry.
              Default is 'block'.
        sum_over_vib (bool, optional): If set to True, the returned function sums
            over basis set contributions with different vibrational quanta.
            If False, these contributions are kept separately in the leading dimension
            of the output array.
            Default is True.

    Returns:
        Union[Dict[Any, Any], np.ndarray]: Wave functions on the grid of Euler angles. 
        If `form` is 'block', it returns a dictionary. For 'full', it returns a 5D array,
        if `sum_over_vib` is False, and a 4D array, if `sum_over_vib` is True.
        See examples below.

        Example for 'block':

        .. code-block:: python

            psi = psi_grid(h, alpha, beta, gamma, form='block', sum_over_vib=False)
            assign, _ = h.assign(form='block') # state assignment
            for J, psi_J in psi.items():
                for sym, (psi_sym, vib_ind) in psi_J.items():
                    assgn = assign[J][sym]
                    for istate in range(psi_sym.shape[-1]):
                        print(J, sym, istate, assgn['m'][istate], assgn['k'][istate],
                              psi_sym[iv, ia, ib, ig, istate])
                              # where `ia`, `ib`, and `ig` are indices of Euler angles
                              # α, β, γ, and `iv` is vibrational basis index

        Example for 'full':

        .. code-block:: python

            psi = psi_grid(h, alpha, beta, gamma, form='full')
            assign, _ = h.assign(form='full') # state assignment
            J, sym, m, k = [assign[key] for key in ('J', 'sym', 'm', 'k')]
            for istate in range(psi.shape[-1]):
                print(istate, J[istate], sym[istate], m[istate], k[istate],
                      psi[ia, ib, ig, istate])
                      # where `ia`, `ib`, and `ig` are indices of Euler angles α, β, γ
    """
    if form not in ('full', 'block'):
        raise ValueError(f"Unknown value of parameter 'form' = {form}")

    if form == 'full' and sum_over_vib == False:
        raise ValueError(
            f"Input parameter 'form' = {form} cannot be used together with " + \
            f"'sum_over_vib' = {sum_over_vib}, set 'form' to 'block' (default) " + \
            f"or 'sum_over_vib' to True (default)")

    if not hasattr(h, 'symtop_basis'):
        raise AttributeError(
            f"Input parameter 'h' lacks attribute 'symtop_basis', required for " + \
            f"computing wave functions")

    bas = h.symtop_basis

    psi = defaultdict(dict)

    na, nb, ng = len(alpha), len(beta), len(gamma)

    for J, bas_J in bas.items():
        mu = np.linspace(-int(J), int(J), int(2*J)+1)
        wig = jy_eig(int(J), trid=True)
        for sym, bas_sym in bas_J.items():
            kbas = bas_sym['k']
            k, v = np.array([(int(k_), int(v_)) for (_, k_, v_) in kbas['prim']]).T

            v_list = list(set(v))
            v_ind = [np.where(v==v_)[0] for v_ in v_list]
            k_ind = [np.where(k[iv, None] == mu)[1] for iv in v_ind]

            egamma = np.exp(1j * k[None, :] * gamma[:, None])
            coef = kbas['c'].toarray()
            wig_k = np.array([np.einsum('pk,pi,gp->gki',
                                        coef[iv, :],
                                        wig[ik, :],
                                        egamma[:, iv],
                                        optimize='optimal')
                                 for iv, ik in zip(v_ind, k_ind)])

            mbas = bas_sym['m']
            m = np.array([int(m_) for (_, m_) in mbas['prim']])
            m_ind = np.where(m[:, None] == mu)[1]

            ealpha = np.exp(1j * m[None, :] * alpha[:, None])
            coef = mbas['c'].toarray()
            wig_m = np.einsum('pm,pi,ap->ami',
                              coef,
                              np.conj(wig)[m_ind, :],
                              ealpha,
                              optimize='optimal')

            ebeta = np.exp(-1j * mu[None, :] * beta[:, None])
            res = np.einsum('ami,bi,vgki->vabgmk', wig_m, ebeta, wig_k, optimize='optimal')

            if sum_over_vib:
                psi[J][sym] = np.sum(res, axis=0).reshape(na, nb, ng, -1)
            else:
                psi[J][sym] = (res.reshape(len(v_list), na, nb, ng, -1), v_list)

    # only if `sum_over_vib` == True
    if form == 'full':
        psi = np.block(
            [ psi[J][sym]
                if J in psi.keys()
                    and sym in psi[J].keys()
                else np.zeros((na, nb, ng, -1))
                for J in h.Jlist1 for sym in h.symlist1[J]
            ]
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

