import numpy as np
import py3nj
from richmol.rot import Solution
from richmol.rot.wig import jy_eig
from scipy.sparse import coo_matrix, kron
from collections import defaultdict
from numpy.typing import NDArray



def init_density(basis: Solution, alpha: NDArray[np.float_], beta: NDArray[np.float_],
                 gamma: NDArray[np.float_], thresh: float = 1e-12):

    def cg_dvec(j1, k1, j2, k2, conj, thresh=1e-14):

        k12 = np.array([(_k1, _k2) for _k1 in k1 for _k2 in k2], dtype=int)
        dk12 = k12[:, 0] - k12[:, 1]
        nk12 = len(k12)
        j_list = np.array([j for j in range(abs(j1-j2), j1+j2+1)], dtype=int)

        res = {}
        for j in j_list:
            mu_arr = [mu for mu in range(-j, j+1)]
            cg = py3nj.clebsch_gordan(
                [j1*2]*nk12, [j2*2]*nk12, [j*2]*nk12,
                k12[:,0]*2, -k12[:,1]*2, dk12*2,
                ignore_invalid=True
            )
            ind = np.where(np.abs(cg) > thresh)
            cg = cg[ind]
            ind_dk12 = dk12 + j
            ind_dk12 = ind_dk12[ind]
            d_vec = jy_eig(j, trid=True)
            if conj:
                d_vec = np.conj(d_vec)
            mat = d_vec[ind_dk12, :].T * cg # shape = (mu=-j..j, {k',k})
            k = k12[ind]
            _res = {(j, mu): coo_matrix((m, (k[:,0] + j1, k[:,1] + j2)),
                                        shape=(len(k1), len(k2))).tocsr()
                for mu, m in zip(mu_arr, mat)}
            res = {**res, **_res}

        return res, k12.reshape(len(k1), len(k2), -1)


    mydict = lambda: defaultdict(mydict)
    kmat = mydict()
    mmat = mydict()

    for j1 in basis.keys():
        for sym1 in basis[j1].keys():
            k1 = list(set([k for (_j, k) in basis[j1][sym1].k.table['prim']]))
            m1 = list(set([m for (_j, m) in basis[j1][sym1].m.table['prim']]))
            fac1 = np.sqrt((2*j1+1)/(8*np.pi**2))

            for j2 in basis.keys():
                for sym2 in basis[j2].keys():
                    k2 = list(set([k for (_j, k) in basis[j2][sym2].k.table['prim']]))
                    m2 = list(set([m for (_j, m) in basis[j2][sym2].m.table['prim']]))
                    fac2 = np.sqrt((2*j2+1)/(8*np.pi**2))

                    mat, k12 = cg_dvec(j1, k1, j2, k2, conj=False, thresh=thresh)
                    kmat[(j1, j2)][(sym1, sym2)] = (mat, k12)

                    fac = fac1 * fac2
                    mat, m12 = cg_dvec(j1, m1, j2, m2, conj=True, thresh=thresh)
                    mmat[(j1, j2)][(sym1, sym2)] = ({k: v * fac for k, v in mat.items()}, m12)


    dens = mydict()
    for jp in mmat.keys():
        for sp in mmat[jp].keys():
            km, k12 = kmat[jp][sp]
            mm, m12 = mmat[jp][sp]

            dk = k12[:, :, 0] - k12[:, :, 1]
            dm = m12[:, :, 0] - m12[:, :, 1]
            mu = np.array([_mu for (_j, _mu) in kmat.keys()])

            e_alpha = np.exp(-1j * dm[:, :, None] * alpha[None, None, :]) # (m', m, ipoint)
            e_gamma = np.exp(-1j * dk[:, :, None] * gamma[None, None, :]) # (k', k, ipoint)
            dim = [d1 * d2 for d1, d2 in zip(dm.shape, dk.shape)]
            e_ag = np.einsum('nmg,klg->nkmlg', e_alpha, e_gamma, optimize='optimal').reshape(*dim, -1)
            # km_quanta = np.einsum('nm,kl->nkml', m12, k12, optimize='optimal').reshape(*dim)

            mat = sum(
                kron(mm[irrep], km[irrep]).toarray()[:, :, None] \
                * np.exp(-1j * irrep[1] * beta)[None, None, :]
                for irrep in list(set(mm.keys()) & set(km.keys()))
            )
            mat = mat * e_ag
            dens[jp][sp] = mat

    return dens

