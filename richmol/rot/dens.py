import numpy as np
import py3nj
from richmol.rot import Solution
from richmol.rot.wig import jy_eig
from scipy.sparse import coo_matrix
from collections import defaultdict


def init_density(basis: Solution, thresh=1e-14):

    def cg_dvec(j1, k1, j2, k2, conj, thresh=1e-14):

        k12 = np.array([(_k1, _k2) for _k1 in k1 for _k2 in k2], dtype=int)
        dk12 = k12[:, 0] - k12[:, 1]
        nk12 = len(k12)
        j_list = np.array([j for j in range(abs(j1-j2), j1+j2+1)], dtype=int)
        print(len(j_list))

        res = {}
        for j in j_list:
            cg = py3nj.clebsch_gordan(
                [j1*2]*nk12, [j2*2]*nk12, [j*2]*nk12,
                k12[:,0]*2, -k12[:,1]*2, dk12*2,
                ignore_invalid=True
            )
            ind = np.where(np.abs(cg) > thresh)
            cg = cg[ind]
            ind_dk = dk12 + j
            ind_dk = ind_dk[ind]
            d_vec = jy_eig(j, trid=True)
            if conj:
                d_vec = np.conj(d_vec)
            mat = d_vec[ind_dk, :].T * cg # shape = (mu=-j..j, {k',k})
            k = k12[ind]

            ind = np.where(np.abs(mat) > thresh)
            irrep0 = sum([2*(_j+1)-1 for _j in range(j)])
            irrep = [irrep0 + mu + j for mu in range(-j, j+1)]

            _res = {irr: coo_matrix((m, (k[:,0] + j1, k[:,1] + j2)),
                                    shape=(len(k1), len(k2))).tocsr()
                for irr, m in zip(irrep, mat)}
            res = {**res, **_res}

        return res


    mydict = lambda: defaultdict(mydict)

    kmat = mydict()
    for j1 in basis.keys():
        for sym1 in basis[j1].keys():
            k1 = list(set([k for (_j, k) in basis[j1][sym1].k.table['prim']]))

            for j2 in basis.keys():
                for sym2 in basis[j2].keys():
                    print("kmat", j1, j2, sym1, sym2)
                    k2 = list(set([k for (_j, k) in basis[j2][sym2].k.table['prim']]))

                    kmat[(j1, j2)][(sym1, sym2)] = cg_dvec(j1, k1, j2, k2, conj=False, thresh=thresh)

    mmat = mydict()
    for j1 in basis.keys():
        for sym1 in basis[j1].keys():
            m1 = list(set([m for (_j, m) in basis[j1][sym1].m.table['prim']]))
            fac1 = np.sqrt((2*j1+1)/(8*np.pi**2))

            for j2 in basis.keys():
                for sym2 in basis[j2].keys():
                    print("mmat", j1, j2, sym1, sym2)
                    m2 = list(set([m for (_j, m) in basis[j2][sym2].m.table['prim']]))
                    fac2 = np.sqrt((2*j2+1)/(8*np.pi**2))

                    fac = fac1 * fac2
                    mat = cg_dvec(j1, m1, j2, m2, conj=True, thresh=thresh)
                    mmat[(j1, j2)][(sym1, sym2)] = {k: v * fac for k, v in mat.items()}

