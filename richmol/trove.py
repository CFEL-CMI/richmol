from richmol.field import CarTens
from richmol.rot.wig import jy_eig
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict
import re
import os


class CarTensTrove(CarTens):
    """To initialise Cartesian tensor operator from TROVE output files.

    TROVE format includes 'states file', containing information about
    rovibrational energies and assignments, and 'matrix elements files',
    containing rovibrational matrix elements of an operator. Matrix
    elements for different pairs of angular momentum J (or F) quanta
    are stored in different files.

    This is a subclass of :py:class:`richmol.field.CarTens` class.

    Args:
        filename : str
            Name of states file.
        matelem : str
            In TROVE output format, matrix elements of Cartesian tensors for
            different values of bra and ket J (or F) quanta are stored in
            separate files. The argument `matelem` provides a template for
            generating the names of these files. For example, for `matelem` =
            'me<j1>_<j2>', the following files will be searched: 'me0_0',
            'me0_1", "me1_0", 'me2_0', 'me2_1', and so on, i.e. <j1> and <j2>
            will be replaced by integer values of bra and ket J quanta,
            respectively. For half-integer values of the F quantum number,
            replace <j1> and <j2> in the template by <f1> and <f2>, these will
            then be substituted by the floating point values of bra and ket F
            quanta rounded to the first decimal.
        coefs : str
            Name of file containing rovibrational coefficients, used to
            for plotting rotational densities and wavepackets.
    """
    def __init__(self, filename=None, matelem=None, coefs=None, **kwargs):

        self.read_states(filename, coef_file=coefs, **kwargs)
        if matelem is not None:
            self.read_trans(matelem, **kwargs)

        # apply state filters, note that state selection was already done
        #   before, in read_states, here we call it again to delete some small
        #   elements in mmat and kmat (thresh parameter) as well as reset state
        #   indexes in ind_m1, ind_m2, ind_k1, and ind_k2
        self.filter(**kwargs)


    def read_states(self, filename, coef_file=None, **kwargs):
        """ Reads states file (rovibrational state information)
        and optionally file with rovibrational coefficients, which is used
        for estimations of rotational density function (mainly for plotting)

        Args:
            filename : str
                Name of states file.
            coef_file : str
                Name of coefficients file.
                This file is used to estimate rotational probability
                density functions.

        Kwargs:
            bra : function(**kw)
                State filter for bra basis sets (see `bra` in kwargs of
                :py:class:`CarTens`).
            ket : function(**kw)
                State filter for ket basis sets (see `ket` in kwargs of
                :py:class:`CarTens`).
            thresh : float
                Threshold for neglecting matrix elements when reading from
                file.
        """
        # read states file

        mydict = lambda: defaultdict(mydict)
        energy = mydict()
        assign = mydict()
        map_k_ind = dict()

        with open(filename, 'r') as fl:
            for line in fl:
                w = line.split()
                try:
                    J = round(float(w[0]),1)
                    id = int(w[1])
                    sym = w[2]
                    ndeg = int(w[3])
                    enr = float(w[4])
                    qstr = " ".join([w[i] for i in range(5,len(w))])
                except (IndexError, ValueError):
                    raise ValueError(
                        f"error while reading file '{filename}'"
                    ) from None

                for ideg in range(ndeg):
                    try:
                        energy[J][sym].append(enr)
                        assign[J][sym].append(qstr)
                    except Exception:
                        energy[J][sym] = [enr]
                        assign[J][sym] = [qstr]

                    # mapping between (J,id,ideg) and basis set ind in the
                    #   group of states sharing the same J and symmetry
                    map_k_ind[(J, id, ideg+1)] = (len(energy[J][sym])-1, sym)

        if len(list(energy.keys())) == 0:
            raise Exception(
                f"zero number of states in file '{filename}'"
            ) from None

        # list of m quanta for different J
        mquanta = {
            J : [ round(float(m), 1)
                  for m in np.linspace(-J, J, int(2 * J) + 1) ]
            for J in energy.keys()
        }

        # generate mapping beteween m quanta and basis set index
        map_m_ind = {
            (J, m) : ind_m
            for J in mquanta.keys() for ind_m, m in enumerate(mquanta[J])
        }

        # generate attributes

        Jlist = list(energy.keys())
        symlist = {J : [sym for sym in energy[J].keys()] for J in Jlist}
        dim_m = {
            J : {sym : len(mquanta[J]) for sym in symlist[J]} for J in Jlist
        }
        dim_k = {
            J : {sym : len(energy[J][sym]) for sym in symlist[J]}
            for J in Jlist
        }
        dim = {
            J : {sym : dim_m[J][sym] * dim_k[J][sym] for sym in symlist[J]}
            for J in Jlist
        }
        quanta_m = {
            J : {sym : [m for m in mquanta[J]] for sym in symlist[J]}
            for J in Jlist
        }
        quanta_k = {
            J : { sym : [(q,e) for q, e in zip(assign[J][sym], energy[J][sym])]
                  for sym in symlist[J] }
            for J in Jlist
        }

        self.Jlist1 = Jlist
        self.symlist1 = symlist
        self.quanta_k1 = quanta_k
        self.quanta_m1 = quanta_m
        self.dim_k1 = dim_k
        self.dim_m1 = dim_m
        self.dim1 = dim

        self.Jlist2 = Jlist
        self.symlist2 = symlist
        self.quanta_k2 = quanta_k
        self.quanta_m2 = quanta_m
        self.dim_k2 = dim_k
        self.dim_m2 = dim_m
        self.dim2 = dim

        self.cart = '0'
        self.os = [(0,0)]
        self.rank = 0

        self.mmat = {
            (J, J) : {
                (sym, sym) : {
                    0 : { '0' : csr_matrix(
                            np.eye(len(mquanta[J])), dtype=np.complex128
                           ) }
                } for sym in symlist[J]
            } for J in Jlist
        }

        self.kmat = {
            (J, J) : {
                (sym, sym) : {
                    0 : csr_matrix(
                        np.diag(energy[J][sym]), dtype=np.complex128
                    )
                } for sym in symlist[J]
            } for J in Jlist
        }

        # following attributes to be used by read_trans only
        self.dim_m = dim_m
        self.dim_k = dim_k
        self.map_k_ind = map_k_ind
        self.map_m_ind = map_m_ind
        self.store_exclude = ["dim_m", "dim_k", "map_k_ind", "map_m_ind"] # exclude attributes from stroing into h5 file

        # apply state selection filters
        self.filter(**kwargs)

        # read richmol coefficients file
        #  this data shall be used only for estimations
        #  of the rotational probability density function

        if coef_file is not None:

            # create set of unique k and v quanta
            kv = mydict()
            with open(coef_file, 'r') as fl:
                for line in fl:
                    w = line.split()
                    J = round(float(w[0]),1)
                    id = int(w[1])
                    sym = w[2]
                    ideg = int(w[3])
                    enr = float(w[4])
                    if J not in self.Jlist1:
                        continue
                    istate, sym_ = self.map_k_ind[(J, id, ideg)]
                    assert (sym == sym_), \
                            f"state symmetry {sym} in file '{coef_file}' " + \
                            f"does not match {sym_} in file '{filename}', " + \
                            f"for state (J, id, ideg) = {(J, id, ideg)}"
                    try:
                        state_ind = self.ind_k1[J][sym].index(istate)
                    except (ValueError, KeyError):
                        continue
                    enr_ = self.kmat[(J, J)][(sym, sym)][0].diagonal()[state_ind]
                    assert (abs(enr_ - enr) < 1e-12), \
                            f"state energy {enr} in file '{coef_file}' " + \
                            f"does not match {enr_} in file '{filename}', " + \
                            f"for state (J, id, ideg) = {(J, id, ideg)}"
                    nelem = int(w[5])
                    if len(kv[J][sym]) == 0:
                        kv[J][sym] = []
                    for ielem in range(nelem):
                        v = int(w[8+ielem*4])
                        k = int(w[9+ielem*4])
                        kv[J][sym].append((J, k, v))

            # set of unique (k, v) quanta
            kv = {J: {sym: np.array(list(set(kv[J][sym]))) for sym in kv[J].keys()}
                for J in kv.keys()}
            # map (k, v) to index
            kv_ind = {J: {sym: {(J, k, v): i for i, (J, k, v) in enumerate(kv[J][sym])}
                for sym in kv[J].keys()}
                for J in kv.keys()}

            # read coefficients
            coo_ind = mydict()
            coo_data = mydict()
            stat = mydict()
            with open(coef_file, 'r') as fl:
                for line in fl:
                    w = line.split()
                    J = round(float(w[0]),1)
                    id = int(w[1])
                    sym = w[2]
                    ideg = int(w[3])
                    enr = float(w[4])
                    if J not in self.Jlist1:
                        continue
                    istate, sym_ = self.map_k_ind[(J, id, ideg)]
                    try:
                        state_ind = self.ind_k1[J][sym].index(istate)
                    except (ValueError, KeyError):
                        continue
                    nelem = int(w[5])

                    kv_vec = [(int(w[9+i*4]), int(w[8+i*4])) for i in range(nelem)]
                    c_vec = [float(w[6+i*4])*{0:1, 1:1j}[int(w[7+i*4])] for i in range(nelem)]

                    if len(coo_ind[J][sym]) == 0:
                        coo_ind[J][sym] = []
                        coo_data[J][sym] = []
                    coo_ind[J][sym] += [(kv_ind[J][sym][(J, k, v)], state_ind) for (k, v) in kv_vec]
                    coo_data[J][sym] += c_vec

                    # state assignment
                    nassign_max = 4
                    nassign = min([nassign_max, len(c_vec)])
                    if len(stat[J][sym]) == 0:
                        stat[J][sym] = np.zeros((self.dim_k1[J][sym], nassign_max*4), dtype='U10')
                    c2_vec = np.abs(c_vec)**2
                    ind = np.argpartition(c2_vec, -nassign)[-nassign:]
                    ind = ind[c2_vec[ind].argsort()[::-1]]
                    k, v = np.array(kv_vec)[ind].T
                    c2 = c2_vec[ind]
                    st = [[J, k_, v_, "%6.4f"%c_] for k_, v_, c_ in zip(k, v, c2)]
                    stat[J][sym][state_ind, :nassign*4] = [el for elem in st for el in elem]

            coef_data = mydict()
            for J in coo_data.keys():
                for sym in coo_data[J].keys():
                    if len(coo_data[J][sym]) == 0:
                        del coo_data[J][sym]
                        del coo_ind[J][sym]
                        del stat[J][sym]
                    coef_data[J][sym] = coo_matrix((coo_data[J][sym], np.array(coo_ind[J][sym]).T),
                                                    shape=(len(kv[J][sym]), self.dim_k1[J][sym])).tocsr()

            self.symtop_basis = {
                J: {
                    sym: {
                        'm': {
                            'prim': np.array([(J, m) for m in mquanta[J]]),
                            'stat': np.array([(J, m) for m in self.quanta_m1[J][sym]], dtype='U10'),
                            'c': csr_matrix(np.array([[1.0 if m1 == m2 else 0.0
                                                         for m2 in self.quanta_m1[J][sym]]
                                                         for m1 in mquanta[J]])),
                        },
                        'k': {
                            'prim': np.array(kv[J][sym]),
                            'stat': stat[J][sym],
                            'c': coef_data[J][sym],
                        }
                    }
                    for sym in coef_data[J].keys()
                } for J in coef_data.keys()
            }


    def read_trans(self, filename, thresh=None, **kwargs):
        """ Reads matrix elements of Cartesian tensor

        NOTE: call :py:func:`read_states` before :py:func:`read_trans` to load
        rovibrational states information stored in states file.

        Args:
            filename : str
                Matrix elements of Cartesian tensors for
                different values of bra and ket J (or F) quanta are stored in
                separate files. The parameter `filename` provides a template
                for generating the names of these files. For example, for
                `filename` = 'me<j1>_<j2>', the following files will be searched:
                'me0_0', 'me0_1', 'me1_0', 'me2_0', 'me2_1', and so on, i.e.
                <j1> and <j2> will be replaced by integer values of bra and ket
                J quanta, respectively. For half-integer values of the F
                quantum number, replace <j1> and <j2> in the template by <f1>
                and <f2>, these will then be substituted by the floating point
                values of bra and ket F quanta rounded to the first decimal.
            thresh : float
                Threshold for neglecting matrix elements when reading from
                file.

        Kwargs:
            bra : function(**kw)
                State filter for bra basis sets (see `bra` in kwargs of
                :py:class:`CarTens`).
            ket : function(**kw)
                State filter for ket basis sets (see `ket` in kwargs of
                :py:class:`CarTens`).
        """
        mydict = lambda: defaultdict(mydict)

        # tensor irreducible representation indices irreps[(ncart, nirrep)]
        irreps = {
            (3,1) : [(1,-1), (1,0), (1,1)],                               # rank-1 tensor
            (9,1) : [(2,-2), (2,-1), (2,0), (2,1), (2,2)],                # traceless and symmetric rank-2 tensor
            (9,2) : [(0,0), (2,-2), (2,-1), (2,0), (2,1), (2,2)],         # symmetric rank-2 tensor
            (6,2) : [(0,0), (2,-2), (2,-1), (2,0), (2,1), (2,2)],         # symmetric rank-2 tensor (to read old-style files)
            (9,3) : [ (0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0),
                      (2,1), (2,2) ]                                      # non-symmetric rank-2 tensor
        }

        # tensor ranks ranks[ncart]
        ranks = {3 : 1, 9 : 2, 6 : 2} # "6 : 2" to read old-style files

        self.cart = []
        self.mmat = mydict()
        self.kmat = mydict()

        tens_nirrep = None
        tens_ncart = None

        for J1_ in self.Jlist1:
            for J2_ in self.Jlist2:

                J1, J2 = (J1_, J2_)
                transp = False

                F1_str = str(round(J1,1))
                F2_str = str(round(J2,1))
                J1_str = str(int(round(J1,0)))
                J2_str = str(int(round(J2,0)))

                fname = re.sub(r"\<f1\>", F1_str, filename)
                fname = re.sub(r"\<f2\>", F2_str, fname)
                fname = re.sub(r"\<j1\>", J1_str, fname)
                fname = re.sub(r"\<j2\>", J2_str, fname)

                if not os.path.exists(fname):
                    J1, J2 = (J2_, J1_)
                    transp = True
                    fname = re.sub(r"\<f1\>", F2_str, filename)
                    fname = re.sub(r"\<f2\>", F1_str, fname)
                    fname = re.sub(r"\<j1\>", J2_str, fname)
                    fname = re.sub(r"\<j2\>", J1_str, fname)
                    if not os.path.exists(fname):
                        continue

                # read data from file

                mrow = mydict()
                mcol = mydict()
                mdata = mydict()
                kcol = mydict()
                krow = mydict()
                kdata = mydict()

                with open(fname, "r") as fl:

                    iline = 0
                    eof = False
                    read_m = False
                    read_k = False

                    for line in fl:
                        strline = line.rstrip('\n')

                        if iline == 0:
                            if strline != "Start richmol format":
                                raise Exception(
                                    f"file '{fname}' has bogus header " \
                                        + f"'{strline}'"
                                )
                            iline += 1
                            continue

                        if strline == "End richmol format":
                            eof = True
                            break

                        if iline == 1:
                            w = strline.split()
                            try:
                                tens_name = w[0]
                                nirrep = int(w[1])
                                ncart = int(w[2])
                            except (IndexError, ValueError):
                                raise ValueError(
                                    f"error while reading file " \
                                        + f"'{fname}', line = {iline}"
                                ) from None
                            if tens_nirrep is not None \
                                and tens_nirrep != nirrep:
                                raise ValueError(
                                    f"`nirrep` = '{nirrep}' read from file " \
                                        + f"'{fname}' is different from the " \
                                        + f"value {tens_nirrep} read from " \
                                        + f"previous files"
                                ) from None
                            if tens_ncart is not None and tens_ncart != ncart:
                                raise ValueError(
                                    f"`ncart` = '{ncart}' read from file " \
                                        + f"'{fname}' is different from the " \
                                        + f"value '{tens_ncart}' read from " \
                                        + f"previous files"
                                ) from None
                            tens_nirrep = nirrep
                            tens_ncart = ncart
                            try:
                                self.os = irreps[(ncart, nirrep)]
                                irreps_list = sorted(list(
                                    set(omega for omega, sigma in self.os)
                                ))
                            except KeyError:
                                raise ValueError(
                                    f"can't infer Cartesian tensor irreps " \
                                        + f"from the number of Cartesian " \
                                        + f"components '{ncart}' and number " \
                                        + f"of irreps = '{nirrep}'"
                                ) from None
                            try:
                                self.rank = ranks[ncart]
                            except KeyError:
                                raise ValueError(
                                    f"can't infer rank of Cartesian tensor " \
                                        + f"from the number of Cartesian " \
                                        + f"components = '{ncart}'"
                                ) from None
                            iline += 1
                            continue

                        if strline == "M-tensor":
                            read_m = True
                            read_k = False
                            iline += 1
                            continue

                        if "K-tensor" in strline:
                            w = strline.split()
                            if len(w) > 1:
                                icmplx = int(w[1]) + 1
                                cmplx_fac_k = (1j, 1)[icmplx]
                                old_format = False
                            else:
                                cmplx_fac_k = 1
                                print("warning, reading files from older TROVE format " +\
                                      "where K-tensor and M-tensor are not individually " +\
                                      "hermitian, only their product is Hermitian")
                                old_format = True
                            read_m = False
                            read_k = True
                            iline += 1
                            continue

                        if read_m is True and strline.split()[0] == "alpha":
                            w = strline.split()
                            try:
                                icmplx = int(w[2])
                                cart = w[3].lower()
                            except (IndexError, ValueError):
                                raise ValueError(
                                    f"error while reading file " \
                                        + f"'{fname}', line = '{iline}'"
                                ) from None
                            self.cart = list(set(self.cart + [cart]))
                            cmplx_fac_m = (1j, 1)[icmplx + 1]
                            iline += 1
                            continue

                        if read_m is True:
                            w = strline.split()
                            try:
                                m1 = round(float(w[0]), 1)
                                m2 = round(float(w[1]), 1)
                                mval = [ float(val) * cmplx_fac_m
                                         for val in w[2:] ]
                            except (IndexError, ValueError):
                                raise ValueError(
                                    f"error while reading file " \
                                        + f"'{fname}', line = '{iline}'"
                                ) from None
                            im1 = self.map_m_ind[(J1, m1)]
                            im2 = self.map_m_ind[(J2, m2)]
                            for i,irrep in enumerate(irreps_list):
                                if thresh is not None \
                                    and abs(mval[i]) < thresh:
                                    continue
                                try:
                                    mrow[irrep][cart].append(im1)
                                    mcol[irrep][cart].append(im2)
                                    mdata[irrep][cart].append(mval[i])
                                except Exception:
                                    mrow[irrep][cart] = [im1]
                                    mcol[irrep][cart] = [im2]
                                    mdata[irrep][cart] = [mval[i]]

                        if read_k is True:
                            w = strline.split()
                            try:
                                id1 = int(w[0])
                                id2 = int(w[1])
                                ideg1 = int(w[2])
                                ideg2 = int(w[3])
                                kval = [float(val) * cmplx_fac_k for val in w[4:]]
                            except (IndexError, ValueError):
                                raise ValueError(
                                    f"error while reading file " \
                                        + f"'{fname}', line = '{iline}'"
                                ) from None
                            istate1, sym1 = self.map_k_ind[(J1, id1, ideg1)]
                            istate2, sym2 = self.map_k_ind[(J2, id2, ideg2)]
                            sym = (sym1, sym2)
                            for i,irrep in enumerate(irreps_list):
                                if thresh is not None \
                                    and abs(kval[i]) < thresh:
                                    continue
                                try:
                                    krow[sym][irrep].append(istate1)
                                    kcol[sym][irrep].append(istate2)
                                    kdata[sym][irrep].append(kval[i])
                                except Exception:
                                    krow[sym][irrep] = [istate1]
                                    kcol[sym][irrep] = [istate2]
                                    kdata[sym][irrep] = [kval[i]]

                        iline +=1

                    if eof is False:
                        raise Exception(
                            f"'{fname}' has bogus footer '{strline}'"
                        )

                # add data to M and K tensors

                for sympair in kdata.keys():

                    sym1, sym2 = sympair
                    if transp is True:
                        sym1_, sym2_ = (sym2, sym1)
                    else:
                        sym1_, sym2_ = (sym1, sym2)

                    # indices of pre-filtered states (generated by read_states)
                    try:
                        ik1 = self.ind_k1[J1_][sym1_]
                        ik2 = self.ind_k2[J2_][sym2_]
                        im1 = self.ind_m1[J1_][sym1_]
                        im2 = self.ind_m2[J2_][sym2_]
                    except KeyError:
                        continue

                    # original dimensions of matrices stored in file
                    mshape = (self.dim_m[J1_][sym1_], self.dim_m[J2_][sym2_])
                    kshape = (self.dim_k[J1_][sym1_], self.dim_k[J2_][sym2_])

                    # in the old-format, J1 / F1 and J2 / F2 denote ket and bra
                    #   states, respectively, while here J1 / F1 and J2 / F2
                    #   denote the opposite, bra and ket states, to account for
                    #   this, we need to do additional complex conjugate which
                    #   cancels out conjugation, when `transp` = True, and adds
                    #   it, when `transp` = False

                    if transp is True:

                        mmat = {
                            irrep : {
                                cart : csr_matrix(
                                    ( mdata[irrep][cart],
                                        (mcol[irrep][cart], mrow[irrep][cart]) ),
                                    shape=mshape
                                )[im1, :].tocsc()[:, im2].tocsr()
                                for cart in mdata[irrep].keys() }
                            for irrep in mdata.keys()
                        }

                        kmat = {
                            irrep : csr_matrix(
                                ( kdata[sympair][irrep],
                                    (kcol[sympair][irrep], krow[sympair][irrep])),
                                shape=kshape)[ik1, :].tocsc()[:, ik2].tocsr()
                            for irrep in kdata[sympair].keys()
                        }

                    else:

                        mmat = {
                            irrep : {
                                cart : csr_matrix(
                                    ( np.conj(mdata[irrep][cart]),
                                        (mrow[irrep][cart], mcol[irrep][cart]) ),
                                    shape=mshape)[im1, :].tocsc()[:, im2].tocsr()
                                for cart in mdata[irrep].keys() }
                            for irrep in mdata.keys()
                        }

                        kmat = {
                            irrep : csr_matrix(
                                ( np.conj(kdata[sympair][irrep]),
                                    (krow[sympair][irrep], kcol[sympair][irrep]) ),
                                shape=kshape)[ik1, :].tocsc()[:, ik2].tocsr()
                            for irrep in kdata[sympair].keys()
                        }

                    self.mmat[(J1_, J2_)][(sym1_, sym2_)] = mmat
                    self.kmat[(J1_, J2_)][(sym1_, sym2_)] = kmat

        # delete unnecessary attributes
        del self.map_k_ind
        del self.map_m_ind
        del self.dim_m
        del self.dim_k


    def class_name(self):
        """Generates string containing name of the parent class"""
        base = list(self.__class__.__bases__)[0]
        return base.__module__ + "." + base.__name__
