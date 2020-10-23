"""Module for solving the TDSE with the Hamiltonian represented by a sum of the molecular stationary
Hamiltonian H0 and the time-dependent molecule-field interaction potential V(t).

The potential V(t) for interaction with electric field is represented by a multipole-moment expansion
V(t) = -mu_A E_A(t) - 1/2 alpha_AB E_A(t) E_B(t) -1/6 beta_ABC E_A(t) E_B(t) E_C(t) - ....,
where mu, alpha, and beta are molecular permanent dipole moment, polarisability, and first
hyperpolarizability, respectively, E(t) is the time-dependent electric field,
and A, B, C are X, Y, or Z Cartesian components in the laboratory frame.

The wavepacket is constructed as a linear combination of molecular stationary states,
i.e., eigenstates of H0, with time-dependent coefficients. The molecular energies (eigenvalues of H0)
and the matrix elements of mu, alpha, beta, etc. Cartesian tensors in the basis of molecular
stationary states are assumed to be computed with some other program and stored in separate files
using the so-called Richmol format. The energies and tensor matrix elements can be loaded using
Psi() and Etensor() classes, respectively.
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, kron
from scipy import linalg as la
import re
import sys, time
import os
import itertools
import inspect
import copy


au_to_coulmet_ = {1: 8.47835326e-30,      # to convert electric electric dipole from au to C*m
                  2: 1.6487772754e-41,    #   polarizability from au to C^2*m^2*J^-1
                  3: 3.206361449e-53,     #   hyperpolarizability from au to C^3*m^3*J^-2
                  4: 6.23538054e-65 }     #   second hyperpolarizability from au to C^4*m^4*J^-3

joule_to_cm_ = 5.03411701e+22  # to convert energy from Joule to cm^-1
lightspeed_ = 299792458e0      # in m/s


class Psi():
    """Field-free stationary basis set and wavepacket.
    """
    def __init__(self, fname, **kwargs):

        # read field-free states

        self.states, self.map_id_to_istate = read_states(fname, **kwargs)

        fmax = max(list(self.states.keys()))
        fmin = min(list(self.states.keys()))

        # generate list of M quanta

        if 'mlist' in kwargs:
            mlist = [round(m,1) for m in kwargs['mlist']]
        else:
            if 'mmax' in kwargs:
                mmax = min([kwargs['mmax'],fmax])
                if mmax<kwargs['mmax']:
                    print(f"Psi: mmax is set to {mmax} which is maximum F in states file {fname}")
            else:
                mmax = fmax
            if 'mmin' in kwargs:
                mmin = max([kwargs['mmin'],-fmax])
                if mmin>kwargs['mmin']:
                    print(f"Psi: mmin is set to {mmin} which is minus maximum F in states file {fname}")
            else:
                mmin = -fmax
            if 'dm' in kwargs:
                dm = kwargs['dm']
            else:
                dm = 1
            m1 = round(mmin,1)
            m2 = round(mmax,1)
            d = round(dm)
            assert (m2>=m1),f"mmin={mmin} > mmax={mmax}"
            assert (dm>=1),f"dm={dm}<1"
            self.mlist = [round(m,1) for m in np.linspace(m1,m2,int((m2-m1)/d)+1)]

        # generate basis: combinations of M and field-free state quanta for each F

        self.flist = list(self.states.keys())

        self.quanta_m = {f:[m for m in self.mlist if abs(m)<=f] for f in self.flist}
        self.quanta_istate = {f:[state['istate'] for state in self.states[f]] for f in self.flist}
        self.dim_m = {f:len(self.quanta_m[f]) for f in self.flist}
        self.dim_istate = {f:len(self.quanta_istate[f]) for f in self.flist}

        self.quanta = {}
        self.energy = {}
        self.dim = {}
        self.f = []
        self.m = []
        self.istate = []
        for f in self.flist:
            self.quanta[f] = []
            enr = []
            for m in self.mlist:
                if abs(m)>f:
                    continue
                for state in self.states[f]:
                    istate = state['istate']
                    self.quanta[f].append([m,istate])
                    enr.append(state['enr'])
                    self.f.append(f)
                    self.m.append(m)
                    self.istate.append(istate)
                self.energy[f] = np.array(enr, dtype=np.float64)
            self.dim[f] = len(self.quanta[f])
            assert (self.dim[f] == self.dim_m[f] * self.dim_istate[f]), \
                    f"Basis dimension = {self.dim[f]} for J = {f} is not equal to the product " \
                    +f"of dimensions of m-basis = {self.dim_m[f]} and field-free basis = " \
                    +f"{self.dim_istate[f]}"


    @property
    def j_m_id(self):
        pass


    @j_m_id.setter
    def j_m_id(self, val):
        """Use this function to define the initial wavepacket coefficients, as
        Psi().j_m_id = (j, m, id, ideg, coef), where j, m, id, and ideg identify the stationary
        basis function and coef is the desired coefficient. Call it multiple times define the
        coefficient values for multiple basis functions.
        """
        try:
            f, m, id, ideg, coef = val
        except ValueError:
            raise ValueError(f"Pass an iterable with five items, i.e., j_m_id = (j, m, id, ideg, coef)")
        ff = round(float(f),1)
        mm = round(float(m),1)
        iid = int(id)
        iideg = int(ideg)
        try:
            x = self.flist.index(ff)
        except ValueError:
            raise ValueError(f"Input quantum number J = {ff} is not spanned by the basis") from None
        try:
            istate = self.map_id_to_istate[ff][iid]+iideg-1
        except IndexError:
            raise IndexError(f"Input set of quanta (id,ideg) = ({iid},{iideg}) for J = {ff} " \
                    + f"is not spanned by the basis") from None
        try:
            ibas = self.quanta[ff].index([mm,istate])
        except ValueError:
            raise ValueError(f"Input set of quanta (m,id,ideg,istate) = ({mm},{iid},{iideg},{istate}) " \
                    + f"for J = {ff} is not spanned by the basis") from None
        try:
            x = self.coefs
        except AttributeError:
            self.coefs = {f:np.zeros(len(self.quanta[f]), dtype=np.complex128) for f in self.flist}
        self.coefs[ff][ibas] = coef



class Etensor():
    """Cartesian electric multiple-moment tensor in laboratory frame.

    Matrix elements of all tensors must be in atomic units, use 'units' keyword argument
    if stored matrix elements use different units.

    Args:
        filename (str): Template for generating the names of files containing tensor matrix elements.
                        For example, for filename="matelem_alpha_j<j1>_j<j2>.rchm" the following
                        files will be searched: matelem_alpha_j0_j0.rchm, matelem_alpha_j0_j1.rchm,
                        matelem_alpha_j0_j2.rchm and so on, where "<j1>" and "<j2>" will be replaced
                        by integer numbers running through all J quanta spanned by the basis psi.
                        j2 and j1 are treated as bra and ket state quanta, respectively.
                        For half-integer numbers (e.g., F quanta), substitute "<j1>" and "<j2>"
                        by "<f1>" and "<f2>", which will then be replaced by floating point numbers
                        rounded to the first decimal point after the comma.
        psi (Psi()): Field-free basis set.
        kwargs:
            units (str): Units of stored tensor matrix elements.
                         The following units are implemented:
                         'Debye' for dipole moment

    Attributes:
        name (str): Name of tensor.
        ncart (int): Number of tensor Cartesian components, for example, for dipole moment ncart=3
                     and for polarizability ncart=9.
        rank (int): Tensor rank, for example, for dipole moment rank=1 and for polarizability rank=2.
        nomega (int): Number of nonzero irreducible components, for example, for dipole moment
                      nomega=1 and for polarizability nomega=2.
        mcart (list): List of strings identifying tensor Cartesian components.
                      It has the following structure mcart[icart] for icart in range(ncart).
                      For example, for dipole moment tensor mcart=['x','y','z'] and for
                      polarizability mcart=['xx','xy','xz','yx','yy','yz','zx','zy','zz'].
        mtype (list): List of strings ('real' or 'imaginary') identifying which of tensor Cartesian
                      components are purely real and which are purely imaginary.
                      Its structure is mtype[icart] for icart in range(ncart).
                      For example, for dipole moment mtype=['imaginary','real','imaginary']
                      for mcart=['x','y','z'].
        M (dict): Richmol M-tensor, its structure is M[(f1,f2)][iomega][icart,im1,im2],
                  where f1 and f2 are bra and ket quantum numbers of the total angular momentum
                  (F is equivalent to J for pure rovibrations), iomega in range(nomega),
                  icart in range(ncart), and im1=f1+m1 and im2=f2+m2 for m1,m2 in
                  zip(range(-f1,f1+1),range(-f2,f2+1), although the span for m quanta
                  (=-f..f by default) can be modified, see parameters to Psi().
        MF (dict): Richmol M-tensor contracted with external electric field,
                   its structure is MF[(f1,f2)][iomega][im1,im2].
        K (dict): Richmol K-tensor, K[(f1,f2)][iomega] contains CSR-sparse matrix with row
                  and column indices corresponding to the bra and ket field-free states spanned
                  within the corresponding f1 and f2 quanta.
        prefac (scalar): Some constant prefactor, used to convert tensor to proper units,
                         or to keep the result of tensor product with a scalar.
    """
    def __init__(self, filename, psi, **kwargs):

        self.M = {}
        self.K = {}
        self.prefac = 1.0

        if "units" in kwargs:
            units = kwargs["units"]
            try:
                conv = {"DEBYE":np.float64(0.393456)}[units.upper()]
                print(f"Etensor: using factor {conv} to convert from Debye to atomic units")
            except KeyError:
                raise KeyError(f"Unknown tensor units in 'units={units}'") from None
            self.prefac *= conv

        # read Richmol matrix elements for all combinations of bra and ket F quanta spanned by basis

        for f1 in psi.flist:
            for f2 in psi.flist:

                # generate file name

                f1_str = str(round(f1,1))
                f2_str = str(round(f2,1))
                j1_str = str(int(round(f1,0)))
                j2_str = str(int(round(f2,0)))

                fname = re.sub(r"\<f1\>", f1_str, filename)
                fname = re.sub(r"\<f2\>", f2_str, fname)
                fname = re.sub(r"\<j1\>", j1_str, fname)
                fname = re.sub(r"\<j2\>", j2_str, fname)

                if not os.path.exists(fname):
                    fname_ = fname
                    fname = re.sub(r"\<f1\>", f2_str, filename)
                    fname = re.sub(r"\<f2\>", f1_str, fname)
                    fname = re.sub(r"\<j1\>", j2_str, fname)
                    fname = re.sub(r"\<j2\>", j1_str, fname)
                    if not os.path.exists(fname):
                        print(f"Etensor: skip F1/F2 = {f1}/{f2} pair, no file(s) found: " \
                                + f"{fname_} or {fname}")
                        continue

                # read file

                name, ncart, nomega, mcart, mtype, mtens, ktens, tr = \
                        read_tens(fname, f1, f2, psi.map_id_to_istate)

                try:
                    rank = {3**rank:rank for rank in range(6)}[ncart]
                except KeyError:
                    raise RuntimeError(f"Can't determine tensor rank from the number of its Cartesian " \
                            + f"components = {ncart}, check file {fname}") from None

                # need to add a check that all these values are consistent throughout different files

                if self.__dict__.get("name") is None:
                    self.name = name
                if self.__dict__.get("ncart") is None:
                    self.ncart = ncart
                if self.__dict__.get("rank") is None:
                    self.rank = rank
                if self.__dict__.get("nomega") is None:
                    self.nomega = nomega
                if self.__dict__.get("mcart") is None:
                    self.mcart = mcart
                if self.__dict__.get("mtype") is None:
                    self.mtype = mtype

                self.K[(f1,f2)] = ktens

                # keep only those elements of M-tensor that are spanned by basis,
                # in case user have selected only a reduced subspace of m quanta,
                # also convert M-tensor to a complex-valued form

                dim1 = len([m for m in psi.mlist if abs(m)<=f1])
                dim2 = len([m for m in psi.mlist if abs(m)<=f2])
                self.M[(f1,f2)] = [ np.zeros((self.ncart,dim1,dim2), dtype=np.complex128) \
                                    for iomega in range(self.nomega) ]

                vec1j = np.array([{"real":1,"imaginary":1j}[mt] for mt in mtype], dtype=np.complex128)

                for i,m1 in enumerate([m for m in psi.mlist if abs(m)<=f1]):
                    im1 = int(m1+f1)
                    for j,m2 in enumerate([m for m in psi.mlist if abs(m)<=f2]):
                        im2 = int(m2+f2)
                        for iomega in range(self.nomega):
                            self.M[(f1,f2)][iomega][:,i,j] = mtens[iomega][:,im1,im2] * vec1j

        # modify self.pref such that Etensor*field[V/m]**rank gives units of cm^-1

        try:
            self.prefac *= joule_to_cm_ * au_to_coulmet_[self.rank]
        except KeyError:
            raise NotImplementedError(f"Conversion factor from atomic units to (C*m)^rank*J^(1-rank) " \
                    + f"for tensor rank = {self.rank} is not implemented") from None


    def __mul__(self, arg):
        """
        arg:
            scalar: Multiply tensor with scalar, return a copy of self with self.prefac multiplied by arg.
            list: Contract Cartesian components of M-tensor with field, return result in a copy
                  of self with the new (or updated) attribute MF[(f1,f2)][iomega][im1,im2].
                  Here, the first three elements of arg are assumed to be X, Y, and Z components
                  of electric field, elements arg[3:] are ignored.
            Psi(): Act with tensor on a wavepacket, return result in a copy of arg with updated
                   arg.coefs. Tensor must be contracted with electric field before.
            dict: Same as for Psi(), where arg is the same as Psi().coefs attribute,
                  return result in a copy of arg.
        """
        scalar = (int, float, complex, np.int, np.int8, np.int16, np.int32, np.int64, np.float, \
                  np.float16, np.float32, np.float64, np.complex64, np.complex128)

        if isinstance(arg, scalar):

            # multiply tensor with a constant

            res = copy.deepcopy(self)
            res.prefac *= arg

        elif isinstance(arg, (np.ndarray, list)):

            # multiply tensor with field and contract over Cartesian components

            res = copy.deepcopy(self)

            try:
                fieldx = arg[0]
                fieldy = arg[1]
                fieldz = arg[2]
            except IndexError:
                arg_name = retrieve_name(arg)
                raise IndexError(f"Field '{arg_name}' must be an iterable with three items " \
                        + f"(field's X,Y,Z components)") from None

            res.MF = {fkey : [ np.zeros(res.M[fkey][iomega].shape[1:3], dtype=np.complex128) \
                               for iomega in range(res.nomega) ] for fkey in res.M.keys()}

            xyz = ["XYZ"]*res.rank
            field = [(fieldx,fieldy,fieldz)]*res.rank

            for x,f in zip(itertools.product(*xyz),itertools.product(*field)):
                icart = res.mcart.index("".join(x))
                ff = np.prod(np.array(f))
                for fkey in res.M.keys():
                    for iomega in range(res.nomega):
                        res.MF[fkey][iomega][:,:] += res.M[fkey][iomega][icart,:,:] * ff

        elif isinstance(arg, Psi):

            # apply tensor (pre-contracted with field) to a wavepacket
            # here wavepacket is represented by class Psi()

            psi = arg
            try:
                x = psi.coefs
            except AttributeError:
                arg_name = retrieve_name(arg)
                raise AttributeError(f"Wavepacket '{arg_name}' must be initialized, " \
                        + f"e.g., use {arg_name}.j_m_id = (j,m,id,ideg,coef)") from None

            try:
                x = self.MF
            except AttributeError:
                self_name = retrieve_name(self)
                raise AttributeError(f"Tensor '{self_name}' must be multiplied with field " \
                        + f"before applying it to a wavepacket") from None

            for fkey in self.MF.keys():
                f1, f2 = fkey
                try:
                    x = psi.coefs[f2]
                except KeyError:
                    arg_name = retrieve_name(arg)
                    self_name = retrieve_name(self)
                    raise KeyError(f"Wavepacket '{arg_name}' does not contain functions " \
                            + f"for J = {f2} which are present for tensor '{self_name}'") from None

            coefs = self.MKvec(self.MF, self.K, psi.coefs)
            for f in coefs.keys():
                coefs[f] *= self.prefac
            res = copy.deepcopy(psi)
            res.coefs = coefs

        elif isinstance(arg, dict):

            # apply tensor (pre-contracted with field) to a wavepacket
            # here wavepacket is represented by coefs attribute of class Psi()

            coefs0 = arg

            try:
                x = self.MF
            except AttributeError:
                self_name = retrieve_name(self)
                raise AttributeError(f"Tensor '{self_name}' must be multiplied with field " \
                        + f"before applying it to a wavepacket") from None

            for fkey in self.MF.keys():
                f1, f2 = fkey
                try:
                    x = coefs0[f2]
                except KeyError:
                    arg_name = retrieve_name(arg)
                    self_name = retrieve_name(self)
                    raise KeyError(f"Wavepacket '{arg_name}' does not contain functions " \
                            + f"for J = {f2} which are present for tensor '{self_name}'") from None

            coefs = self.MKvec(self.MF, self.K, coefs0)
            for f in coefs.keys():
                coefs[f] *= self.prefac
            res = coefs

        else:
            raise TypeError(f"Unsupported operand type(s) for *: {type(self).__name__} and " \
                    + f"{type(arg).__name__}") from None

        return res


    __rmul__ = __mul__


    def MKvec(self, MF, K, vec):
        """Computes product (MF x K) * vec, where 'x' and '*' denote tensor and dot products.

        Args:
            MF (dict): M-tensor contracted with electric field, same as Etensor().MF.
            K (dict): K-tensor, same as Etensor().K.
            vec (dict): Wavepacket vector, same as Psi().coefs.

        Returns:
            vec_new (dict): result of (MF x K) * vec, has the same structure as vec.
        """
        vec_new = {f : np.zeros(vec[f].shape, dtype=np.complex128) for f in vec.keys()}
        for fkey in list(set(MF.keys()) & set(K.keys())):
            f1, f2 = fkey
            dim1 = MF[fkey][0].shape[1]
            dim2 = K[fkey][0].shape[1]
            dim = MF[fkey][0].shape[0] * K[fkey][0].shape[0]
            vecT = np.transpose(vec[f2].reshape(dim1, dim2))
            for mm,kk in zip(MF[fkey],K[fkey]):
                tmat = csr_matrix.dot(kk, vecT)
                vec_new[f1] += np.dot(mm,np.transpose(tmat)).reshape(dim)
        return vec_new


    def U(self, t2, t1, psi, **kwargs):
        """Computes psi(t2) = U(t2, t1) psi(t1) using split-operator approach.

        Hamiltonian can be naturally split into stationary and field-dependent parts, i.e,
        H = H0 + V(t), where H0 is field-free molecular Hamiltonian, diagonal by the choice
        of the basis set.
        Using split-operator approach, the time-evolution operator can be calculated as
        exp(-i*dt/hbar*H) ~ exp(-i*dt/hbar/2*H0) * exp(-i*dt/hbar*V(t)) * exp(-i*dt/hbar/2*H0).

        Args:
            t2, t1 (float): Final and initial propagation times.
            psi (Psi()): Wavepacket at time t1, psi(t1).
            kwargs:
                units (str): Units of time, use 'ps' for picoseconds, 'fs' for femtoseconds.
                method (str): Method for computing matrix exponential exp(-i*dt/hbar*V(t)):
                              'taylor' - use Taylor series expansion
                              'lanszos' - (not implemented)
                              'pade' - (not implemented)
                maxorder (int): Maximum order, default is 20.
                conv (float): convergence tolerance, default is 1e-12.

        Returns:
            psi_new (Psi()): Wavepacket at time t2, psi(t2).
        """

        # time units
        if "units" in kwargs:
            units = kwargs["units"]
            try:
                lightspeed = lightspeed_ * {"ps":100.0/1.0e12, "fs":100.0/1.0e15}[units.lower()]
            except KeyError:
                raise KeyError(f"Unknown time units in 'units={units}'") from None
        else:
            lightspeed = lightspeed_ * 100.0/1.0e12 # default time units = ps

        # method
        methods = {"taylor" : 1, "arnoldi" : 2, "lanczos" : 3}
        if "method" in kwargs:
            method = kwargs["method"].lower()
            try:
                ind = methods[method]
            except KeyError:
                raise KeyError(f"Unknown method in 'method={method}'") from None
        else:
            method = "taylor"

        # maximal order
        maxorders = {"taylor" : 100, "arnoldi" : 100, "lanczos" : 100}
        if "maxorder" in kwargs:
            maxorder = kwargs["maxorder"]
        else:
            maxorder = maxorders[method]

        # convergence tolerance
        if "conv" in kwargs:
            conv = kwargs["conv"]
        else:
            conv = 1e-12

        dt = t2 - t1

        # exponential of field-free part exp(-i*dt/hbar/2 H0)

        fac = -1j * dt * np.pi * lightspeed
        expH0 = {}
        for f in psi.flist:
            expH0[f] = np.exp(fac*psi.energy[f])

        # apply exp(-i*dt/hbar/2 H0) to wavepacket

        coefs = {f : expH0[f] * psi.coefs[f] for f in psi.flist}

        # compute exp(-i*dt/hbar H'(t))

        coefs_new = {}

        if method=="taylor":

            # use Taylor series expansion of matrix exponential

            fac = -1j * dt * 2*np.pi*lightspeed * self.prefac

            time0 = time.time()

            V = []

            # zeroth power

            V.append(copy.deepcopy(coefs))

            # higher powers

            conv_k, k = 1, 0
            while k < maxorder and conv_k > conv:

                k += 1

                v = self.MKvec(self.MF, self.K, V[k - 1])
                v = {f : fac / k * v[f] for f in psi.flist}
                conv_k = sum([np.sum(np.abs(v[f])**2) for f in psi.flist])

                V.append(v)

            print("   t_tay = " + str(round(time.time() - time0, 4)) + " sec   k = " + str(k) + "   conv = " + str(conv_k))

            if k == maxorder:
                raise RuntimeError(f"Taylor series expansion of matrix exponential failed" \
                        +f" to converge, max expansion order {maxorder} reached")

            coefs_new = {f : sum([v[f] for v in V]) for f in psi.flist}

        elif method == "arnoldi":

            # use Arnoldi iteration

            fac = -1j * dt * 2*np.pi*lightspeed * self.prefac

            time0 = time.time()

            V = []
            H = np.zeros((maxorder, maxorder), dtype=np.complex128)

            # first Krylov basis vector

            V.append(copy.deepcopy(coefs))

            coefs_kminus1, coefs_k, conv_k, k = {}, V[0], 1, 1
            while k < maxorder and conv_k > conv:

                # extend ONB of Krylov subspace by another vector

                v = self.MKvec(self.MF, self.K, V[k - 1])
                for j in range(k):
                    H[j, k - 1] = sum([np.vdot(V[j][f], v[f]) for f in psi.flist])
                    v = {f : v[f] - H[j, k - 1] * V[j][f] for f in psi.flist}

                # calculate current approximation and convergence

                coefs_kminus1 = coefs_k
                expH_k = la.expm(fac * H[: k, : k])
                coefs_k = {f : sum([v_i[f] * expH_k[i, 0] for i,v_i in enumerate(V)]) for f in psi.flist}
                conv_k = sum([np.sum(np.abs(coefs_k[f] - coefs_kminus1[f])**2) for f in psi.flist])

                # stop if new vector vanishes

                H[k, k - 1] = np.sqrt(sum([np.sum(np.abs(v[f])**2) for f in psi.flist]))
                if H[k, k - 1] < conv:
                    break

                v = {f : v[f] / H[k, k - 1] for f in psi.flist}
                V.append(v)
                k += 1

            print("   t_arn = " + str(round(time.time() - time0, 4)) + " sec   k = " + str(k) + "   conv = " + str(conv_k))

            if k == maxorder:
                raise RuntimeError(f"Arnoldi iteration of matrix exponential failed" \
                        +f"to converge, max Krylov subspace dimension {maxorder} reached")

            coefs_new = coefs_k

        elif method == "lanczos" :

            # use Lanczos iteration

            fac = -1j * dt * 2*np.pi*lightspeed * self.prefac

            time0 = time.time()

            V, W = [], []
            T = np.zeros((maxorder, maxorder), dtype=np.complex128)

            # first Krylov basis vector

            V.append(copy.deepcopy(coefs))
            w = self.MKvec(self.MF, self.K, V[0])
            T[0, 0] = sum([np.vdot(w[f], V[0][f]) for f in psi.flist])
            W.append({f : w[f] - T[0, 0] * V[0][f] for f in psi.flist})

            coefs_kminus1, coefs_k, conv_k, k = {}, V[0], 1, 1
            while k < maxorder and conv_k > conv:

                # extend ONB of Krylov subspace by another vector and

                T[k - 1, k] = np.sqrt(sum([np.sum(np.abs(W[k - 1][f])**2) for f in psi.flist]))
                T[k, k - 1] = T[k - 1, k]

                if not T[k - 1, k] == 0:
                    v = {f : W[k - 1][f] / T[k - 1, k] for f in psi.flist}
                    V.append(v)

                # reorthonormalize ONB of Krylov subspace, if neccesary

                else:
                    v = {f : np.ones(V[k - 1][f].shape, dtype=np.complex128) for f in psi.flist}
                    for j in range(k):
                        proj_j = sum([np.vdot(V[j][f], v[f]) for f in psi.flist])
                        v = {f : v[f] - proj_j * V[j][f] for f in psi.flist}
                    norm_v = np.sqrt(sum([np.sum(np.abs(v[f])**2) for f in psi.flist]))
                    v = {f : v[f] / norm_v for f in psi.flist}
                    V.append(v)

                w = self.MKvec(self.MF, self.K, V[k])
                T[k, k] = sum([np.vdot(w[f], V[k][f]) for f in psi.flist])
                w = {f : w[f] - T[k, k] * V[k][f] - T[k - 1, k] * V[k - 1][f] for f in psi.flist}
                W.append(w)

                coefs_kminus1 = coefs_k

                expT_k = la.expm(fac * T[: k + 1, : k + 1])
                coefs_k = {f : sum([v_i[f] * expT_k[i, 0] for i,v_i in enumerate(V)]) for f in psi.flist}
                conv_k = sum([np.sum(np.abs(coefs_k[f] - coefs_kminus1[f])**2) for f in psi.flist])

                k += 1

            print("   t_lan = " + str(round(time.time() - time0, 4)) + " sec   k = " + str(k) + "   conv = " + str(conv_k))

            if k == maxorder:
                raise RuntimeError(f"Lanczos iteration of matrix exponential failed" \
                        +f"to converge, max Krylov subspace dimension {maxorder} reached")

            coefs_new = coefs_k

        # apply again exp(-i*dt/hbar/2 H0) to wavepacket

        coefs_new = {f : expH0[f] * coefs_new[f] for f in psi.flist}

        # return result in Psi() class

        psi_new = copy.deepcopy(psi)
        psi_new.coefs = coefs_new
        return psi_new


    def matrix(self, psi, ix=1, plus_diag=True):
        """Returns matrix representation of tensor operator or Hamiltonian (i.e. tensor times field)
        in the field-free basis 'psi'. For matrix elements of Hamiltonian, adds a field-free diagonal
        part when required.

        Args:
            psi (Psi()): Field-free basis set.
            ix (int): Tensor's Cartesian component index for which the matrix elements are computed.
            plus_diag (bool): if True, the field free energies will be added to the diagonal
                of Hamiltonian matrix.

        Returns:
            Matrix representation of tensor or Hamiltonian in the field-free basis as numpy array.
        """
        flist = psi.flist
        mat = {(f1,f2) : np.zeros( (len(psi.quanta[f1]), len(psi.quanta[f2])), dtype=np.complex128 ) \
                for f1 in flist for f2 in flist}

        try:
            # will return Hamiltonian matrix

            Mtens = self.MF

            for fkey in list(set(Mtens.keys()) & set(self.K.keys())):
                for mm,kk in zip(Mtens[fkey],self.K[fkey]):
                    mat[fkey] += kron(mm, kk).todense() * self.prefac

            if plus_diag==True:
                for f in flist:
                    mat[(f,f)] += np.diag(psi.energy[f])

        except AttributeError:
            # will return matrix elements of ix Cartesian component of tensor

            try:
                x = self.mcart[ix]
            except IndexError:
                raise IndexError(f"Cartesian component index '{ix}' is out of range for tensor '{self.name}' " \
                        + f"\nlist of components with indices for tensor '{self.name}': " \
                        + f"{[val for val in enumerate(self.mcart)]} ") from None

            Mtens = self.M

            for fkey in list(set(Mtens.keys()) & set(self.K.keys())):
                for mm,kk in zip(Mtens[fkey],self.K[fkey]):
                    mat[fkey] += kron(mm[ix,:,:], kk).todense()

        return np.block([[mat[(f1,f2)] for f2 in flist] for f1 in flist])



def read_states(filename, **kwargs):
    """Reads field free energies and quantum numbers
    """
    fl = open(filename, "r")

    # scan file for the number of states with different F quanta, max ID number, symmetry, etc.
    nstates = {}
    maxid = {}
    for line in fl:
        w = line.split()
        f = round(float(w[0]),1)
        id = np.int64(w[1])
        sym = w[2].upper()
        ndeg = int(w[3])
        enr = float(w[4])
        if 'emin' in kwargs and enr<kwargs['emin']:
            continue
        if 'emax' in kwargs and enr>kwargs['emax']:
            continue
        try:
            nstates[(f,sym)] += 1
        except:
            nstates[(f,sym)] = 1
        try:
            maxid[f] = max([id,maxid[f]])
        except:
            maxid[f] = id

    # create list of F quanta
    if 'flist' in kwargs:
        flist = [round(f,1) for f in kwargs['flist']]
    else:
        if 'fmax' in kwargs:
            fmax = min([ kwargs['fmax'], max([key[0] for key in nstates.keys()]) ])
            if fmax<kwargs['fmax']:
                print(f"read_states: fmax is set to {fmax} which is maximal F in states file {filename}")
        else:
            fmax = max([key[0] for key in nstates.keys()])
        if 'fmin' in kwargs:
            fmin = max([ kwargs['fmin'], min([key[0] for key in nstates.keys()]) ])
            if fmin>kwargs['fmin']:
                print(f"read_states: fmin is set to {fmin} which is minimal F in states file {filename}")
        else:
            fmin = min([key[0] for key in nstates.keys()])
        if 'df' in kwargs:
            df = kwargs['df']
        else:
            df = 1
        f1 = round(fmin,1)
        f2 = round(fmax,1)
        d = round(df)
        assert (f1>=0 and f2>=0),f"fmin={fmin} or fmax={fmax} is less than zero"
        assert (f2>=f1),f"fmin={fmin} > fmax={fmax}"
        assert (df>=1),f"df={df}<1"
        flist = [round(f,1) for f in np.linspace(f1,f2,int((f2-f1)/d)+1)]

    # create list of state symmetries
    if 'sym' in kwargs:
        sym_list = list( set(elem.upper() for elem in kwargs['sym']) & \
                         set([key[1] for key in nstates.keys()]) )
        bad_sym = list( set(elem.upper() for elem in kwargs['sym']) - \
                         set([key[1] for key in nstates.keys()]) )
        if len(bad_sym)>0:
            print(f"read_states: there are no states with symmetries {bad_sym} in states file {filename}")
    else:
        sym_list = set([key[1] for key in nstates.keys()])

    # read states

    states = {}
    map_id_to_istate = {}
    for f in flist:
        nst = sum([nstates[(f,sym)] for sym in sym_list if (f,sym) in nstates])
        if nst==0:
            continue
        states[f] = np.zeros( nst, dtype={'names':('f', 'id', 'ideg', 'istate', 'sym', 'enr', 'qstr'), \
            'formats':('f8', 'i8', 'i4', 'i8', 'U10', 'f8', 'U300')} )
        map_id_to_istate[f] = np.zeros( maxid[f]+1, dtype=np.int64 )
        map_id_to_istate[f][:] = -1

    fl.seek(0)

    nstates = {key[0]:0 for key in nstates}
    for line in fl:
        w = line.split()
        f = round(float(w[0]),1)
        id = np.int64(w[1])
        sym = w[2]
        ndeg = int(w[3])
        enr = float(w[4])
        qstr = ' '.join([w[i] for i in range(5,len(w))])
        if 'emin' in kwargs and enr<kwargs['emin']:
            continue
        if 'emax' in kwargs and enr>kwargs['emax']:
            continue
        if f not in flist:
            continue
        if sym.upper() not in sym_list:
            continue

        map_id_to_istate[f][id] = nstates[f]

        for ideg in range(ndeg):
            istate = nstates[f]
            states[f]['f'][istate] = f
            states[f]['id'][istate] = id
            states[f]['ideg'][istate] = ideg
            states[f]['sym'][istate] = sym
            states[f]['enr'][istate] = enr
            states[f]['qstr'][istate] = qstr
            states[f]['istate'][istate] = istate

            nstates[f] += 1

    fl.close()

    return states, map_id_to_istate


def read_tens(filename, f1, f2, map_id_to_istate, me_tol=1.0e-14):
    """Reads matrix-elements of Cartesian tensor operator
    """
    # extract F1 and F2 from the filename, check if they match the input values for F1 and F2

    m = re.findall(r'[-+]?\d*\.\d+|\d+', filename)
    ff1 = round(float(m[0]),1)
    ff2 = round(float(m[1]),1)

    if [round(f1,1),round(f2,1)] != [ff1, ff2]:
        if [round(f1,1),round(f2,1)] == [ff2, ff1]:
            swap_quanta = lambda x, y: (copy.copy(y), copy.copy(x))
            conjugate = lambda x, numtype: copy.copy(x) * {"imaginary":-1.0,"real":1.0}[numtype]
            transp = True
        else:
            raise RuntimeError(f"Error: values of f1 and f2 parameters = {f1} and {f2} don't match " \
                    + f"the values = {ff1} and {ff2} extracted from the name of matrix-elements " \
                    + f"file '{filename}'")
    else:
        swap_quanta = lambda x, y: (copy.copy(x), copy.copy(y))
        conjugate = lambda x, numtype: copy.copy(x)
        transp = False

    # read matrix elements

    fl = open(filename, "r")

    iline = 0
    eof = False
    read_m = False
    read_k = False
    icart = None
    icmplx = None
    maxdeg = 0
    nstates1 = 0
    nstates2 = 0

    for line in fl:
        strline = line.rstrip('\n')

        if iline==0:
            if strline!="Start richmol format":
                raise RuntimeError(f"Matrix-elements file '{filename}' has bogus header = '{strline}'")
            iline+=1
            continue

        if strline == "End richmol format":
            eof = True
            break

        if iline==1:
            w = strline.split()
            name = w[0]
            nomega = int(w[1])
            ncart = int(w[2])
            iline+=1
            continue

        if strline=="M-tensor":
            mtensor = [ np.zeros(( ncart, *swap_quanta(int(2.0*ff1+1), int(2.0*ff2+1)) ), \
                                   dtype=np.float64) for i in range(nomega) ]
            mtype = [None for i in range(ncart)]
            mcart = [None for i in range(ncart)]
            read_m = True
            read_k = False
            iline+=1
            continue

        if strline=="K-tensor":
            read_m = False
            read_k = True
            list_kval = [[] for i in range(nomega)]
            list_istate1 = [[] for i in range(nomega)]
            list_istate2 = [[] for i in range(nomega)]
            iline+=1
            continue

        if read_m is True and strline.split()[0]=="alpha":
            w = strline.split()
            icart = int(w[1])
            icmplx = int(w[2])
            scart = w[3]
            mtype[icart-1] = ("imaginary", "real")[icmplx+1]
            mcart[icart-1] = scart.upper()
            iline+=1
            continue

        if read_m is True:
            w = strline.split()
            m1 = float(w[0])
            m2 = float(w[1])
            mval = [ float(val) for val in w[2:] ]
            im1,im2 = swap_quanta( int(m1+ff1), int(m2+ff2) )
            for val,iomega in zip(mval,range(nomega)):
                mtensor[iomega][icart-1,im1,im2] = conjugate(val,mtype[icart-1])

        if read_k is True:
            w = strline.split()
            id1 = int(w[0])
            id2 = int(w[1])
            ideg1 = int(w[2])
            ideg2 = int(w[3])
            kval = [float(val) for val in w[4:]]
            istate1,istate2 = swap_quanta( map_id_to_istate[ff1][id1]+ideg1-1, \
                                           map_id_to_istate[ff2][id2]+ideg2-1 )
            if istate1<0 or istate2<0:
                continue
            ind_omega = [i for i in range(nomega) if abs(kval[i])>me_tol]
            for iomega in ind_omega:
                list_istate1[iomega].append(istate1)
                list_istate2[iomega].append(istate2)
                list_kval[iomega].append(kval[iomega])
            maxdeg = max([maxdeg,ideg1,ideg2])
            nstates1 = max([nstates1,istate1])
            nstates2 = max([nstates2,istate2])

        iline +=1
    fl.close()

    if eof is False:
        raise RuntimeError(f"Matrix-elements file '{filename}' has bogus footer = '{strline}'")

    ktensor = []
    for iomega in range(nomega):
        kmat = coo_matrix( (list_kval[iomega][:], (list_istate1[iomega],list_istate2[iomega])), \
                           shape=(nstates1+1,nstates2+1), dtype=np.float64 )
        ktensor.append( kmat.tocsr() )

    return name, ncart, nomega, mcart, mtype, mtensor, ktensor, transp


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
