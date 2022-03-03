import numpy as np
from scipy.sparse import csr_matrix
from richmol.hyperfine.basis import nearEqualCoupling
from richmol.field import CarTens
import py3nj
from collections import defaultdict


class LabTensor(CarTens):
    """Laboratory-frame Cartesian tensor operator in spin-rovibrational (hyperfine)
    basis

    Given hyperifne solutions `h0` and a Cartesian tensor operator in rovibraitonal
    basis `tens`, transforms the latter into hyperfine basis

    This is a subclass of :py:class:`richmol.field.CarTens` class.

    Args:
        h0 : :py:class:`richmol.field.CarTens`
            Spin-rovibrational solutions
        tens : :py:class:`richmol.field.CarTens`
            Cartesian tensor operator in rovibrational basis
        thresh : float
            Threshold for neglecting matrix elements
    """

    # NOTE: at some point need to create an abstract class with generic types
    #       to unify both hyperfine.labtens.LabTensor and rot.labtens.LabTensor.
    #       For now, just copy some of attributes from rot.labtens.LabTensor.

    # transformation matrix from Cartesian to spherical-tensor representation
    # for tensors of different ranks (dict keys)

    tmat_s = {1 : np.array([ [np.sqrt(2.0)/2.0, -np.sqrt(2.0)*1j/2.0, 0], \
                             [0, 0, 1.0], \
                             [-np.sqrt(2.0)/2.0, -np.sqrt(2.0)*1j/2.0, 0] ], dtype=np.complex128),
              2 : np.array([ [-1.0/np.sqrt(3.0), 0, 0, 0, -1.0/np.sqrt(3.0), 0, 0, 0, -1.0/np.sqrt(3.0)], \
                             [0, 0, -0.5, 0, 0, 0.5*1j, 0.5, -0.5*1j, 0], \
                             [0, 1.0/np.sqrt(2.0)*1j, 0, -1.0/np.sqrt(2.0)*1j, 0, 0, 0, 0, 0], \
                             [0, 0, -0.5, 0, 0, -0.5*1j, 0.5, 0.5*1j, 0], \
                             [0.5, -0.5*1j, 0, -0.5*1j, -0.5, 0, 0, 0, 0], \
                             [0, 0, 0.5, 0, 0, -0.5*1j, 0.5, -0.5*1j, 0], \
                             [-1.0/np.sqrt(6.0), 0, 0, 0, -1.0/np.sqrt(6.0), 0, 0, 0, (1.0/3.0)*np.sqrt(6.0)], \
                             [0, 0, -0.5, 0, 0, -0.5*1j, -0.5, -0.5*1j, 0], \
                             [0.5, 0.5*1j, 0, 0.5*1j, -0.5, 0, 0, 0, 0] ], dtype=np.complex128) }

    # inverse spherical-tensor to Cartesian transformation matrix

    tmat_x = {key : np.linalg.pinv(val) for key, val in tmat_s.items()}

    # Cartesian components and irreducible representations for tensors of different ranks

    cart_ind = {1 : ["x","y","z"],
                2 : ["xx","xy","xz","yx","yy","yz","zx","zy","zz"]}

    irrep_ind = {1 : [(1,-1),(1,0),(1,1)],
                 2 : [(o,s) for o in range(3) for s in range(-o,o+1)] }


    def __init__(self, h0, tens, thresh=None):

        if not thresh:
            thresh_ = np.finfo(np.complex128).eps
        else:
            thresh_ = thresh

        self.rank = tens.rank
        self.os = tens.os
        self.cart = tens.cart

        try:
            cart_ind_ = self.cart_ind[self.rank]
            cart_ind = [cart_ind_.index(cart) for cart in self.cart]
            self.Ux = self.tmat_x[self.rank]
        except KeyError:
            raise KeyError(
                f"Hyperfine matrix elements for tensor {retrieve_name(tens)} " + \
                f"of rank = {self.rank} are not implemented") from None

        mydict = lambda: defaultdict(mydict)
        self.kmat = mydict()
        self.mmat = mydict()

        # K-tensor
        for f1 in h0.Jlist1:
            for sym1 in h0.symlist1[f1]:
                q1 = h0.quantaSpinJSym[f1][sym1]
                v1 = h0.eigvec[f1][sym1]
                for f2 in h0.Jlist2:
                    for sym2 in h0.symlist1[f2]:
                        q2 = h0.quantaSpinJSym[f2][sym2]
                        v2 = h0.eigvec[f2][sym2]
                        for omega in set([o for (o, s) in self.os]):
                            me = self._kPrim(f1, f2, q1, q2, tens, omega)
                            me = np.dot(np.conj(v1).T, np.dot(me, v2))
                            me[np.abs(me) < thresh_] = 0
                            me_csr = csr_matrix(me)
                            if me_csr.nnz > 0:
                                self.kmat[(f1, f2)][(sym1, sym2)][omega] = me_csr

        # M-tensor
        for f1 in h0.Jlist1:
            for sym1 in h0.symlist1[f1]:
                q1 = h0.quanta_m1[f1][sym1]
                for f2 in h0.Jlist2:
                    for sym2 in h0.symlist1[f2]:
                        q2 = h0.quanta_m2[f2][sym2]
                        for omega in set([o for (o, s) in self.os]):
                            me = self._mPrim(f1, f2, q1, q2, omega)
                            me = np.take(me, cart_ind, axis=2)
                            me[np.abs(me) < thresh_] = 0
                            for icart, cart in enumerate(self.cart):
                                me_csr = csr_matrix(me[:, :, icart])
                                if me_csr.nnz > 0:
                                    self.mmat[(f1, f2)][(sym1, sym2)][omega][cart] = me_csr


        # copy other attributes from h0
        attrs = ("Jlist1", "Jlist2", "symlist1", "symlist2", "dim1", "dim2",
                 "dim_k1", "dim_k2", "dim_m1", "dim_m2", "quanta_k1", "quanta_k2",
                 "quanta_m1", "quanta_m2")
        for attr in attrs:
            val = getattr(h0, attr)
            setattr(self, attr, val)


    def _kPrim(self, f1, f2, quantaSpinJSym1, quantaSpinJSym2, tens, omega):
        hMat = []
        for i, (spin1, j1, rvSym1, dim1) in enumerate(quantaSpinJSym1):
            I1 = spin1[-1]
            hMat_ = []
            for j, (spin2, j2, rvSym2, dim2) in enumerate(quantaSpinJSym2):
                mat = np.zeros((dim1, dim2), dtype=np.complex128)
                if all(s1 == s2 for s1, s2 in zip(spin1, spin2)):
                    I2 = spin2[-1]
                    fac = f1 + I2
                    assert (float(fac).is_integer()), f"Non-integer power in (-1)**f: '(-1)**{fac}'"
                    fac = int(fac)
                    coef = (-1)**fac * np.sqrt((2*j1+1) * (2*j2+1)) \
                         * py3nj.wigner6j(int(j1*2), int(f1*2), int(I2*2),
                                          int(f2*2), int(j2*2), int(omega*2))
                    try:
                        mat = tens.kmat[(j1, j2)][(rvSym1, rvSym2)][omega].toarray() * coef
                    except (AttributeError, KeyError):
                        pass
                hMat_.append(mat)
            hMat.append(hMat_)
        return np.bmat(hMat)


    def _mPrim(self, f1, f2, quanta_m1, quanta_m2, omega):
        sigma_ind = np.array([i for i, (o, s) in enumerate(self.os) if o==omega])
        sigma = np.array([s for i, (o, s) in enumerate(self.os) if o==omega])
        ns = len(sigma)
        assert (len(sigma_ind) > 0), \
            f"Tensor irrep 'omega' = {omega} is not contained in 'self.os' = {self.os}"
        Ux = np.take(self.Ux, sigma_ind, axis=1)
        me = np.zeros((len(quanta_m1), len(quanta_m2), Ux.shape[0]), dtype=np.complex128)
        for i, m1 in enumerate(quanta_m1):
            fac = f1 - m1
            assert (float(fac).is_integer()), f"Non-integer power in (-1)**f: '(-1)**{fac}'"
            fac = int(fac)
            coef = (-1)**fac * np.sqrt((2*f1+1) * (2*f2+1))
            for j, m2 in enumerate(quanta_m2):
                threej = py3nj.wigner3j([int(f1*2)]*ns, [int(omega*2)]*ns, [int(f2*2)]*ns,
                                        [int(-m1*2)]*ns, sigma*2, [int(m2*2)]*ns)
                me[i, j, :] = np.dot(Ux, threej) * coef
        return me


    def class_name(self):
        """Generates string containing name of the parent class"""
        base = list(self.__class__.__bases__)[0]
        return base.__module__ + "." + base.__name__


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [ var_name for var_name, var_val in fi.frame.f_locals.items() \
                  if var_val is var ]
        if len(names) > 0:
            return names[0]

