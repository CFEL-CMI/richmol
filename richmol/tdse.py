import numpy as np
import scipy.constants as const
import functools
from scipy.sparse.linalg import expm
from richmol import convert_units


def update_counter(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        vecs = func(self, *args, **kwargs)
        time = self._time_grid[1][wrapper.count]
        wrapper.count += 1
        return vecs, time # time corresponding to updated vectors
    wrapper.count = 0
    return wrapper


class TDSE():

    def __init__(self, **kwargs):

        # starting time
        if 't_start' in kwargs:
            assert (type(kwargs['t_start']) in [int, float]), \
                f"`initial time `t_start` has bad type: " \
                    + f"'{type(kwargs['t_start'])}', (use 'int', 'float')"
            self.t_start = kwargs['t_start']
        else:
            self._t_start = 0.0

        # terminal time
        assert ('t_end' in kwargs), \
            f"terminal time 't_end' not found in kwargs"
        assert (type(kwargs['t_end']) in [int, float]), \
            f"terminal time `t_end` has bad type: " \
                + f"'{type(kwargs['t_end'])}', (use 'int', 'float')"
        assert (kwargs['t_end'] > self._t_start), \
            f"terminal time `t_end` has bad value: '{kwargs['t_end']}', " \
                + f"(must be > '{self._t_start}')"
        self._t_end = kwargs['t_end']

        # time step
        assert ('dt' in kwargs), \
            f"time step `dt` not found in kwargs"
        assert (type(kwargs['dt']) in [int, float]), \
            f"time step `dt` has bad type: '{type(kwargs['dt'])}', " \
                + f"(use 'int', 'float')"
        assert (kwargs['dt'] <= (self._t_end - self._t_start)), \
            f"time step `dt` has bad value : '{kwargs['dt']}', " \
                + f"(must be <= '{self._t_end - self._t_start}')"
        self._dt = kwargs['dt']

        # time units
        t_units = {
            'fs' : 1e-15,
            'ps' : 1e-12,
            'ns' : 1e-9,
            'au' : const.value("atomic unit of time")
        }
        if 't_units' in kwargs:
            assert (type(kwargs['t_units']) == str), \
                f"time units `t_units` has bad type: " \
                    + f"'{type(kwargs['t_units'])}', (use 'str')"
            assert (kwargs['t_units'] in list(t_units.keys())), \
                f"time units `t_units` has bad value: " \
                    + f"'{kwargs['t_units']}', (use 'fs', 'ps', 'ns', 'au')"
            self._t_to_s = t_units[kwargs['t_units']]
        else:
            self._t_to_s = t_units['ps']

        # energy units
        enr_units = {'invcm' : 1 / convert_units.J_to_invcm()}
        if 'enr_units' in kwargs:
            assert (type(kwargs['enr_units']) == str), \
                f"energy units `enr_units` has bad type: " \
                    + f"'{type(kwargs['enr_units'])}', (use 'str')"
            assert (kwargs['enr_units'] in list(enr_units.keys())), \
                f"energy units `enr_units` has bad value: " \
                    + f"'{kwargs['enr_units']}', (use 'invcm')"
            self._enr_to_J = enr_units[kwargs['enr_units']]
        else:
            self._enr_to_J = enr_units['invcm']


    @update_counter
    def update(self, H, H0=None, vecs=None, matvec_lib='scipy', tol=1e-15):
        # this factor should make (dt/hbar * H) dimensionless
        exp_fac = -1j * self._dt * self._t_to_s * self._enr_to_J \
            / const.value("reduced Planck constant")

        # numpy array to CarTens compatible vec
        def array_to_vec(array):
            vec_, ind = dict(), 0
            for J in H.Jlist2:
                vec_[J] = dict()
                for sym in H.symlist2[J]:
                    vec_[J][sym] = array[ind: ind + H.dim2[J][sym]]
                    ind += H.dim2[J][sym]
            return vec_

        # CarTens compatible vec to numpy array
        def vec_to_array(vec_):
            return np.concatenate(
                tuple( [ vec_[J][sym] for J in H.Jlist2
                         for sym in H.symlist2[J] ] )
            )

        # `CarTens.vec()` wrapper
        def cartensvec(v):
            vec_ = H.vec(array_to_vec(v), matvec_lib=matvec_lib)
            return vec_to_array(vec_)

        # propagate
        vecs2 = []
        if H0 is not None:
            for vec in vecs:
                if 'expfacH0' not in list(self.__dict__.keys()):
                    self.__dict__['expfacH0'] = np.exp(
                        exp_fac / 2 * H0.tomat(form='full').diagonal()
                    )
                res = self.__dict__['expfacH0'] * vec
                res = _expv_lanczos(
                    res, exp_fac, lambda v : cartensvec(v), tol=tol
                )
                vecs2.append(self.__dict__['expfacH0'] * res)
        else:
            for vec in vecs:
                vecs2.append(_expv_lanczos(
                    vec, exp_fac, lambda v : cartensvec(v), tol=tol)
                )

        return vecs2


    def time_grid(self, grid_type='equidistant'):
        """ Generates the time-grid on which to perform propagation

            Args:
                grid_type : str
                    type of time grid

            Returns:
                t_c : numpy.ndarray
                    times at which to evaluate Hamiltonian
        """

        assert (grid_type.lower() in ['equidistant']), \
            f"time grid type `grid_type` has bad value: '{grid_type}', " \
                + f"(use 'equidistant')"

        if grid_type.lower() == 'equidistant':
            grid_size = int((self._t_end - self._t_start) / self._dt)
            t_1 = np.linspace(
                self._t_start, self._t_end, num=grid_size, endpoint=False
            )
            t_2 = t_1 + self._dt
            t_c = t_1 + self._dt / 2

        self._time_grid = (t_1, t_2, t_c)

        return t_c


    def init_state(self, H, temp=None, pf_thr=1e-3):
        """Generates initial state vectors as eigenfunctions of Hamiltonian `H`
        for given temperature `temp` (in Kelvin).
        If temperature is None, all eigenfunctions without Boltzmann weighting will be returned.

        Args:
            H : :py:class:`field.CarTens`
                Hamiltonian operator
            temp : float
                Temperature in Kelvin
            pf_thr : float
                Threshold for neglecting all higher energy states whose collective contribution
                to a partition function is less than `pf_thr`

        Returns:
            vec : list
                List of Boltzmann-weighted initial vectors
        """
        # convert field-free tensor into Hamiltonian
        try:
            if H.cart[0] == "0":
                H.field([0, 0, 1])
        except AttributeError:
            pass

        # eigenvectors of H
        try:
            x = H.mfmat
            hmat = H.tomat(form="full", repres="dense")
        except AttributeError:
            raise AttributeError(f"input argument 'H' is not a Hamiltonian") from None
        e, v = np.linalg.eigh(hmat)

        # Boltzmann weights
        if temp is None:
            w = [1.0 for i in range(v.shape[1])]
        elif temp == 0:
            w = [1.0]
        else:
            e *= self.energy_units # in Joules
            beta = 1.0 / (constants.value("Boltzmann constant") * temp) # in 1/Joules
            zpe = e[0]
            w = np.exp(-beta * (e - zpe))
            pf = np.sum(w)
            w /= pf
            mask = [0]
            mask += [i for i in range(1, len(w)) if np.abs(sum(w[:i+1]) - 1) > np.abs(pf_thr)]
            w = w[mask]
        vec = [v[:,i] * w[i] for i in range(len(w))]
        return vec


def _expv_lanczos(vec, t, matvec, maxorder=100, tol=0):
    """ Computes  epx(fac * H) * vec  using in-house Lanczos algorithm 

        Args:
            vec : numpy.ndarray
                vector to multiply with matrix exponential
            fac : int, float, complex
                factor of matrix exponential
            matvec : lambda
                function to use for matrix-vector multiplication (must take
                vector as argument)
            maxorder : int
                the maximum order of Krylov subspace basis (default: 100)
            tol : int, float
                the tolerance w.r.t. metric of two subsequent solutions below
                which to stop expanding basis of Krylov subspace (default: 0)

        Returns:
            u_k : numpy.ndarray
                numerical solution
    """

    V, W = [], []
    T = np.zeros((maxorder, maxorder), dtype=vec.dtype)

    # first Krylov subspace basis vector
    V.append(vec)
    w = matvec(V[0])
    T[0, 0] = np.vdot(w, V[0])
    W.append(w - T[0, 0] * V[0])

    # higher orders
    u_kminus1, u_k, conv_k, k = {}, V[0], 1, 1
    while k < maxorder and conv_k > tol:

        # extend ONB of Krylov subspace by another vector
        T[k - 1, k] = np.sqrt(sum(np.abs(W[k - 1])**2))
        T[k, k - 1] = T[k - 1, k]
        if not T[k - 1, k] == 0:
            V.append(W[k - 1] / T[k - 1, k])

        # reorthonormalize ONB of Krylov subspace, if neccesary
        else:
            v = np.ones(V[k - 1].shape, dtype=np.complex128)
            for j in range(k):
                proj_j = np.vdot(V[j], v)
                v = v - proj_j * V[j]
            norm_v = np.sqrt(sum(np.abs(v)**2))
            V.append(v / norm_v)

        w = matvec(V[k])
        T[k, k] = np.vdot(w, V[k])
        w = w - T[k, k] * V[k] - T[k - 1, k] * V[k - 1]
        W.append(w)

        # current approximation; metric with previous approximation
        u_kminus1 = u_k
        expT_k = expm(t * T[: k + 1, : k + 1])
        u_k = sum([expT_k[i, 0] * v_i for i,v_i in enumerate(V)])
        conv_k = sum(np.abs(u_k - u_kminus1)**2)

        k += 1

    if k == maxorder:
        print("lanczos reached maximum order of {}".format(maxorder))

    return u_k

