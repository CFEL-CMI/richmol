import numpy as np
import scipy.constants as const
import functools
from scipy.sparse import diags
from scipy.sparse.linalg import expm, onenormest
from richmol import convert_units
from richmol.pyexpokit import zhexpv


def update_counter(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        vecs = func(self, *args, **kwargs)
        func_name = func.__name__
        try:
            icall = self._ncalls[func_name]
        except KeyError:
            self._ncalls[func_name] = 0
            icall = self._ncalls[func_name]
        time = self._time_grid[1][icall]
        self._ncalls[func_name] += 1
        return vecs, time # time corresponding to updated vectors
    return wrapper


class TDSE():
    """ Class for time-propagator

        Attrs:
            _t_start : int, float
                Starting time of time-propagation
            _t_end : int, float
                Terminal time of time-propagation
            _dt : int, float
                Time-step of propagation
            _t_to_s : float
                Factor to convert time into units of second
            _enr_to_J : float
                Factor to convert energy into units of Joule
            _time_grid : tuple()
                Lower borders, upper borders and centers of time-intervals to
                propagate over
            _exp_fac_H0 : numpy.ndarray
                Matrix exponential of the field-free part used during split
                operator approach

        Methods:
            __init__(**kwargs):
                Initializes `TDSE` object
            time_grid(grid_type='equidistant')
                Generates time-grid to propagate on
            init_state(
            update(H, vecs, H0=None, matvec_lib='scipy', tol=1e-15):
                Propagates vectors by one time-step
    """

    _ncalls = dict() # keep track of the number of calls for various functions


    def __init__(self, **kwargs):
        """ Initializes a time-propagator object

            Kwargs:
                t_start : int, float
                    Starting time of time-propagation
                t_end : int, float
                    Terminal time of time-propagation
                dt : int, float
                    Time-step of propagation
                t_units : str
                    Time units
                enr_units : str
                    Energy units
        """
        self._ncalls = dict()

        # starting time
        if 't_start' in kwargs:
            assert (type(kwargs['t_start']) in [int, float]), \
                f"`initial time `t_start` has bad type: " \
                    + f"'{type(kwargs['t_start'])}', (use 'int', 'float')"
            self._t_start = kwargs['t_start']
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


    def time_grid(self, grid_type='equidistant'):
        """ Generates time-grid to propagate on

            Args:
                grid_type : str
                    type of time grid

            Returns:
                t_c : numpy.ndarray
                    times at which to evaluate Hamiltonian
        """

        # grid type
        assert (grid_type.lower() in ['equidistant']), \
            f"time grid type `grid_type` has bad value: '{grid_type}', " \
                + f"(use 'equidistant')"

        # grid
        if grid_type.lower() == 'equidistant':
            grid_size = int((self._t_end - self._t_start) / self._dt)
            t_1 = np.linspace(
                self._t_start, self._t_end, num=grid_size, endpoint=False
            )
            t_2 = t_1 + self._dt
            t_c = t_1 + self._dt / 2

        self._time_grid = (t_1, t_2, t_c)

        return t_c


    def init_state(self, H, temp=None, thresh=1e-3, zpe=None):
        """ Generates initial state vectors

            Initial state vectors are eigenfunctions of Hamiltonian `H`. If
                `temp` is None, all eigenfunctions are returned unweighted. If
                `temp` is 0, only lowest energy eigenfunction is returned. If
                `temp` is > 0, all eigenfunctions are returned Boltzmann-
                -weighted.

        Args:
            H : :py:class:`field.CarTens`
                Hamiltonian operator
            temp : float
                Temperature in Kelvin
            thresh : float
                Collective threshold below which to neglect higher energy
                states
            zpe : float
                Zero-point energy, by default lowest eigenvalue of H is taken

        Returns:
            vecs : numpy.ndarray
                Initial state vectors (each row represents an initial state)
        """

        # temperature
        assert (temp is None or type(temp) in [int, float]), \
            f"temperature `temp` has bad type: '{type(temp)}', " \
                + f"(must be 'None', 'int', 'float')"
        if type(temp) in [int, float]:
            assert (temp >= 0), \
                f"temperature `temp` has bad value: '{temp}', " \
                    + f"(must be >= 0)"

        # partition function threshold
        assert (type(thresh) in [int, float]), \
            f"partition function threshold `thresh` has bad type: " \
                + f"'{type(thresh)}', (must be 'int', 'float')"
        assert (thresh >= 0), \
            f"partition function threshold `thresh` has bad value: " \
                + f"'{thresh}' (must be >= 0)"

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
            raise AttributeError(
                f"Hamiltonian `H` has bad type (must be Hamiltonian)"
            ) from None
        enrs, vecs = np.linalg.eigh(hmat)

        if zpe is None:
            zpe = enrs[0]
        if enrs[0] - zpe < 0:
            raise ValueError(
                f"input zero-point energy '{zpe}' is greater than " \
                    + f"lowest energy '{enrs[0]}'"
            ) from None

        # Boltzmann weights
        vecs = vecs.T
        if temp is None:
            pass
        elif temp == 0:
            vecs = vecs[0 : 1]
            # weights = [np.exp(-beta * (enrs[0] - zpe))]
        else:
            enrs *= self._enr_to_J
            beta = 1.0 / (const.value("Boltzmann constant") * temp) # (1/J)
            weights = np.exp(-beta * (enrs - zpe))
            weights /= np.sum(weights)
            inds = [ i for i in range(len(weights))
                     if (1 - np.sum(weights[: i + 1])) > thresh ]
            vecs = vecs[inds] * np.expand_dims(weights[inds], axis=1)
 
        return vecs


    @update_counter
    def update(self, H, vecs, H0=None, matvec_lib='scipy', tol=1e-15):
        """ Propagates vectors by one time-step

            Args:
                H : :py:class:`field.CarTens`
                    Hamiltonian operator
                vecs : list
                    Vectors to propagate in time

            Kwargs:
                H0 : :py:class:`field.CarTens`
                    Field-free Hamiltonian operator (default: None)
                matvec_lib : str
                    Library to use for matrix-vector products (default:
                    'scipy')
                tol : int, float
                    See :py:func:`_Lanczos_expmv`
               
            Returns:
                vecs2 : list
                    Vectors propagated in time
        """

        # factor in exponent
        exp_fac = -1j * self._dt * self._t_to_s * self._enr_to_J \
            / const.value("reduced Planck constant")

        # `CarTens.vec()` wrapper
        def cartensvec(v):

            # numpy array to CarTens compatible vec
            vec_, ind = dict(), 0
            for J in H.Jlist2:
                vec_[J] = dict()
                for sym in H.symlist2[J]:
                    vec_[J][sym] = v[ind: ind + H.dim2[J][sym]]
                    ind += H.dim2[J][sym]

            vec_ = H.vec(vec_, matvec_lib=matvec_lib)

            # CarTens compatible vec to numpy array
            vec_ = np.concatenate(
                tuple(
                    [ vec_[J][sym]
                      if J in vec_.keys() and sym in vec_[J].keys()
                      else np.zeros(H.dim2[J][sym], dtype=np.complex128)
                      for J in H.Jlist2 for sym in H.symlist2[J] ]
                )
            )

            return vec_

        # propagate
        if H0 is not None:
            if '_exp_fac_H0' not in list(self.__dict__.keys()):
                H0_mat = H0.tomat(form='full', cart='0')
                assert ((H0_mat - diags(H0_mat.diagonal())).nnz == 0), \
                    f"non-diagonal H0: split-operator approach not applicable/implemented"
                self._exp_fac_H0 = np.exp(exp_fac / 2 * H0_mat.diagonal())
            vecs2 = vecs * self._exp_fac_H0
            if hasattr(H, 'mfmat') and len(H.mfmat) > 0:
                for ind, vec in enumerate(vecs2):
                    vecs2[ind] = _expv_lanczos(
                        vec, exp_fac, lambda v : cartensvec(v), tol=tol
                    )
                    # vecs2[ind] = zhexpv(
                    #     vec,
                    #     onenormest(H.tomat(form='full')),
                    #     12,
                    #     exp_fac.imag,
                    #     lambda v : cartensvec(v) * 1j,
                    #     tol = tol
                    # )
            vecs2 *= self._exp_fac_H0
        else:
            vecs2 = np.empty(vecs.shape, dtype=vecs.dtype)
            for ind, vec in enumerate(vecs):
                vecs2[ind] = _expv_lanczos(
                    vec, exp_fac, lambda v : cartensvec(v), tol=tol
                )

        return vecs2


def _expv_lanczos(vec, fac, matvec, maxorder=100, tol=1e-15):
    """ Computes  epx(fac * H) * vec  using in-house Lanczos algorithm 

        Args:
            vec : numpy.ndarray
                Vector to multiply with matrix exponential
            fac : int, float, complex
                Factor of matrix exponential
            matvec : lambda
                Matrix-vector multiplication function
            maxorder : int
                Maximum order of Krylov subspace basis (default: 100)
            tol : int, float
                Tolerance w.r.t. metric of two subsequent solutions below
                which to stop expanding basis of Krylov subspace (default: 0)

        Returns:
            u_k : numpy.ndarray
                Numerical solution
    """

    # tolerance
    assert (type(tol) in [int, float]), \
        f"Tolerance `tol` has bad type: '{type(tol)}', (must be 'int', " \
            + f"'float')"
    assert (tol > 0 and tol <= 1), \
        f"Tolerance `tol` has bad value: '{tol}', (must be > 0 and <= 1)"

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
        expT_k = expm(fac * T[: k + 1, : k + 1])
        u_k = sum([expT_k[i, 0] * v_i for i,v_i in enumerate(V)])
        conv_k = sum(np.abs(u_k - u_kminus1)**2)

        k += 1

    if k == maxorder:
        raise ValueError(
            f"Lanczos reached maximum order of '{maxorder}' without " \
                + f"convergence"
        )

    return u_k

