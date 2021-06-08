import numpy as np
from scipy import constants
import functools
from scipy.sparse.linalg import expm
from richmol import convert_units
# from mpi4py import MPI


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

    @update_counter
    def update(self, H, H0=None, vecs=None, matvec_lib='scipy', tol=1e-15):
        # this factor should make (dt/hbar * H) dimensionless
        exp_fac = -1j * self.dt * self.time_units * self.energy_units \
            / constants.value("reduced Planck constant")

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

        # initialize v (TODO: maybe put more thought into this)
        if vecs is None:
            vecs = np.zeros(
                sum( [ dim2_J_sym for J, dim2_J in H.dim2.items()
                       for sym, dim2_J_sym in dim2_J.items() ] ),
                dtype=np.complex128
            )
            vecs[0] += 1
            vecs = [vecs]
        elif type(vecs) is not list:
            vecs = [vecs]

        # propagate
        vecs2 = []
        if H0 is not None:
            for vec in vecs:
                if 'expfacH0' not in list(self.__dict__.keys()):
                    self.__dict__['expfacH0'] = np.exp(
                        exp_fac / 2 * H0.tomat(form='full').diagonal()
                    )
                res = self.__dict__['expfacH0'] * vec
                res = expv_lanczos(
                    res, exp_fac, lambda v : cartensvec(v), tol=tol
                )
                vecs2.append(self.__dict__['expfacH0'] * res)
        else:
            for vec in vecs:
                vecs2.append(expv_lanczos(
                    vec, exp_fac, lambda v : cartensvec(v), tol=tol)
                )

        return vecs2


    def time_grid(self, grid='equidistant', field=None):
        if grid.lower() == 'equidistant':
            grid_size = int((self.tend - self.tstart) / self.dt)
            t1 = np.linspace(
                self.tstart, self.tend, num=grid_size, endpoint=False
            )
            t2 = t1 + self.dt
            tc = t1 + self.dt / 2
        else:
            raise ValueError(f"unknown time grid type: '{grid}'") from None
        self._time_grid = (t1, t2, tc)
        return self._time_grid[2] # times at which to evaluate Hamiltonian


    @property
    def energy_units(self):
        try:
            return self.enr_to_J
        except AttributeError:
            raise AttributeError(
                f"energy units not set, use 'energy_units = <units>' " \
                    + f"with <units> one of ('cm-1')"
            ) from None

    @energy_units.setter
    def energy_units(self, units):
        if units.lower() in ("cm-1", "cm^-1", "1/cm", "invcm"):
            self.enr_to_J = 1 / convert_units.J_to_invcm()
        else:
            raise ValueError(f"unknown energy units: '{units}'") from None


    @property
    def time_units(self):
        try:
            return self.time_to_sec
        except AttributeError:
            raise AttributeError(
                f"time units not set, use 'time_units = <units>' "
                    + f"with <units> one of ('ps', 'fs', ns', 'aut')"
            ) from None

    @time_units.setter
    def time_units(self, units):
        if units.lower() in ("ps", "picoseconds", "pico"):
            self.time_to_sec = 1e-12
        elif units.lower() in ("fs", "femtoseconds", "femto"):
            self.time_to_sec = 1e-15
        elif units.lower() in ("ns", "nanoseconds", "nano"):
            self.time_to_sec = 1e-9
        elif units.lower() in ("au", "aut"):
            self.time_to_sec = constants.value("atomic unit of time")
        else:
            raise ValueError(f"unknown time units: '{units}'") from None


    @property
    def tstart(self):
        try:
            return self._tstart
        except AttributeError:
            raise AttributeError(
                f"initial time not set, use 'tstart = <value>'"
            ) from None

    @tstart.setter
    def tstart(self, val):
        try:
            assert (val < self.tend), \
                f"initial time '{val}' greater than terminal time '{self.t2}'"
        except AttributeError:
            pass
        self._tstart = val


    @property
    def tend(self):
        try:
            return self._tend
        except AttributeError:
            raise AttributeError(
                f"terminal time not set, use 'tend = <value>'"
            ) from None

    @tend.setter
    def tend(self, val):
        try:
            assert (val > self.tstart), \
                f"terminal time '{val}' smaller than initial time '{self.t1}'"
        except AttributeError:
            pass
        self._tend = val


    @property
    def dt(self):
        try:
            return self._dt
        except AttributeError:
            raise AttributeError(
                f"time step not set, use 'dt = <value>'"
            ) from None

    @dt.setter
    def dt(self, val):
        try:
            assert (int((self.tend - self.tstart) / val) > 0), \
                f"time step '{val}' greater than time interval width " \
                    + f"'{self.tend - self.tstart}'"
        except AttributeError:
            pass
        self._dt = val


    def initial_state(self, H, temp=None):
        """Generates initial state vectors as eigenfunctions of Hamiltonian `H`
        for given temperature `temp` (in Kelvin).
        If temperature is None, all eigenfunctions without weighting will be returned.

        Args:
            H : :py:class:`field.CarTens`
                Hamiltonian operator
            temp : float
                Temperature in Kelvin

        Returns:
            vec : list
                List of initial vectors
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
            w = [1.0 if i==0 else 0 for i in range(v.shape[1])]
        else:
            e *= self.energy_units # in Joules
            beta = 1.0 / (constants.value("Boltzmann constant") * temp) # in 1/Joules
            zpe = e[0]
            w = np.exp(-beta * (e - zpe))
            pf = np.sum(w)
            w /= pf
        vec = [v[:,i] * w[i] for i in range(v.shape[1])]
        return vec


def expv_lanczos(vec, t, matvec, maxorder=100, tol=0):
    """ Computes epx(t*a)*v using Lanczos algorithm """

    V, W = [], []
    T = np.zeros((maxorder, maxorder), dtype=vec.dtype)

    # first Krylov basis vector
    V.append(vec)
    w = matvec(V[0])
    T[0, 0] = np.vdot(w, V[0])
    W.append(w - T[0, 0] * V[0])

    # higher orders
    u_kminus1, u_k, conv_k, k = {}, V[0], 1, 1
    while k < maxorder and conv_k > tol:

        # extend ONB of Krylov subspace by another vector
        T[k - 1, k] = np.sqrt(np.sum(np.abs(W[k - 1])**2))
        T[k, k - 1] = T[k - 1, k]
        if not T[k - 1, k] == 0:
            V.append(W[k - 1] / T[k - 1, k])

        # reorthonormalize ONB of Krylov subspace, if neccesary
        else:
            v = np.ones(V[k - 1].shape, dtype=np.complex128)
            for j in range(k):
                proj_j = np.vdot(V[j], v)
                v = v - proj_j * V[j]
            norm_v = np.sqrt(np.sum(np.abs(v)**2))
            V.append(v / norm_v)

        w = matvec(V[k])
        T[k, k] = np.vdot(w, V[k])
        w = w - T[k, k] * V[k] - T[k - 1, k] * V[k - 1]
        W.append(w)

        # calculate current approximation and convergence
        u_kminus1 = u_k
        expT_k = expm(t * T[: k + 1, : k + 1])
        u_k = sum([expT_k[i, 0] * v_i for i,v_i in enumerate(V)])
        conv_k = np.sum(np.abs(u_k - u_kminus1)**2)

        k += 1

    if k == maxorder:
        print("lanczos reached maximum order of {}".format(maxorder))

    return u_k

