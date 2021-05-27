import numpy as np
from scipy import constants
import functools
from richmol.pyexpokit import expv_lanczos
import time

"""
# very rough idea

from tdse import tdse

tdse.tstart = 0
tdse.tend = 100
tdse.dt = 0.01
tdse.time_units = "ps"
tdse.energy_units = "1/cm"

# external fields
dc = lambda time: [0, 0, 1000 * time]
ac = lambda time: [0, 0, 1e+12 * np.cos(10000 * time)]

vec = ... # initial vector

for t in tdse.times():

    mu.fied(-dc(t))
    alpha.field(-1/2 * ac(t))

    H = mu + alpha

    vec, t2 = tdse.update(H+H0, vec=vec)
    vec, t2 = tdse.update(H, H0=H0, vec=vec) # use split operator

    print(f"vec at time {t2} is: {vec}")
"""

def update_counter(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        vec = func(self, *args, **kwargs)
        time = self.time_grid[1][wrapper.count]
        wrapper.count += 1
        return vec, time # returns time corresponding to updated vector
    wrapper.count = 0
    return wrapper


class TDSE():

    @update_counter
    def update(self, H, H0=None, vecs=None, matvec_lib='scipy'):
        # this factor should make (dt/hbar * H) dimensionless
        exp_fac = -1j * self.dt * self.time_units / constants.value("reduced Planck constant") * self.energy_units

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
        tol = 1e-15
        if H0 is not None:
            for vec in vecs:
                if 'expfacH0' not in list(self.__dict__.keys()):
                    self.__dict__['expfacH0'] = np.exp(
                        exp_fac / 2 * H0.tomat(form='full').diagonal()
                    )
                res = self.__dict__['expfacH0'] * vec
                res = expv_lanczos(res, exp_fac, lambda v : cartensvec(v), tol=tol)
                vecs2.append(self.__dict__['expfacH0'] * res)
        else:
            for vec in vecs:
                vecs2.append(expv_lanczos(vec, exp_fac, lambda v : cartensvec(v), tol=tol))

        return vecs2


    def times(self, grid="equidistant", field=None):
        if grid.lower() == "equidistant":
            npt = int((self.tend - self.tstart) / self.dt)
            t1 = np.array([self.tstart + self.dt * (i) for i in range(npt+1)])
            t2 = np.array([self.tstart + self.dt * (i+1) for i in range(npt+1)])
            tc = 0.5*(t1 + t2)
        else:
            raise ValueError(f"unknown grid type: '{grid}'") from None
        self.time_grid = (t1, t2, tc)
        return tc # returns time at which Hamiltonian need to be computed


    @property
    def energy_units(self):
        try:
            return self.enr_joule
        except AttributeError:
            raise AttributeError(f"energy units were not set, use 'energy_units = <units>' with <units> one of ('cm-1')") from None

    @energy_units.setter
    def energy_units(self, units):
        if units.lower() in ("cm-1", "cm^-1", "1/cm", "invcm"):
            self.enr_joule = constants.value('Planck constant') * constants.value('speed of light in vacuum') * 1e2
        else:
            raise ValueError(f"unknown energy units: '{units}'") from None

    @property
    def time_units(self):
        try:
            return self.time_sec
        except AttributeError:
            raise AttributeError(f"time units were not set, use 'time_units = <units>' with <units> one of ('ps', 'fs', ns', 'aut')") from None

    @time_units.setter
    def time_units(self, units):
        if units.lower() in ("ps", "picoseconds", "pico"):
            self.time_sec = 1e-12
        elif units.lower() in ("fs", "femtoseconds", "femto"):
            self.time_sec = 1e-15
        elif units.lower() in ("ns", "nanoseconds", "nano"):
            self.time_sec = 1e-9
        elif units.lower() in ("au", "aut"):
            self.time_sec = constants.value("atomic unit of time")
        else:
            raise ValueError(f"unknown time units: '{units}'") from None

    @property
    def tstart(self):
        try:
            return self.t1
        except AttributeError:
            raise AttributeError(f"initial time was not set, use 'tstart = value' to set it") from None

    @tstart.setter
    def tstart(self, val):
        try:
            assert (val < self.t2), f"initial time '{val}' is greater than terminal time '{self.t2}'"
        except AttributeError:
            pass
        self.t1 = val

    @property
    def tend(self):
        try:
            return self.t2
        except AttributeError:
            raise AttributeError(f"terminal time was not set, use 'tend = value' to set it") from None

    @tend.setter
    def tend(self, val):
        try:
            assert (val > self.t1), f"terminal time '{val}' is smaller than initial time '{self.t1}'"
        except AttributeError:
            pass
        self.t2 = val

    @property
    def dt(self):
        try:
            return self.tstep
        except AttributeError:
            raise AttributeError(f"time step was not set, use 'dt = value' to set it") from None

    @dt.setter
    def dt(self, val):
        self.tstep = val





