import numpy as np
import functools

"""
# rough idea

tdse.tstart = 0
tdse.tend = 100
tdse.time_units = "ps"
tdse.energy_units = "1/cm"
tdse.dt = 0.01

dc = lambda t: [0, 0, t * 1000]
ac = lambda t: [0, 0, 1e+12 * np.cos(10000 * t)]

vec = .... # initial vector

for time in tdse.times():

    mu.field(-dc(time))
    alpha.field(-1/2 * ac(time))
    H = mu + alpha

    tdse.update(vec, H, H0) # use split
    tdse.update(vec, H+H0) # no split

"""

def times(grid="equidistant"):
    if grid.lower() == "equidistant":
        npt = int((self.tend - self.tstart) / self.dt)
        t1 = np.array([self.tstart + self.dt * (i) for i in range(npt+1)])
        t2 = np.array([self.tstart + self.dt * (i+1) for i in range(npt+1)])
        tc = 0.5*(t1 + t2)
    return tc


def update(v, H, H0=None):
    # this factor should make (dt/hbar * H) dimensionless
    exp_fac = -1j * self.dt * self.time_units * constants.value("reduced Planck constant") * self.energy_units
    #
    # essentially move CarTens.U here
    # for two options, split and full Hamiltonian
    #
    pass


@property
def energy_units():
    try:
        return self.enr_joule
    except AttributeError:
        raise AttributeError(f"energy units were not set, use 'energy_units = <units>' with <units> one of ('cm-1')") from None

@energy_units.setter
def energy_units(units):
    if units.lower() in ("cm-1", "cm^-1", "1/cm", "invcm"):
        self.enr_joule = (constants.value('Planck constant') / constants.value('speed of light in vacuum')) * 1e2
    else:
        raise ValueError(f"unknown energy units: '{units}'") from None


@property
def time_units():
    try:
        return self.time_sec
    except AttributeError:
        raise AttributeError(f"time units were not set, use 'time_units = <units>' with <units> one of ('ps', 'fs', ns', 'aut')") from None


@time_units.setter
def time_units(units):
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
def tstart():
    try:
        return self.t1
    except AttributeError:
        raise AttributeError(f"initial time was not set, use 'tstart = value' to set it") from None

@tstart.setter
def tstart(val):
    try:
        assert (val > self.t2), f"initial time '{val}' is greater than terminal time '{self.t2}'"
    except AttributeError:
        pass
    self.t1 = val


@property
def tend():
    try:
        return self.t2
    except AttributeError:
        raise AttributeError(f"terminal time was not set, use 'tend = value' to set it") from None

@tend.setter
def tend(val):
    try:
        assert (val < self.t1), f"terminal time '{val}' is smaller than initial time '{self.t1}'"
    except AttributeError:
        pass
    self.t2 = val


@property
def dt():
    try:
        return self.tstep
    except AttributeError:
        raise AttributeError(f"time step was not set, use 'dt = value' to set it") from None

@dt.setter
def dt(val):
    self.tstep = val

