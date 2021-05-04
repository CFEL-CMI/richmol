from scipy import constants
import numpy as np


def MHz_to_invcm(*args):
    """Converts MHz to cm^-1"""
    fac = 1/constants.value('speed of light in vacuum') * 1e4
    return convert(fac, *args)


def Debye_to_au(*args):
    """Converts dipole moment from Debye to atomic units"""
    fac = 0.393456
    return convert(fac, *args)


def Debye_x_Vm_to_invcm(*args):
    """Converts product of dipole moment (Debye) with field (Volts/meter) to cm^-1"""
    fac = constants.value('atomic unit of electric dipole mom.') \
        / (constants.value('Planck constant') \
        * constants.value('speed of light in vacuum')) \
        / 1e2 \
        * Debye_to_au()
    return convert(fac, *args)


def AUdip_x_Vm_to_invcm(*args):
    """Converts product of dipole moment (atomic units) with field (Volts/meter) to cm^-1"""
    fac = constants.value('atomic unit of electric dipole mom.') \
        / (constants.value('Planck constant') \
        * constants.value('speed of light in vacuum')) \
        / 1e2
    return convert(fac, *args)


def AUpol_x_Vm_to_invcm(*args):
    """Converts product of polarizability (atomic units) with field**2 (Volts**2/meter**2) to cm^-1"""
    fac = constants.value('atomic unit of electric polarizability') \
        / (constants.value('Planck constant') \
        * constants.value('speed of light in vacuum')) \
        / 1e2
    return convert(fac, *args)


def convert(fac, *args):
    if len(args) == 0:
        return fac
    else:
        res = []
        for arg in args:
            try:
                res.append([elem * fac for elem in arg])
            except TypeError:
                res.append(arg * fac)
        return tuple(res)
