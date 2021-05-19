from scipy import constants
import numpy as np


def MHz_to_invcm(*args):
    """Converts `MHz` to :math:`cm^{-1}`"""
    fac = 1/constants.value('speed of light in vacuum') * 1e4
    return convert(fac, *args)


def Debye_to_au(*args):
    """Converts dipole moment from `Debye` to atomic units"""
    fac = 1e-21 / constants.value('speed of light in vacuum') \
        / constants.value('elementary charge') \
        / constants.value('Bohr radius')
    return convert(fac, *args)


def Debye_to_sqrt_erg_x_sqrt_cm3(*args):
    """Converts dipole moment from `Debye` to `erg^(1/2)*cm^(3/2)`"""
    fac = 1e-18
    return convert(fac, *args)


def Debye_x_Vm_to_invcm(*args):
    """Converts product of dipole moment (in `Debye`) with field (in `Volts/meter`) to :math:`cm^{-1}`"""
    fac = constants.value('atomic unit of electric dipole mom.') \
        / (constants.value('Planck constant') \
        * constants.value('speed of light in vacuum')) \
        / 1e2 \
        * Debye_to_au()
    return convert(fac, *args)


def Buckingham_to_au(*args):
    """Converts quadrupole moment from `Buckingham` to atomic units"""
    fac = Debye_to_au() * constants.value('angstrom') \
        / constants.value('Bohr radius')
    return convert(fac, *args)


def Buckingham_to_sqrt_erg_x_sqrt_cm5(*args):
    """Converts quadrupole moment from `Buckinghom` to `erg^(1/2)*cm^(5/2)`"""
    fac = Debye_to_sqrt_erg_x_sqrt_cm3() * constants.value('angstrom')
    return convert(fac, *args)


def AUdip_x_Vm_to_invcm(*args):
    """Converts product of dipole moment (in atomic units) with field (in `Volts/meter`) to :math:`cm^{-1}`"""
    fac = constants.value('atomic unit of electric dipole mom.') \
        / (constants.value('Planck constant') \
        * constants.value('speed of light in vacuum')) \
        / 1e2
    return convert(fac, *args)


def AUpol_x_Vm_to_invcm(*args):
    """Converts product of polarizability (in atomic units) with field (in :math:`Volts^2/meter^2`) to :math:`cm^{-1}`"""
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
