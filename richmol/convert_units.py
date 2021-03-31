from scipy import constants
import numpy as np


def MHz_to_invcm(*args):
    """Converts MHz to cm^-1"""
    fac = 1/constants.value('speed of light in vacuum') * 1e4
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
