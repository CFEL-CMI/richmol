import sympy as sp
import numpy as np
import math
from numpy.polynomial.legendre import leggauss, legval, legder
from numpy.polynomial.hermite import hermgauss, hermval, hermder
import scipy.special as ss
#import scipy.special.eval_genlaguerre
#import scipy.special.roots_genlaguerre as laggauss
import sys
singular_tol = 1e-10 # tolerance for considering matrix singular
symmetric_tol = 1e-10 # tolerance for considering matrix symmetric
