import os
import setuptools
from numpy.distutils.core import Extension
from distutils.command.build import build
import subprocess

# expokit set of Fortran functions for computing matrix exponential
expokit = Extension(name = 'expokit',
                     sources = ['expokit/expokit.pyf',
                                'expokit/expokit.f'
                                ],
                     extra_link_args = ["-llapack", "-lblas"]
                     )

# modules for computing Wigner functions and symmetric-top functions
wigner = Extension(name = 'richmol_wigner',
                    sources = ['wigner/symtop.pyf',
                               'wigner/accuracy.f90',
                               'wigner/dffs_m.f90',
                               'wigner/djmk.f90',
                               'wigner/wigner_d.f90',
                               'wigner/symtop.f90',
                               ],
                    extra_link_args = ["-llapack", "-lblas"],
                    extra_compile_args = ["-O3"] 
                    )

# external potential energy functions
potentials = Extension(name = 'potentials',
                    sources = ['potentials/potentials.pyf',
                               'potentials/h2s_tyuterev.f90',
                               ],
                    extra_link_args = ["-llapack", "-lblas"],
                    extra_compile_args = ["-O3"] 
                    )

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), "rt", encoding="utf-8").read()

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        name                = "richmol",
        version             = "0.1a1",
        author              = "Andrey Yachmenev, Cem Saribal, Yahya Saleh, Emil Zak, Linda Thesing",
        author_email        = "andrey.yachmenev@cfel.de",
        description         = ("Richmol is an open-source suit of programs designed for variational"
                               "calculations of molecular rotational-vibrational energy levels,"
                               "wave functions, and subsequent molecular dynamics in the presence"
                               "of internal (hyperfine) and external (laser) electromagnetic fields"),
        license             = "GPL",
        packages            = ["richmol"],
        package_data        = {"richmol":["rot/*"]},
        scripts             = [],
        ext_modules         = [expokit, potentials],
        long_description    = read("README.md"),
)
