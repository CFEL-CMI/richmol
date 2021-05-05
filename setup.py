import os
import setuptools
from numpy.distutils.core import Extension
from distutils.command.build import build
import subprocess

class Build(build):
    def run(self):
        build.run(self)
        # compile symtoplib
        #command = "cd richmol/symtoplib"
        #command += " && make clean && make"
        #print("\nBuild symtoplib:",command)
        #process = subprocess.Popen(command, shell=True)
        #process.wait()
        # compile wigxjpf
        command = "cd richmol/wigxjpf-1.5"
        command += " && make clean && make shared"
        print("\nBuild wigxjpf-1.5:",command)
        process = subprocess.Popen(command, shell=True)
        process.wait()

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
        #cmdclass            = {'build': Build},
        package_data        = {"richmol":["rot/*","symtoplib/symtoplib*","wigxjpf-1.5/lib/libwigxjpf_shared*"]},
        scripts             = [],
        ext_modules         = [expokit, wigner],
        long_description    = read("README.md"),
)
