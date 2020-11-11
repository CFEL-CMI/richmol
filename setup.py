import os
from setuptools import setup
from distutils.command.build import build
import subprocess

class Build(build):
    def run(self):
        build.run(self)
        # compile symtoplib
        command = "cd richmol/symtoplib"
        command += " && make clean && make"
        print("\nBuild symtoplib:",command)
        process = subprocess.Popen(command, shell=True)
        process.wait()
        # compile wigxjpf
        command = "cd richmol/wigxjpf-1.5"
        command += " && make clean && make shared"
        print("\nBuild wigxjpf-1.5:",command)
        process = subprocess.Popen(command, shell=True)
        process.wait()
        # compile blas
        #command = "cd richmol/BLAS-3.8.0"
        #command += " && make clean && make"
        #print("\nBuild blas-3.8.0:",command)
        #process = subprocess.Popen(command, shell=True)
        #process.wait()
        # compile lapack
        #command = "cd richmol/lapack-3.9.0"
        #command += " && make clean && make"
        #print("\nBuild lapack-3.9.0:",command)
        #process = subprocess.Popen(command, shell=True)
        #process.wait()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), "rt", encoding="utf-8").read()

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
    cmdclass            = {'build': Build},
    package_data        = {"richmol":["symtoplib/symtoplib*","wigxjpf-1.5/lib/libwigxjpf_shared*"]},
    scripts             = [],
    install_requires    = ["hypothesis", "mendeleev"],
    long_description    = read("README.md"),
)

