import os
from setuptools import setup
from distutils.command.build import build
import subprocess

class Build(build):
    def run(self):
        build.run(self)
        pass

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), "rt", encoding="utf-8").read()

setup(
    name                = "richmol",
    version             = "",
    author              = "",
    author_email        = "",
    description         = (""),
    license             = "TBA",
    packages            = ["richmol"],
    cmdclass            = {'build': Build},
    package_data        = {"richmol":[]},
    scripts             = [],
    long_description    = read("README"),
)

