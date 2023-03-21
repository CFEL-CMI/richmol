import os
import setuptools
from numpy.distutils.core import Extension

# quad
quad = Extension(name = 'richmol_quad',
                 sources = ['quad/richmol_quad.pyf',
                            'quad/lebedev.f90',
                            'quad/sphere_lebedev_rule.f90'
                            ],
                 extra_compile_args= ['-O3']
                 )

# expokit
expokit = Extension(name = 'expokit',
                    sources = ['expokit/expokit.pyf',
                               'expokit/expokit.f'
                              ],
                    extra_link_args = ["-llapack", "-lblas"],
                    extra_compile_args = ["-O3", "-fallow-argument-mismatch"]
                   )

# computes Wigner and symmetric-top functions
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

# (external) potential energy functions
potentials = Extension(name = 'potentials',
                       sources = ['potentials/potentials.pyf',
                                  'potentials/h2s_tyuterev.f90',
                                 ],
                       extra_link_args = ["-llapack", "-lblas"],
                       extra_compile_args = ["-O3"] 
                      )


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), "rt", encoding="utf-8").read()

__version__ = "1.0rc1"

install_requires = [
    "numpy>=1.19.5",
    "scipy>=1.6",
    "mendeleev>=0.7.0",
    "py3nj>=0.1.2",
    "quadpy>=0.16.7",
    "quaternionic>=0.3.4",
    "spherical>=1.0.8",
    "h5py>=2.10.0",
    "jax>=0.2.13",
    "numba>=0.53.1",
    "matplotlib==3.3.4",
    ]

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        name="richmol",
        version=__version__,
        description="Richmol: a general variational approach to nuclear motion dynamics in fields",
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3.0",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            ],
        author="Richmol team",
        maintainer_email="andrey.yachmenev@cfel.de",
        url="https://github.com/CFEL-CMI/richmol",
        download_url = 'https://pypi.python.org/pypi/richmol',
        packages=setuptools.find_packages(),
        include_package_data=True,
        ext_modules=[expokit, potentials, wigner, quad],
        install_requires=install_requires,
        use_scm_version=True,
        setup_requires=['setuptools_scm'],
)
