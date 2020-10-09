Installation
************

Manual installation from github repo
====================================
Manual installation requires
`numpy <http://www.numpy.org/>`_,
`scipy <http://www.scipy.org/>`_,
`h5py <http://www.h5py.org/>`_,
`hypothesis <https://hypothesis.readthedocs.io/en/latest/>`_, and
`mendeleev <https://pypi.org/project/mendeleev/>`_
libraries, as well as Fortran (gfortran, Intel Fortran) and C (gcc) compilers.
You can download the latest Richmol (or the development branch) from github::

  $ git clone https://github.com/CFEL-CMI/richmol
  $ cd richmol
  $ git checkout develop  # optional if you'd like to try out the development branch

Build and install the project::

  $ python3 setup.py build
  $ python3 setup.py install --user

To ensure the installation is successful, start a Python3 shell, and type::

  >>> import richmol

Using optimized libraries
=========================
The default installation tries to find the mathematical libraries such as BLAS and LAPACK
automatically (not yet implemented). You can compile the package with other BLAS and LAPACK vendors,
such as, for example the Intel Math kernel Library (MKL)::

  $ need to set this up soon
