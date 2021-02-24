Here we create a python wrapper for expokit collection of Fortran functions
for computing matrix exponentials.
The original expokit files are stored in folder 'orig'

1. Generated a signature file using
    python3 -m numpy.f2py expokit.f -m expokit -h expokit.pyf

2. Edited expokit.pyf. Since vector arguments of a user-defined function
matvec in expokit are passed by reference, I had to make modifications in all
matvec calls in expokit.f to pass also dimension of vectors as an extra argument.

3. Generated the extension module by running
    python3 -m numpy.f2py -c expokit.pyf expokit.f blas.f lapack.f