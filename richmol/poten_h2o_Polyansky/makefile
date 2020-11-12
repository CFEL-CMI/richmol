all: rsta20170149supp1.so

UNAME := $(shell uname)


ifeq ($(UNAME), Darwin)

######## Intel Fortran, MAC OS ##########
<<<<<<< HEAD
FOR = ifort
FFLAGS = -O3 -qopenmp
F77LAGS = -O3
LAPACK =  -mkl=parallel
SOLIB_FLAGS = -dynamiclib

######## GFORTRAN, MAC OS ##########
#FOR = gfortran
#FFLAGS = -O3 -fopenmp -ffree-line-length-0
#F77LAGS = -O3
#LAPACK =  -lblas -llapack
#SOLIB_FLAGS = -dynamiclib

=======
#FOR = ifort
#FFLAGS = -O3 -qopenmp
#F77LAGS = -O3
#LAPACK =  -mkl=parallel
#SOLIB_FLAGS = -dynamiclib

######## GFORTRAN, MAC OS ##########
FOR = gfortran
FFLAGS = -O3 -fopenmp -ffree-line-length-0
F77LAGS = -O3
LAPACK =  -lblas -llapack
SOLIB_FLAGS = -dynamiclib

>>>>>>> f1b7e5d0f7f73792128808fa59f83e55d9ca7a21
else ifeq ($(UNAME), Linux)

######## Intel Fortran, LINUX ###########
FOR = ifort
FFLAGS = -O3 -fpic -qopenmp
F77LAGS = -O3 -fpic
LAPACK =  -mkl=parallel
SOLIB_FLAGS = -shared

else
$(error Unknown platform)
endif

<<<<<<< HEAD
LIB = 
=======
LIB =
>>>>>>> f1b7e5d0f7f73792128808fa59f83e55d9ca7a21


# Polyansky potential for water molecule, with parameters embedded inthe source code
# see http://rsta.royalsocietypublishing.org/content/376/2115/20170149
rsta20170149supp1.so: rsta20170149supp1.o
	$(FOR) $(SOLIB_FLAGS) $(LIB) rsta20170149supp1.o -o rsta20170149supp1.so
rsta20170149supp1.o: rsta20170149supp1.f
	$(FOR) -c -cpp rsta20170149supp1.f $(FFLAGS)
