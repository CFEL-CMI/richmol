all: rsta20170149supp1.so

UNAME := $(shell uname)


ifeq ($(UNAME), Darwin)

######## Intel Fortran, MAC OS ##########
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

else ifeq ($(UNAME), Linux)

######## Intel Fortran, LINUX ###########
#FOR = ifort
#FFLAGS = -O3 -fpic -qopenmp
#F77LAGS = -O3 -fpic
#LAPACK =  -mkl=parallel
#SOLIB_FLAGS = -shared

######## Gfortran, LINUX ###########
FOR = gfortran
FFLAGS = -O3 -fopenmp -fPIC -ffree-line-length-0
F77LAGS = -O3
LAPACK =  -lblas -llapack
SOLIB_FLAGS = -shared


else
$(error Unknown platform)
endif

LIB = $(LAPACK)


# Polyansky potential for water molecule, with parameters embedded inthe source code
# see http://rsta.royalsocietypublishing.org/content/376/2115/20170149
rsta20170149supp1.so: rsta20170149supp1.o
	$(FOR) $(SOLIB_FLAGS) rsta20170149supp1.o $(LIB) -o rsta20170149supp1.so
rsta20170149supp1.o: rsta20170149supp1.f
	$(FOR) -c -cpp rsta20170149supp1.f $(FFLAGS)
