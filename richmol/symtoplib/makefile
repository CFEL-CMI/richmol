goal: symtoplib

UNAME := $(shell uname)


ifeq ($(UNAME), Darwin)

SO_EXT = dylib

######## Intel Fortran, MAC OS ##########
#FOR = ifort
#FFLAGS = -O3 -qopenmp
#F77LAGS = -O3
#CPPFLAGS = -D_WIGD_FOURIER_
##CPPFLAGS = -D_WIGD_FOURIER_BIGJ_
#LAPACK =  -mkl=parallel
#SOLIB_FLAGS = -dynamiclib

######## GFORTRAN, MAC OS ##########
FOR = gfortran
FFLAGS = -O3 -fopenmp -ffree-line-length-0
F77LAGS = -O3
#CPPFLAGS = -D_WIGD_FOURIER_
#CPPFLAGS = -D_WIGD_FOURIER_BIGJ_
LAPACK =  -lblas -llapack
SOLIB_FLAGS = -dynamiclib


else ifeq ($(UNAME), Linux)

SO_EXT = so

######## Intel Fortran, LINUX ###########
FOR = ifort
FFLAGS = -O3 -fpic -qopenmp
F77LAGS = -O3 -fpic
#CPPFLAGS = -D_WIGD_FOURIER_
#CPPFLAGS = -D_WIGD_FOURIER_BIGJ_
LAPACK =  -mkl=parallel
SOLIB_FLAGS = -shared

else
$(error Unknown platform)
endif


LIB = $(LAPACK) wigxjpf-1.5/lib/libwigxjpf.a


OBJ = symtop.o accuracy.o dffs_m.o wigner_d.o fwigxjpf.o


symtoplib: $(OBJ)
	$(FOR) $(SOLIB_FLAGS) $(LIB) $(OBJ) -o symtoplib.$(SO_EXT)

symtop.o: symtop.f90 $(OBJ)
	$(FOR) -c -cpp symtop.f90 $(FFLAGS) $(CPPFLAGS)

accuracy.o: accuracy.f90
	$(FOR) -c -cpp accuracy.f90 $(FFLAGS) $(CPPFLAGS)

fwigxjpf.o: fwigxjpf.f90
	$(FOR) -c fwigxjpf.f90 $(FFLAGS) $(CPPFLAGS)
	cd wigxjpf-1.5  && $(MAKE) shared

dffs_m.o: wigner_dmat/dffs_m.f90
	$(FOR) -c wigner_dmat/dffs_m.f90 $(FFLAGS) $(CPPFLAGS)

wigner_d.o: wigner_dmat2/wigner_d.f90
	$(FOR) -c wigner_dmat2/wigner_d.f90 $(FFLAGS) $(CPPFLAGS)

clean:
	rm $(OBJ) *.mod *.so *.dylib
	cd wigxjpf-1.5  && $(MAKE) clean
