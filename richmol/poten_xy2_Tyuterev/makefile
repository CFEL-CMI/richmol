all: poten_xy2_tyuterev.so

UNAME := $(shell uname)


ifeq ($(UNAME), Darwin)

######## GFORTRAN, MAC OS ##########
FOR = gfortran
FFLAGS = -O3 -fopenmp -ffree-line-length-0
F77LAGS = -O3
LAPACK =  -lblas -llapack
SOLIB_FLAGS = -dynamiclib
EXT = ".dylib"

else ifeq ($(UNAME), Linux)

######## GFORTRAN, LINUX ###########
FOR = gfortran
FFLAGS = -O3 -fopenmp -fPIC -ffree-line-length-0
F77LAGS = -O3
LAPACK =  -lblas -llapack
SOLIB_FLAGS = -shared
EXT = ".so"

else
$(error Unknown platform)
endif

LIB = $(LAPACK)

poten_xy2_tyuterev.so: poten_xy2_tyuterev.o
	 $(FOR) $(SOLIB_FLAGS) poten_xy2_tyuterev.o $(LIB) -o poten_xy2_tyuterev$(EXT)
poten_xy2_tyuterev.o: poten_xy2_tyuterev.f90
	$(FOR) -c -cpp poten_xy2_tyuterev.f90 $(FFLAGS)
clean:
	$(RM) *.mod *.o *.so *.dylib
