module symtop

  use accuracy
  use iso_c_binding
  use dffs_m
  use wigner_d
  implicit none

  contains

#include "symtop_3d_grid.f90"
#include "symtop_2d_grid.f90"
#include "angular_lebedev.f90"
#include "sphere_lebedev_rule.f90"
#include "djmk_small.f90"

end module symtop