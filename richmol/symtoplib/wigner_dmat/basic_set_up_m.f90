! last modifed on 2014/10/31(y/m/d)
module basic_set_up_m
  use ISO_FORTRAN_ENV, only:stdin=>INPUT_UNIT,&
&                           stdout=>OUTPUT_UNIT,&
&                           stderr=>ERROR_UNIT
  implicit none
! integer,parameter::stdin=5,stdout=6,stderr=0
  integer,parameter::i4=selected_int_kind(r=9) ! 4-byte integer
  integer,parameter::i8=selected_int_kind(r=18) ! 8-byte integer
  integer,parameter::r4=selected_real_kind(p=6,r=37) ! 4-byte real number
  integer,parameter::r8=selected_real_kind(p=15,r=307) ! 8-byte real number
  integer,parameter::r16=selected_real_kind(p=33,r=4931) ! 16-byte real number
! integer,parameter::i4=4,i8=8,r4=4,r8=8,r16=16
  integer,parameter::wp=r8 ! working precision real floating-point numbers
  integer,parameter::cl=256 ! default sufficiently length for character strings
end module basic_set_up_m
