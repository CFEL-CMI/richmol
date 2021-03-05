module accuracy
  implicit none

  integer, parameter :: ik   = selected_int_kind(8)
  integer, parameter :: hik  = selected_int_kind(16)
  integer, parameter :: rk   = selected_real_kind(12,25)
  integer, parameter :: ark  = selected_real_kind(25,32)
  integer, parameter :: inp  = 5
  integer, parameter :: out  = 6
  integer, parameter :: cl   = 80
  real(rk), parameter :: pi = 3.1415926535897932384626433832795d0

end module accuracy
