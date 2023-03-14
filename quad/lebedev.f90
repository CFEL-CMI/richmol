! Returns number of points in Lebedev quadrature given the quadrature order.
!
! Parameters
! ----------
! order : c_int
!   Quadrature order, an integer number between 0 and max_leb_order(=32)
!
! Returns
! -------
! npoints : c_int
!   Number of quadrature points

subroutine lebedev_npoints(order, npoints)
  implicit none
  integer(4), intent(in) :: order
  integer(4), intent(out) :: npoints
  integer(4), parameter :: max_leb_order = 32
  integer(4), parameter :: leb_npoints(0:max_leb_order) = &
    (/6,6,14,26,38,50,74,86,110,146,170,194,230,266,302,350,434,590,770,974,&
    1202,1454,1730,2030,2354,2702,3074,3470, 3890,4334,4802,5294,5810/)
  if (order < 0 .or. order > max_leb_order) then
    write(*, '(/a,1x,i4,1x,a,1x,i4)') &
    'lebedev_npoints error: argument `order` =', order, &
    '< 0 or > max_leb_order =', max_leb_order
    stop
  endif
  npoints = leb_npoints(order)
end subroutine lebedev_npoints

! Computes Lebedev quadrature.
!
! Parameters
! ----------
! npoints : c_int
!   Number of quadrature points.
!
! Returns
! -------
! grid(2,npoints) : c_double
!   Contains quadrature points for Euler angles theta=grid(1,ipoint) and phi=grid(2,ipoint).
! weight(npoints) : c_double
!   Contains quadrature weights, note that integration volume element "sin(theta)"
!   is already taken into account in the quadrature weight.

subroutine lebedev(npoints, grid, weight)
  implicit none

  integer(4), intent(in) :: npoints
  real(8), intent(out) :: grid(2, npoints), weight(npoints)

  integer(4) :: ipoint
  real(8) :: x(npoints), y(npoints), z(npoints), pi4
  real(8), parameter :: pi = 3.1415926535897932384626433832795d0

  pi4 = 4.0d0 * pi

  grid = 0
  weight = 0

  call ld_by_order(npoints, x(1:npoints), y(1:npoints), z(1:npoints), weight(1:npoints))

  do ipoint=1, npoints
    call xyz_to_tp(x(ipoint), y(ipoint), z(ipoint), grid(2, ipoint), grid(1, ipoint))
  enddo

  weight(1:npoints) = weight(1:npoints) * pi4 ! note that integration volume element "sin(theta)" is already taken into account in Lebedev quadrature weight

end subroutine lebedev

