! Computes Lebedev quadrature.
!
! Parameters
! ----------
! nponts : c_int
!   Number of quadrature points.
!
! Returns
! -------
! grid(2,npoints) : c_double
!   Contains quadrature points for Euler angles theta=grid(1,ipoint) and phi=grid(2,ipoint).
! weight(npoints) : c_double
!   Contains quadrature weights, note that integration volume element "sin(theta)" is already
!   taken into accout in the quadrature weight.

subroutine angular_lebedev(npoints, grid, weight) bind(c, name='angular_lebedev')

  integer(c_int), intent(in), value :: npoints
  real(c_double), intent(out) :: grid(2,npoints), weight(npoints)

  integer(ik), parameter :: order(0:32) = (/6,6,14,26,38,50,74,86,110,146,170,194,230,266,302,350,&
    434,590,770,974,1202,1454,1730,2030,2354,2702,3074,3470, 3890,4334,4802,5294,5810/)
  integer(ik) :: i
  real(rk) :: pi4, x(npoints), y(npoints), z(npoints)
  integer(ik) :: ipoint
  logical :: match

  pi4 = 4.0_rk * pi

  match = .false.
  do i=1, size(order)
    if (order(i)==npoints) then
      match = .true.
      exit
    endif
  enddo

  if (.not.match) then
    write(out, '(/a,1x,i4,1x,a/1x,a,1x,100(1x,i4))') 'angular_lebedev error: input number of quadrature points =', &
      npoints, 'does not match any of the Lebedev quadratures', 'possible values are:', order
    stop
  endif

  grid = 0
  weight = 0

  call ld_by_order(npoints, x(1:npoints), y(1:npoints), z(1:npoints), weight(1:npoints))

  do ipoint=1, npoints
    call xyz_to_tp(x(ipoint), y(ipoint), z(ipoint), grid(2,ipoint), grid(1,ipoint))
    if (sign(1.0_rk,grid(ipoint,2))<0) then
      grid(2,ipoint) = 2.0_rk*pi + grid(2,ipoint)
    endif
  enddo

  weight(1:npoints) = weight(1:npoints) * pi4 ! note that integration volume element "sin(theta)"
                                              ! is already taken into accout in Lebedev quadrature
                                              ! weight "wght"

end subroutine angular_lebedev
