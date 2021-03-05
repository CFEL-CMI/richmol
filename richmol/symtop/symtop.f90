module symtop
  use accuracy
  use djmk
  use dffs_m
  use wigner_d
  implicit none

  contains

  ! Computes symmetric-top functions |J,k,m> on a three-dimensional grid of Euler angles.
  !
  ! Parameters
  ! ----------
  ! npoints : integer
  !   Number of grid points.
  ! Jmin, Jmax : integer
  !   Min and max value of J quantum number spanned by the list of symmetric-top functions.
  ! grid(3,npoints) : double precision
  !   3D grid of different values of Euler angles, grid(1:3,ipoint) = (/phi,theta,chi/),
  !   where "chi" and "phi" are Euler angles associated with "k" and "m" quantum numbers, respectively.
  !
  ! Returns
  ! -------
  ! val(npoints,-Jmax:Jmax,-Jmax:Jmax,Jmin:Jmax) : complex(16)
  !   Values of symmetric-top functions on grid, |j,k,m> = val(ipoint,m,k,j), where ipoint=1..npoints

  ! There are three versions of the code, each using different approach to compute small-d matrix

  subroutine threed_grid(npoints, Jmin, Jmax, grid, val)

    integer(ik), intent(in) :: npoints
    real(rk), intent(in) :: grid(3,npoints)
    integer(ik), intent(in) :: Jmin, Jmax
    complex(rk), intent(out) :: val(npoints,-Jmax:Jmax,-Jmax:Jmax,Jmin:Jmax)

    integer(ik) :: info, j, m, k, ipoint, iounit
    real(rk) :: fac
    complex(rk) :: one_imag, res

    one_imag = cmplx(0.0_rk, 1.0_rk)

    val = 0

    do j=Jmin, Jmax

      fac = sqrt(real(2*j+1,rk)/(8.0_rk*pi**2))

      do m=-j, j
        do k=-j, j
          do ipoint=1, npoints
            res = djmk_small(real(j,rk), real(m,rk), real(k,rk), grid(2,ipoint)) &
                * exp( one_imag * k * grid(3,ipoint) ) &
                * exp( one_imag * m * grid(1,ipoint) )
            val(ipoint,m,k,j) = res * fac
          enddo ! ipoint
        enddo ! k
      enddo ! m

    enddo ! j

  end subroutine threed_grid

  subroutine threed_grid_dffs(npoints, Jmin, Jmax, grid, val)

    integer(ik), intent(in) :: npoints
    real(rk), intent(in) :: grid(3,npoints)
    integer(ik), intent(in) :: Jmin, Jmax
    complex(rk), intent(out) :: val(npoints,-Jmax:Jmax,-Jmax:Jmax,Jmin:Jmax)

    integer(ik) :: info, j, m, k, ipoint, iounit
    real(rk) :: fac
    complex(rk) :: one_imag, res

    one_imag = cmplx(0.0_rk, 1.0_rk)

    ! initialize some data required for computing Wigner D-matrix
    call dffs_read_coef(Jmax*2)

    val = 0

    do j=Jmin, Jmax

      fac = sqrt(real(2*j+1,rk)/(8.0_rk*pi**2))

      do m=-j, j
        do k=-j, j
          do ipoint=1, npoints
            res = dffs( j*2, m*2, k*2, grid(2,ipoint) ) &
                * exp( one_imag * k * grid(3,ipoint) ) &
                * exp( one_imag * m * grid(1,ipoint) )
            val(ipoint,m,k,j) = res * fac
          enddo ! ipoint
        enddo ! k
      enddo ! m

    enddo ! j

  end subroutine threed_grid_dffs

  subroutine threed_grid_wigd(npoints, Jmin, Jmax, grid, val)

    integer(ik), intent(in) :: npoints
    real(rk), intent(in) :: grid(3,npoints)
    integer(ik), intent(in) :: Jmin, Jmax
    complex(rk), intent(out) :: val(npoints,-Jmax:Jmax,-Jmax:Jmax,Jmin:Jmax)

    integer(ik) :: info, j, m, k, ipoint, iounit
    real(rk), allocatable :: wd_matrix(:,:,:), diffwd_matrix(:,:)
    real(rk) :: fac
    complex(rk) :: one_imag, res

    one_imag = cmplx(0.0_rk, 1.0_rk)

    ! initialize some data required for computing Wigner D-matrix
    allocate(wd_matrix(npoints,2*Jmax+1,2*Jmax+1), diffwd_matrix(2*Jmax+1,2*Jmax+1), stat=info)
    if (info/=0) then
      write(out, '(/a/a,10(1x,i6))') &
        'threed_grid_wigd error: failed to allocate wd_matrix(npoints,2*Jmax+1,2*Jmax+1), &
        diffwd_matrix(2*Jmax+1,2*Jmax+1)', 'npoints, Jmax =', npoints, Jmax
      stop
    endif

    val = 0

    do j=Jmin, Jmax

      fac = sqrt(real(2*j+1,rk)/(8.0_rk*pi**2))

      do ipoint=1, npoints
        call Wigner_dmatrix(real(j,rk), grid(2,ipoint), wd_matrix(ipoint,1:2*j+1,1:2*j+1), &
          diffwd_matrix(1:2*j+1,1:2*j+1))
      enddo

      do m=-j, j
        do k=-j, j
          do ipoint=1, npoints
            res = wd_matrix(ipoint,j+m+1,j+k+1) &
                * exp( one_imag * k * grid(3,ipoint) ) &
                * exp( one_imag * m * grid(1,ipoint) )
            val(ipoint,m,k,j) = res * fac
          enddo ! ipoint
        enddo ! k
      enddo ! m

    enddo ! j

    deallocate(wd_matrix, diffwd_matrix)

  end subroutine threed_grid_wigd

end module symtop
