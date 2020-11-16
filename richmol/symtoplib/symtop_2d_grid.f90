! Computes symmetric-top functions |J,k,m> on a two-dimensional grid of Euler angles for selected value of m
!                                                                                    -----------------------
! Parameters
! ----------
! npoints : integer
!   Number of grid points.
! Jmin, Jmax : integer
!   Min and max value of J quantum number spanned by the list of symmetric-top functions.
! m : integer
!   Value of m quantum number.
! grid(2,npoints) : c_double
!   2D grid of different values of Euler angles, grid(1:2,ipoint) = (/theta,chi/),
!   where "chi" is the Euler angles associated with the "k" quantum number.
!
! Returns
! -------
! val_r(npoints,-Jmax:Jmax,Jmin:Jmax), val_i(npoints,-Jmax:Jmax,Jmin:Jmax) : c_double
!   Values, real and imaginary parts, of symmetric-top functions on grid,
!   |j,k,m> = val(ipoint,k,j), where ipoint=1..npoints

subroutine symtop_2d_grid_m(npoints, Jmin, Jmax, m, grid, val_r, val_i) bind(c, name='symtop_2d_grid_m')

  integer(c_int), intent(in), value :: npoints
  real(c_double), intent(in) :: grid(2,npoints)
  integer(c_int), intent(in), value :: Jmin, Jmax
  integer(c_int), intent(in), value :: m
  real(c_double), intent(out) :: val_r(npoints,-Jmax:Jmax,Jmin:Jmax), &
                                 val_i(npoints,-Jmax:Jmax,Jmin:Jmax)
  
  integer(ik) :: info, j, k, ipoint, iounit
  real(rk) :: fac
  real(rk), allocatable :: wd_matrix(:,:,:), diffwd_matrix(:,:)
  complex(rk) :: one_imag, res, val(npoints,-Jmax:Jmax,Jmin:Jmax)
  character(cl) :: sj, sk, sm, fname
  logical :: iftest
  
  one_imag = cmplx(0.0_rk,1.0_rk)
  
  ! initialize some data required for computing Wigner D-matrix
  
#if defined(_WIGD_FOURIER_)

  call dffs_read_coef(Jmax*2)

#elif defined(_WIGD_FOURIER_BIGJ_)

  allocate(wd_matrix(npoints,2*Jmax+1,2*Jmax+1), diffwd_matrix(2*Jmax+1,2*Jmax+1), stat=info)
  if (info/=0) then
    write(out, '(/a/a,10(1x,i6))') &
      'symtop_2d_grid_m error: failed to allocate wd_matrix(npoints,2*Jmax+1,2*Jmax+1), &
      diffwd_matrix(2*Jmax+1,2*Jmax+1)', 'npoints, Jmax =', npoints, Jmax
    stop
  endif

#endif
  
  ! start computing values of symmetric-top functions on grid for all values of J=Jmin..Jmax

  val = 0

  do j=Jmin, Jmax

    fac = sqrt(real(2*j+1,rk)/(8.0_rk*pi**2))

#if defined(_WIGD_FOURIER_BIGJ_)
    do ipoint=1, npoints
      call Wigner_dmatrix(real(j,rk), grid(1,ipoint), wd_matrix(ipoint,1:2*j+1,1:2*j+1), &
                          diffwd_matrix(1:2*j+1,1:2*j+1))
    enddo
#endif
  
    do k=-j, j
      do ipoint=1, npoints
#if defined(_WIGD_FOURIER_)
        res = dffs( j*2, m*2, k*2, grid(1,ipoint) ) &
            * exp( one_imag * k * grid(2,ipoint) )
#elif defined(_WIGD_FOURIER_BIGJ_)
        res = wd_matrix(ipoint,j+m+1,j+k+1) &
            * exp( one_imag * k * grid(2,ipoint) )
#else
        res = djmk_small(real(j,rk), real(m,rk), real(k,rk), grid(1,ipoint)) &
            * exp( one_imag * k * grid(2,ipoint) )
#endif
        val(ipoint,k,j) = res * fac
      enddo ! ipoint
    enddo ! k

  enddo ! j

#if defined(_WIGD_FOURIER_BIGJ_)
  deallocate(wd_matrix, diffwd_matrix)
#endif

  val_r = real(val, kind=rk)
  val_i = aimag(val)

end subroutine symtop_2d_grid_m


! Computes symmetric-top functions |J,k=0,m> on a two-dimensional grid of Euler angles theta and phi.
!
! Parameters
! ----------
! npoints : integer
!   Number of grid points.
! grid(2,npoints) : c_double
!   2D grid of different values of Euler angles, grid(1:2,ipoint) = (/theta,phi/),
!   where "phi" is the Euler angle associated with "m" quantum number.
! Jmax : integer
!   Max value of J quantum number spanned by the list of symmetric-top functions.
!
! Returns
! -------
! val_r(-Jmax:Jmax,0:Jmax,npoints) and val_i(-Jmax:Jmax,0:Jmax,npoints) : c_double
!   Values, real and imaginary parts, of symmetric-top functions on grid,
!   |j,k=0,m> = val(m,j,ipoint), where ipoint=1..npoints

subroutine symtop_2d_grid_theta_phi(npoints, Jmax, grid, val_r, val_i) &
    bind(c, name='symtop_2d_grid_theta_phi')

  integer(c_int), intent(in), value :: npoints
  real(c_double), intent(in) :: grid(2,npoints)
  integer(c_int), intent(in), value :: Jmax
  real(c_double), intent(out) :: val_r(-Jmax:Jmax,0:Jmax,npoints), val_i(-Jmax:Jmax,0:Jmax,npoints)

  integer(ik) :: info, j, m, k, ipoint, iounit
  real(rk) :: fac
  real(rk), allocatable :: wd_matrix(:,:,:), diffwd_matrix(:,:)
  complex(rk) :: one_imag, res, val(-Jmax:Jmax,0:Jmax,npoints)
  character(cl) :: sj, sk, sm, fname
  logical :: iftest

  !write(out, '(a)') 'symtop_2d_grid_theta_phi: compute symmetric-top functions on a 2D grid of theta and phi Euler angles'
  
  one_imag = cmplx(0.0_rk,1.0_rk)
    
  !write(out, '(a,1x,i4,1x,i8)') 'symtop_2d_grid_theta_phi: Jmax, npoints =', Jmax, npoints
    
  ! initialize some data required for computing Wigner D-matrix
    
#if defined(_WIGD_FOURIER_)
  
  !write(out, '(a)') 'symtop_2d_grid_theta_phi: use dffs_m module for Wigner D-matrix'
  call dffs_read_coef(Jmax*2)
  
#elif defined(_WIGD_FOURIER_BIGJ_)
  
  !write(out, '(a)') 'symtop_2d_grid_theta_phi: use wigner_d module for Wigner D-matrix'

  ! allocate matrices for computing Wigner small-d matrices using module wigner_dmat2/wigner_d.f90

  allocate(wd_matrix(npoints,2*Jmax+1,2*Jmax+1), diffwd_matrix(2*Jmax+1,2*Jmax+1), stat=info)
  if (info/=0) then
    write(out, '(/a/a,10(1x,i6))') &
      'symtop_2d_grid_theta_phi error: failed to allocate wd_matrix(npoints,2*Jmax+1,2*Jmax+1), &
      diffwd_matrix(2*Jmax+1,2*Jmax+1)', 'npoints, Jmax =', npoints, Jmax
    stop
  endif
  
#else

  !write(out, '(a)') 'symtop_2d_grid_theta_phi: use slow djmk_small routine to compute Wigner D-matrix'

#endif

  ! start computing values of symmetric-top functions on grid for all values of J=0..Jmax

  val = 0
  
  do j=0, Jmax
    
    !write(out, '(a,1x,i4)') 'symtop_2d_grid_theta_phi: J =', j
    
    fac = sqrt(real(2*j+1,rk)/(8.0_rk*pi**2))
  
#if defined(_WIGD_FOURIER_BIGJ_)
    do ipoint=1, npoints
      call Wigner_dmatrix(real(j,rk), grid(1,ipoint), wd_matrix(ipoint,1:2*j+1,1:2*j+1), &
                          diffwd_matrix(1:2*j+1,1:2*j+1))
    enddo
#endif
    
    k = 0
    do m=-j, j
      do ipoint=1, npoints
#if defined(_WIGD_FOURIER_)
        res = dffs( j*2, m*2, k*2, grid(1,ipoint) ) * exp(one_imag * m * grid(2,ipoint))
#elif defined(_WIGD_FOURIER_BIGJ_)
        res = wd_matrix(ipoint,j+m+1,j+k+1) * exp(one_imag * m * grid(2,ipoint))
#else
        res = djmk_small(real(j,rk), real(m,rk), real(k,rk), grid(1,ipoint)) &
            * exp(one_imag * m * grid(2,ipoint))
#endif
        val(m,j,ipoint) = res * fac
      enddo ! ipoint
    enddo ! m
  
  enddo ! j
  
#if defined(_WIGD_FOURIER_BIGJ_)
  deallocate(wd_matrix, diffwd_matrix)
#endif

  val_r = real(val, rk)
  val_i = aimag(val)

  !write(out, '(a)') 'symtop_2d_grid_theta_phi: done'

  ! for testing, print values of primitive functions into ASCII files
  
  iftest = .false.
  if (iftest) then
    do j=0, Jmax
      write(sj,*) j
      do m=-j, j
        write(sm,*) m
        fname = 'symtop_func_j'//trim(adjustl(sj))//'_m'//trim(adjustl(sm))
        open(iounit,form='formatted',position='rewind',action='write',file=fname)
        do ipoint=1, npoints
          write(iounit,'(3(1x,es16.8),3x,2(1x,es16.8))') grid(1:2,ipoint), val(m,j,ipoint)
        enddo
        close(iounit)
      enddo
    enddo
  endif

end subroutine symtop_2d_grid_theta_phi



! Computes symmetric-top functions |J,k,m=0> on a two-dimensional grid of Euler angles theta and chi.
!
! Parameters
! ----------
! npoints : integer
!   Number of grid points.
! grid(2,npoints) : c_double
!   2D grid of different values of Euler angles, grid(1:2,ipoint) = (/theta,chi/),
!   where "chi" is the Euler angle associated with "k" quantum number.
! Jmax : integer
!   Max value of J quantum number spanned by the list of symmetric-top functions.
!
! Returns
! -------
! val_r(-Jmax:Jmax,0:Jmax,npoints) and val_i(-Jmax:Jmax,0:Jmax,npoints) : c_double
!   Values, real and imaginary parts, of symmetric-top functions on grid,
!   |j,k,m=0> = val(k,j,ipoint), where ipoint=1..npoints

subroutine symtop_2d_grid_theta_chi(npoints, Jmax, grid, val_r, val_i) &
    bind(c, name='symtop_2d_grid_theta_chi')

  integer(c_int), intent(in), value :: npoints
  real(c_double), intent(in) :: grid(2,npoints)
  integer(c_int), intent(in), value :: Jmax
  real(c_double), intent(out) :: val_r(-Jmax:Jmax,0:Jmax,npoints), val_i(-Jmax:Jmax,0:Jmax,npoints)

  integer(ik) :: info, j, m, k, ipoint, iounit
  real(rk) :: fac
  real(rk), allocatable :: wd_matrix(:,:,:), diffwd_matrix(:,:)
  complex(rk) :: one_imag, res, val(-Jmax:Jmax,0:Jmax,npoints)
  character(cl) :: sj, sk, sm, fname
  logical :: iftest

  !write(out, '(a)') 'symtop_2d_grid_theta_chi: compute symmetric-top functions on a 2D grid of theta and chi Euler angles'
  
  one_imag = cmplx(0.0_rk,1.0_rk)
    
  !write(out, '(a,1x,i4,1x,i8)') 'symtop_2d_grid_theta_chi: Jmax, npoints =', Jmax, npoints
    
  ! initialize some data required for computing Wigner D-matrix
    
#if defined(_WIGD_FOURIER_)
  
  !write(out, '(a)') 'symtop_2d_grid_theta_chi: use dffs_m module for Wigner D-matrix'
  call dffs_read_coef(Jmax*2)
  
#elif defined(_WIGD_FOURIER_BIGJ_)
  
  !write(out, '(a)') 'symtop_2d_grid_theta_chi: use wigner_d module for Wigner D-matrix'

  ! allocate matrices for computing Wigner small-d matrices using module wigner_dmat2/wigner_d.f90

  allocate(wd_matrix(npoints,2*Jmax+1,2*Jmax+1), diffwd_matrix(2*Jmax+1,2*Jmax+1), stat=info)
  if (info/=0) then
    write(out, '(/a/a,10(1x,i6))') &
      'symtop_2d_grid_theta_chi error: failed to allocate wd_matrix(npoints,2*Jmax+1,2*Jmax+1), &
      diffwd_matrix(2*Jmax+1,2*Jmax+1)', 'npoints, Jmax =', npoints, Jmax
    stop
  endif
  
#else

  !write(out, '(a)') 'symtop_2d_grid_theta_chi: use slow djmk_small routine to compute Wigner D-matrix'

#endif

  ! start computing values of symmetric-top functions on grid for all values of J=0..Jmax

  val = 0
  
  do j=0, Jmax
    
    !write(out, '(a,1x,i4)') 'symtop_2d_grid_theta_chi: J =', j
    
    fac = sqrt(real(2*j+1,rk)/(8.0_rk*pi**2))
  
#if defined(_WIGD_FOURIER_BIGJ_)
    do ipoint=1, npoints
      call Wigner_dmatrix(real(j,rk), grid(1,ipoint), wd_matrix(ipoint,1:2*j+1,1:2*j+1), &
                          diffwd_matrix(1:2*j+1,1:2*j+1))
    enddo
#endif

    m = 0
    do k=-j, j
      do ipoint=1, npoints
#if defined(_WIGD_FOURIER_)
        res = dffs( j*2, m*2, k*2, grid(1,ipoint) ) * exp(one_imag * k * grid(2,ipoint))
#elif defined(_WIGD_FOURIER_BIGJ_)
        res = wd_matrix(ipoint,j+m+1,j+k+1) * exp(one_imag * k * grid(2,ipoint))
#else
        res = djmk_small(real(j,rk), real(m,rk), real(k,rk), grid(1,ipoint)) &
            * exp(one_imag * k * grid(2,ipoint))
#endif
        val(k,j,ipoint) = res * fac
      enddo ! ipoint
    enddo ! k
  
  enddo ! j
  
#if defined(_WIGD_FOURIER_BIGJ_)
  deallocate(wd_matrix, diffwd_matrix)
#endif

  val_r = real(val, kind=rk)
  val_i = aimag(val)

  !write(out, '(a)') 'symtop_2d_grid_theta_chi: done'

  ! for testing, print values of primitive functions into ASCII files
  
  iftest = .false.
  if (iftest) then
    do j=0, Jmax
      write(sj,*) j
      do k=-j, j
        write(sk,*) k
        fname = 'symtop_func_j'//trim(adjustl(sj))//'_k'//trim(adjustl(sk))
        open(iounit,form='formatted',position='rewind',action='write',file=fname)
        do ipoint=1, npoints
          write(iounit,'(3(1x,es16.8),3x,2(1x,es16.8))') grid(1:2,ipoint), val(k,j,ipoint)
        enddo
        close(iounit)
      enddo
    enddo
  endif

end subroutine symtop_2d_grid_theta_chi
