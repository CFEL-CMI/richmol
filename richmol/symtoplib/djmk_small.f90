! Computes small Wigner d-matrix d_{mk}^j

function djmk_small(j, m, k, theta)

  real(rk), intent(in) :: j, m, k, theta
  real(rk) :: djmk_small

  integer(ik) :: itmin1, itmin2, itmin, itmax1, itmax2, itmax, ij1, ij2, it, iphase, ia, ib, ic
  real(ark) :: cosb2, sinb2, sqrt_fac, sumt, denom, term

  cosb2 = cos(theta*0.5_ark)
  sinb2 = sin(theta*0.5_ark)

  itmin1 = 0
  itmin2 = nint(m-k)
  itmin = max(itmin1,itmin2)
  itmax1 = nint(j+m)
  itmax2 = nint(j-k)
  itmax = min(itmax1,itmax2)

  ij1 = nint(j-m)
  ij2 = nint(j+k)

  sqrt_fac = sqrt( fac10(itmax1) * fac10(ij1) * fac10(ij2) * fac10(itmax2) )

  sumt = 0
  do it = itmin, itmax
    iphase = (-1)**it
    ia = itmax1 - it
    ib = itmax2 - it
    ic = it + nint(k-m)
    denom = fac10(ia) * fac10(ib) * fac10(it) * fac10(ic)
    term = iphase * cosb2**(ia+ib) * sinb2**(it+ic) / denom
    sumt = sumt + term
  enddo

  djmk_small = real(sqrt_fac * sumt, rk)

end function djmk_small



function fac10(n)

  integer(ik), intent(in) :: n
  real(ark) :: fac10

  integer(ik) :: i
  real(ark) :: q

 ! integer(ik), parameter :: maxn=1000
 ! real(ark), save :: fac10_save(1:maxn) = -1.0_ark

  if (n==0) then
    fac10 = 1.0_ark
 ! elseif (n<=maxn .and. fac10_save(n)>0.0) then
 !   fac10 = fac10_save(n)
  else
    fac10 = 1.0_ark
    q = 1.0_ark
    do i = 1, n
      fac10 = fac10 * q / 10.0_ark
      q = q + 1.0_ark
    enddo
 !   fac10_save(n) = fac10
  endif

end function fac10
