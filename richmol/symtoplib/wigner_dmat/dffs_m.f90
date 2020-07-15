!========================================================================================
module dffs_m
!
! (y/m/d)2013/11/5,8,18,20,12/26,2014/10/31,11/1 by N.Tajima
! dffs_m.f90 : d-function computed by means of a Fourier series
!
! contents
! 1: module dffs_m
! 1-1: function icoef: is used only inside the module
! 1-2: function dffs: returns d-funtion value computed with a Fourier series expression
! 1-3: subroutine dffs_read_coef: reads the coefficients of the Fourier series from a
!          text data file "rmfsc.dat" (rotaion matrix Fourier series coefficients)
! 1-4: subroutine dffs_read_coef_binary : reads the coefficients from a binary file
!          "rmfsc_binary.dat", which can be created by first calling dffs_read_coef and
!          second callilng dffs_write_coef_binary
! 1-5: subroutine dffs_write_coef_binary : writes the coefficients to a binary data file
!          "rmfsc_binary.dat", which is much faster to read than the text data file.
!
! An important explanation of the variables in the program:
!   Let "j_tam" mean the true angular momentum quantum number, = 0, 1/2, 1, 3/2, 2,...,
!   where "true" means not being an integer index "j" used in this program to label the
!   quantum number: j = 2*j_tam = 0,1,2,3,....
!   Likewise, let m_tam and k_tam mean the z-component of j_tam,
!     m_tam, k_tam = -j_tam, -j_tam+1, ... , j_tam-1, j_tam
!   Variables m and k used in this program are integer indices to label m_tam and k_tam:
!     m = 2*m_tam = -j, -j+2, -j+4, ... ,j-2, j
!     k = 2*k_tam = -j, -j+2, -j+4, ... ,j-2, j
!   A varible febo (a flag to distinguish between Fermionic and Bosonic) is defined as
!       febo=mod(j,2)=0 for Bosonic (integer j_tam)
!       febo=mod(j,2)=1 for Ferminonic (half integer j_tam)
!   A variable evod (a flag to distinguish whether d-function is an even or odd function
!   of theta, i.e., it is expressed by whether cos or sin function) is defined as
!       evod=1 when m-k is even i.e., when mod(m-k,4)==0
!       evod=2 when m-k is odd  i.e., when mod(m-k,4)/=0
!   A half integer means an integer plus a half. One might call it "half odd integer".
!   {Integer} and {half integer} are exclusive sets of numbers.
!
  use accuracy, only: stdin=>inp, stdout=>out, wp=>rk, cl
!  use basic_set_up_m
  implicit none
  private
  public dffs, dffs_read_coef, dffs_read_coef_binary, dffs_write_coef_binary
  real(wp),allocatable,save::coef(:) ! coefficients of Fourier series of rotation matrix
  real(wp),parameter::bignum=1.0e+30_wp, biggnum=2.0e+30_wp
  integer,save::jmax=-1 ! =2*(maximum j_tam to be used in this calculation)
  integer,save::coef_index_shift ! used to calculate index of coef for febo=1.
!               Coefficients for febo=0(1) are stored in the first (second) half of coef.
contains

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function icoef(j,m,k)
  implicit none
  integer::icoef
  integer,intent(in)::j,m,k
  if(mod(j,2)==0) then
    icoef=(j*(j+2)/8)**2+(m**2+2*(m+k))*(j+2)/8
  else
    icoef=(((j-1)*(j+1)*(j+3)/48)*((3*j-1)/2))/2 &
&        +(m-1)*(m+1)*(j+1)/8 + (m+k)*(j+1)/4 + coef_index_shift
  end if
end function icoef

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function dffs(j,m,k,theta) ! d-function computed with the Fourier series expression
  implicit none
  real(wp)::dffs ! the value of the d-function^{j_tam}_{m_tam,k_tam}(theta)
  integer,intent(in)::j ! 2*j_tam, must be in {0, 1, ..., jmax}
  integer,intent(in)::m ! 2*m_tam, must be in {-j, -j+2, ... , j-2, j}
  integer,intent(in)::k ! 2*k_tam, must be in {-j, -j+2, ... , j-2, j}
  real(wp),intent(in)::theta ! angle in units of radian
  integer,parameter::nvecmax=201  ! Increase if jmax>400 so as not to stop this program.
  integer::i ! index =1, 2, ... ,imax to specify the elements of coef
  integer::i0 ! (starting value of i for coef used in this call) minus one
  integer::mc ! m may be transformed to -m, k, or -k if necessary
  integer::kc ! k may be transformed to -k, m, or -m if necessary
  integer::evod ! =1:mod(m-k,4)==0(even func), =2:otherwise(odd func)
  integer::febo ! =0:mod(j,2)==0(Bosonic),     =1:mod(j,2)==1(Ferminoic)
  integer::phase ! +/-1 for the sign change due to the transformation (m,k) to (mc,kc)
  integer::nvec ! =(j+2)/2, the number of the coefficients of the Fourier series
  real(wp),save::csnt(nvecmax,2,0:1) ! csnt(n,evod,febo)=f((n-1+febo/2)*theta),
!                                      f=cos(sin) for evod=1(2), n=1..nvec=(j+2)/2
  real(wp),save::theta_csnt(0:1)=[0.0_wp,0.0_wp] ! theta for stored csnt for febo={0,1}
  integer,save::nvec_csnt(0:1)=[-1,-1] ! nvec for stored csnt for febo={0,1}
  real(wp)::ct, st, halftheta ! =cos(theta),sin(theta),theta/2
  real(wp)::sum ! temporary variable to calculate a summation

  if(j<0 .or. abs(m)>j .or. abs(k)>j .or. mod(j-m,2)/=0 .or. mod(j-k,2)/=0) then
     write(stdout,'(A,3I12,A)') 'dffs:error: j,m,k=',j,m,k,' returned value=0'
     dffs=0.0_wp ! not allowed combination of (j,m,k) -> d-function=0
     return
!   stop
  end if

  if(j>jmax) then
     write(stdout,'(A,I4,A,I12,A,2I12,A)') 'dffs:error:jmax=',jmax,'>j=',j,' m,k=',m,k&
&    ,' returned value=2'
     dffs=2.0_wp ! As d-function is in [-1..1], 2 means a failure in calculation.
     return
!   write(stdout,'(A,3I12,A,I12)') 'dffs:error in arguments: j,m,k=',j,m,k,' jmax=',jmax
!   stop
  end if

  febo=mod(j,2)
  if(mod(m-k,4) == 0) then
    evod=1
  else
    evod=2
  end if

  if(m>=k) then
    if(m>=-k) then ! region A (called as such a name in my notebook)
      mc=m
      kc=k
      phase=1
    else ! if(m<-k) then ! region D
      mc=-k
      kc=-m
      phase=1
    end if
  else ! if(m<k) then
    if(m>=-k) then  ! region B
      mc=k
      kc=m
      phase=3-2*evod
    else ! if(m<-k) then ! region C
      mc=-m
      kc=-k
      phase=3-2*evod
    end if
  end if
! write(stdout,'(A,7I4)') 'j,m,k,evod,mc,kc,phase',j,m,k,evod,mc,kc,phase
  if(-mc > kc .or. kc > mc) then ! never happens if this program is correcrt.
    write(stdout,'()') 'dffs: error in transforming (m,k) s.t. |k|<=m: needs debug !?'
    stop
  end if

  nvec=(j+2)/2
  if(nvec > nvecmax) then
    write(stdout,'(A,2I12)') 'dffs: error: ',nvecmax,nvec
    write(stdout,'(A)') 'Increase nvecmax by editing a line in the program containing'
    write(stdout,'(A)') 'integer,parameter::nvecmax='
    stop
  end if

  if(abs(theta-theta_csnt(febo)) > 1.0e-15_wp .or. nvec > nvec_csnt(febo)) then
    if(febo==0) then
      csnt(1,1,febo)=1 ! cos(0)
      csnt(1,2,febo)=0 ! sin(0)
      ct=cos(theta)
      st=sin(theta)
    else
      halftheta=theta*0.5_wp
      csnt(1,1,febo)=cos(halftheta)
      csnt(1,2,febo)=sin(halftheta)
      ct=csnt(1,1,febo)**2-csnt(1,2,febo)**2
      st=2*csnt(1,1,febo)*csnt(1,2,febo)
    end if
    do i=1,nvec-1
      csnt(i+1,1,febo)=ct*csnt(i,1,febo)-st*csnt(i,2,febo)
      csnt(i+1,2,febo)=st*csnt(i,1,febo)+ct*csnt(i,2,febo)
    end do
    nvec_csnt(febo)=nvec
    theta_csnt(febo)=theta
!   write(stdout,'(A,I12,F16.8)') 'dffs:calculate csnt ',j,theta ! debug info
! else
!   write(stdout,'(A,I12,F16.8)') 'dffs:reuse csnt ',j,theta ! debug info
  end if

  i0=icoef(j,mc,kc)
  if(coef(i0+1) > bignum) then
     write(stdout,'(A,3I12,A)') 'dffs: error: missing coef for',j,m,k,' returned value=2'
     dffs=2.0_wp ! As d-function is in [-1..1], 2 means a failure in calculation.
     return
!    write(stdout,'(A,3I12)') 'dffs: error: missing coef for',j,m,k
!    stop ! An alternative is stopping the calculation.
  end if
  sum=0
  do i=1,nvec
    sum=sum+coef(i+i0)*csnt(i,evod,febo)
!   write(stdout,'(A,I5,A,ES16.7)') 'i=',i,' c=',coef(i+i0) ! debug info
  end do
  dffs=sum*phase
end function dffs

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
subroutine dffs_read_coef(jmax_to_use)
! reading Fourier series coefficients from a text data file 'rmfsc.dat'
  implicit none
  integer,parameter::printlevel=7 ! =0 for silence, =9 for all debug info to be shown
  integer,intent(in)::jmax_to_use ! 2*(max j_tam to be used in this calculation)
  integer::i, i0, imax, iadd, is ! i is the index of coef
  integer::imax1 ! max i for integer j
  integer::imax2 ! max i for half integer j
  integer::jmax1 ! max j for integer j
  integer::jmax2 ! max j for half integer j
  integer::status ! IO error code
  integer::j, m, k ! indices for j_tam, m_tam, k_tam
  integer::jmin ! minimum value of j = 0(1) for mod(j,2)=0(1)
  integer::n ! index for nu, where nu is for Fourier series terms cos or sin (nu*theta)
  integer::fread=20 ! IOe unit number to read rmfsc.dat (d-functin coef text data file)
  integer::nvec ! the number of elements of the coefficient vector for each (j,m,k)
  real(wp)::vdummy ! dummy variable to read unnecessary data
  real(wp)::tmp
  integer::jlast ! to detect the change of the value of j read from the data file
!                  to print it on screen as progress notification of data reading
  integer::nerr, n_zero, n_message, n_total, n_missing, n_large

  jmax=jmax_to_use
  if(printlevel>=1) write(stdout,'(2A,I12,A)') 'dffs_read_coef: reading text data file '&
& ,'rmfsc.dat up to j<=',jmax,'/2'
  if(jmax < 0) then
    write(stdout,'(A,I12)') 'dffs_read_coef: error : jmax=',jmax
    stop
  end if

  if(mod(jmax,2)==0) then
    jmax1=jmax
    jmax2=jmax-1
  else
    jmax1=jmax-1
    jmax2=jmax
  end if

  coef_index_shift=0
  imax1=icoef(jmax1+2,0,0)
! imax1=((jmax1+2)*(jmax1+4))**2/64 ! explicit expression
  imax2=icoef(jmax2+2,1,-1)
! imax2=(jmax2+1)*(3*jmax2+5)*(jmax2+3)*(jmax2+5)/192 ! explicit expression
  if(printlevel>=8) write(stdout,'(A,2I12)') &
& 'dffs_read_coef: no. of coef for integer j and half integer j=',imax1,imax2
  coef_index_shift=imax1
  imax=imax1+imax2

  if(imax<=0 .or. imax1<0 .or. imax2<0) then ! This is not a perfect check.
    write(stdout,'(2A,3I12)') 'dffs_read_coef: error: too large size of coef for'&
&   ,' default-type integer: imax,imax1,imax2',imax,imax1,imax2
    stop
  end if

  if(printlevel>=9) then
    write(stdout,'(A,3I4,2I12)') 'dffs_read_coef: jmax,jmax1,jmax2,imax1,imax2'&
&   ,jmax,jmax1,jmax2,imax1,imax2
  end if

  if(allocated(coef)) then
    deallocate(coef,stat=status)
    if(status /= 0) then
      write(stdout,'(A)') 'dffs_read_coef: error in deallocating coef'
      stop
    end if
  end if
  allocate(coef(imax),stat=status)
  if(status /= 0) then
      write(stdout,'(A,I12)') 'dffs_read_coef: error in allocating coef, imax=',imax
    stop
  end if

  if(printlevel>=5) write(stdout,'(A,I4,A)') 'dffs_read_coef: successfully allocated '&
& ,imax*8/2**20,' MiB for coef'

  test_of_icoef:&
& if(.false.) then ! begin : test of the index calculator function icoef (only for debug)
    write(stdout,'(A)') 'dffs_read_coef: test of the index calculator function icoef'
    coef(1:imax)=0
    n=0
    nerr=0 ! the number of errors
    do jmin=0,1
!      write(stdout,'(A,I4)') 'jmin=',jmin
      do j=jmin,jmax,2
!        write(stdout,'(A,I4)') 'j=',j
        do m=jmin,j,2
!          write(stdout,'(A,I4)') 'm=',m
          do k=-m,m,2
            i0=icoef(j,m,k)
!            write(stdout,'(A,5I4,I12)') 'jmin,j,m,k,(j+2)/2,i0=',jmin,j,m,k,(j+2)/2,i0
            do iadd=1,(j+2)/2
              n=n+1
              i=i0+iadd
              if(i>imax) then
                write(stdout,'(A,4I4,3I12)') 'i > imax' &
&                ,j,m,k,iadd,i0,i,n
              else if(coef(i)==0) then
                coef(i)=1
              else
                write(stdout,'(A,4I4,3I12,ES12.3E3)') 'overwrite coef' &
&                ,j,m,k,iadd,i0,i,n,coef(i)
                coef(i)=coef(i)+1
                nerr=nerr+1
              end if
            end do
          end do
        end do
      end do
    end do
    write(stdout,'(A,2I12,A,I12)') 'n=',n,imax1+imax2,' overlap errors #=',nerr
    nerr=0
    do i=1,n
      if(coef(i) /= 1) then
        write(stdout,'(A,I12,ES16.7E3)') 'i,coef=',i,coef(i)
        nerr=nerr+1
      end if
    end do
    write(stdout,'(A,I12)') 'gap or overlap errors #=',nerr
  end if test_of_icoef ! end : test of the index calculator function icoef

  open(unit=fread,file='rmfsc.dat',status='OLD',action='READ',iostat=status)
  if(status /= 0) then
    write(stdout,'(A,2I12)')  'dffs_read_coef: error in opening rmfsc.dat. unit,status='&
&   ,fread,status
    stop
  end if

  coef(1:imax)=biggnum ! means that data are missing
  jlast=-1
  read_loop: do
    read(fread,*,iostat=status) j,m,k
    if(status /= 0) then
      if(printlevel>=1) write(stdout,'(A)') 'dffs_read_coef: end of rmfsc.dat reached'
      exit read_loop
    end if
    if(j<0 .or. m<0 .or. m>j .or. abs(k)>m .or. mod(j-m,2)/=0 .or. mod(j-k,2)/=0) then
      write(stdout,'(A,3I12)') 'dffs_read_coef:error in reading data: j,m,k=',j,m,k
      stop
    end if
    nvec=(j+2)/2
    if(jlast /= j) then
      jlast=j
      if(printlevel>=9 .or. printlevel>=5 .and. mod(j,20)==0) write(stdout,'(A,I3,A)')&
&     'dffs_read_coef: reading coef data for j=',j,'/2'
    end if
    ifblock_j:if(j<=jmax) then ! read data which has places in allocated array
      i0=icoef(j,m,k)
      if(coef(i0+1) < bignum) then
        write(stdout,'(A,3I12)') 'dffs_read_coef:warning:duplicated data, j,m,k=',j,m,k
!       stop
      end if
      if(mod(j,2)==0 .and. mod(m-k,4)/=0) then
        coef(i0+1)=0 ! coef of sin(0*theta) is omitted in my text data file, while it's
!                      value (defined as zero) is used in func. dffs.
        is=2 ! starting index to read coef values is i0+is
      else
        is=1
      end if
      do i=is,nvec
        read(fread,*,iostat=status) coef(i0+i)
        if(status /= 0) then
          write(stdout,'(A)') 'dffs_read_coef:error in text data rmfsc.dat'
!         exit read_loop
          stop
        end if
      end do
    else ifblock_j ! skip data which are not covered by allocated array
      if(mod(j,2)==0 .and. mod(m-k,4)/=0) then
        is=2
      else
        is=1
      end if
      do i=is,nvec
        read(fread,*,iostat=status) vdummy
        if(status /= 0) then
          write(stdout,'(A,3I12,A)') 'dffs_read_coef:error in reading data: j,m,k='&
&         ,j,m,k,' not to be used'
          exit read_loop
!         stop
        end if
      end do
    end if ifblock_j
  end do read_loop

  close(unit=fread,iostat=status)
  if(status /= 0) then
    write(stdout,'(A,2I12)')  'dffs_read_coef:error in closing rmfsc.dat. unit,status='&
&   ,fread,status
    stop
  end if

  find_missing_coef:if(printlevel>=7) then ! begin : find missing coef
!   write(stdout,'(A)') 'dffs_read_coef: search for missing coefficients'
    n_message=0 ! the number of error messages printed so far
    n_total=0
    n_missing=0
    n_large=0
    do jmin=0,1
      do j=jmin,jmax,2
        do m=jmin,j,2
          do k=-m,m,2
            i0=icoef(j,m,k)
            do n=1,(j+2)/2
              n_total=n_total+1
              tmp=coef(i0+n)
              if(tmp>bignum) then
                n_missing=n_missing+1
                n_message=n_message+1
                if(n_message<=1000) write(stdout,'(4I12,A)') j,m,k,n&
&               ,' =j,m,k,n : missing coef data'
              else if(abs(tmp)>1.0_wp) then
                n_large=n_large+1
                n_message=n_message+1
                if(n_message<=1000) write(stdout,'(4I12,ES16.7,A)') j,m,k,n,tmp&
&               ,' =j,m,k,n,coef : too large !?'
              end if
            end do
          end do
        end do
      end do
    end do
    write(stdout,'(A,I10,A,I10,A,I10,A)') 'dffs_read_coef:among ',n_total,' coef,'&
&   ,n_missing,' are missing, ',n_large,' are not in [-1,1]'
  end if find_missing_coef ! end : find missing coef

  if(printlevel>=9) then ! begin : count the number of non-zero coefficients
    n_zero=0
    do i=1,imax
      if(coef(i) == 0.0_wp) then
        n_zero=n_zero+1
      end if
    end do
    write(stdout,'(A,I12,A,I12)') 'The number of zero-value/total coefficients ='&
&   ,n_zero,' / ',imax
  end if ! end : count the number of non-zero coefficients

end subroutine dffs_read_coef

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
subroutine dffs_read_coef_binary(jmax_to_use)
! writes Fourier series coefficients to a binary file 'rmfsc_binary.dat'
  implicit none
  integer,intent(in)::jmax_to_use
  integer::jmax_dat,imax1_dat,imax2_dat
  integer::imax1 ! max i for integer j
  integer::imax2 ! max i for half odd integer j
  integer::jmax1 ! max j for integer j
  integer::jmax2 ! max j for half odd integer j
  integer::imax  ! = imax1+imax2
  integer::status ! IO error code
  integer::fread=21 ! IO unit number to write rmfsc_binary.dat (d-functin coef data)
! module variables : jmax,coef,coef_index_shift

  if(jmax_to_use < 0) then
    write(stdout,'(A,I12)')  'dffs_read_coef_binary:error:jmax=',jmax_to_use
    stop
  end if

  open(unit=fread,file='rmfsc_binary.dat',status='OLD',action='READ' &
& ,form='UNFORMATTED',iostat=status)
  if(status /= 0) then
    write(stdout,'(A)')  'dffs_read_coef_binary:error in opening rmfsc_binary.dat to read'
    write(stdout,'(A,2I12)')  'unit,status=',fread,status
    stop
  end if

  read(fread,iostat=status) jmax_dat,imax1_dat,imax2_dat
  if(status /= 0) then
    write(stdout,'(A,3I12)') 'dffs_read_coef_binary:error in reading jmax,imax1,imax2'&
&   ,jmax_dat,imax1_dat,imax2_dat
    stop
  end if

  if(jmax_dat < jmax_to_use) then
    write(stdout,'(2A,2I12)') 'dffs_read_coef_binary:error: jmax of binary data file < '&
&   ,'jmax to be used ',jmax_dat,jmax_to_use
    stop
  end if

  jmax=jmax_to_use
  if(mod(jmax,2)==0) then
    jmax1=jmax
    jmax2=jmax-1
  else
    jmax1=jmax-1
    jmax2=jmax
  end if

  coef_index_shift=0
  imax1=icoef(jmax1+2,0,0)
  if(imax1 > imax1_dat) then
    write(stdout,'(A,2I12)') 'dffs_read_coef_binary:error:imax1>imax1_dat',imax1,imax1_dat
    stop
  end if
  imax2=icoef(jmax2+2,1,-1)
  if(imax2 > imax2_dat) then
    write(stdout,'(A,2I12)') 'dffs_read_coef_binary:error:imax2>imax2_dat',imax2,imax2_dat
    stop
  end if
  coef_index_shift=imax1
  imax=imax1+imax2

  if(allocated(coef)) then
    deallocate(coef,stat=status)
    if(status /= 0) then
      write(stdout,'(A)') 'dffs_read_coef_binary:error in deallocating coef'
      stop
    end if
  end if
  allocate(coef(imax),stat=status)
  if(status /= 0) then
    write(stdout,'(A,I12)') 'dffs_read_coef_binary:error in allocating coef, imax=',imax
    stop
  end if

  read(fread,iostat=status) coef(1:imax1)
  if(status /= 0) then
    write(stdout,'(A,I12)') 'dffs_read_coef_binary:error in reading coef(1:imax1)',status
    stop
  end if
  read(fread,iostat=status) coef(imax1+1:imax)
  if(status /= 0) then
    write(stdout,'(A,I12)') 'dffs_read_coef_binary:error in reading coef(imax1+1:imax)'&
&   ,status
    stop
  end if
  close(fread,iostat=status)
  if(status /= 0) then
    write(stdout,'(2A,2I12)') 'dffs_read_coef_binary:error in closing rmfsc_binary.dat '&
&   ,'after having read it. unit,status=',fread,status
    stop
  end if

end subroutine dffs_read_coef_binary

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
subroutine dffs_write_coef_binary()
! writes Fourier series coefficients to a binary file 'rmfsc_binary.dat'
  implicit none
  integer::imax1 ! max i for integer j
  integer::imax2 ! max i for half odd integer j (if it starts from 1)
  integer::jmax1 ! max j for integer j
  integer::jmax2 ! max j for half odd integer j
  integer::imax  ! =imax1+imax2
  integer::status ! IO error code
  integer::fwrt=22 ! IO unit number to write rmfsc_binary.dat (d-functin coef data)
! module variables indispensably needed : jmax,coef
! module variable used for check : coef_index_shift

  if(mod(jmax,2)==0) then
    jmax1=jmax
    jmax2=jmax-1
  else
    jmax1=jmax-1
    jmax2=jmax
  end if

  imax1=icoef(jmax1+2,0,0)
  if(coef_index_shift /= imax1) then
    write(stdout,'(A,2I12)') 'error: imax1!=coef_index_shift :',imax1,coef_index_shift
    stop
  end if
  imax=icoef(jmax2+2,1,-1) ! NB. now coef_index_shift>0
  imax2=imax-imax1

  open(unit=fwrt,file='rmfsc_binary.dat',status='REPLACE',action='WRITE' &
& ,form='UNFORMATTED',iostat=status)
  if(status /= 0) then
    write(stdout,'(2A,2I12)')  'dffs_write_coef_binary:error in opening rmfsc_binary.dat'&
&   ,' to write.  unit,status=',fwrt,status
    stop
  end if
  write(fwrt,iostat=status) jmax,imax1,imax2
  if(status /= 0) stop 'dffs_write_coef_binary:error 1'
  write(fwrt,iostat=status) coef(1:imax1)
  if(status /= 0) stop 'dffs_write_coef_binary:error 2'
  write(fwrt,iostat=status) coef(imax1+1:imax)
  if(status /= 0) stop 'dffs_write_coef_binary:error 3'
  close(fwrt,iostat=status)
  if(status /= 0) then
    write(stdout,'(2A,I12)')  'dffs_write_coef_binary:error in closing rmfsc_binary.dat '&
&   ,'after having written to it. error code=',status
    stop
  end if
  write(stdout,'(A)') 'dffs_write_coef_binary:rmfsc_binary.dat has been created.'
  write(stdout,'(A,I4,A,I12,A,I12,A,F8.2,A)') ' j<=',jmax,'/2,  no. of coef=',imax1,' +'&
& ,imax2,' file size=',imax*8.0_wp/2**20,' MiB'

end subroutine dffs_write_coef_binary

end module dffs_m
