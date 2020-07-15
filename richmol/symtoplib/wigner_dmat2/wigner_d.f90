module wigner_d

use accuracy

contains

!************************************************************************************************************************************************************ 
!  FUNCTIONS: Wigner's d-matrix
!  Methods: (a) Complex Fourier-series expansion of the d-matrix; 
!           (b) the Fourier coefficients are obtained by numerical diagonalizing the angular-momentum operator Jy, using the ZHBEV subroutine of LAPACK.
!
!  update:  (y/m/d) 2015/09/30 by G.R.J
!*************************************************************************************************************************************************************
!    
!   [1] The defination of the term X_{m}=<j,m-1|J_{-}|j,m>, where J_{-} is the ladder operator, 
!       obeying J_{-}|j,m>=X_{m} |j,m-1>, and {|j,m>} eigenvectors of the angular-momentum operator Jz.  
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++	
    function X(j,n)
	real(rk) X,n,j
	  X=dsqrt((j+n)*(j-n+1.d0))
	end function X

!****************************************************************************************************
!   [2]  The Hermitian matrix of J_{y}
!******************************************************************************************************
	subroutine coeffi(A,Ndim)
	implicit real(rk) (a-h,o-z)
 	integer(ik) Ndim,m
    real(rk) jmq
	COMPLEX(rk) A(Ndim,Ndim), im
! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    im=(0.d0,1.d0)
    jmq=(dble(Ndim)-1.d0)/2.d0
  	do i=1,Ndim
		do j=1,Ndim
            A(i,j)=0.d0

           if(i.eq.1) then
	        A(i,i+1)=X(jmq,dble(i)-jmq)*im/(2.d0)
            endif
            if(i.eq.Ndim) then
	        A(i,i-1)=-X(jmq,-dble(i)+jmq+2)*im/(2.d0)
            endif
             
 	       if((i.gt.1).and.(i.lt.Ndim))then
    	        A(i,i+1)=X(jmq,dble(i)-jmq)*im/(2.d0)
	            A(i,i-1)=-X(jmq,-dble(i)+jmq+2)*im/(2.d0)
	       endif
   	    enddo
    enddo
	return
    end subroutine coeffi

!**************************************************************************************************************************
!   [3] Calculation of the eigenvalues and the right eigenvectors for the Hermitian matrix J_{y}, 
!       using ZHBEV subroutine of LAPACK.            
!****************************************************************************************************************************
    subroutine eigen(A,Eigenvalue,Eigenvec,Ndim)
    character *1 JOBZ,UPLO
    integer(ik) Ndim,KD,LDAB,LDZ,INFO
    double precision Eigenvalue(Ndim)
    double precision,allocatable:: RWORK(:)
    double complex,allocatable:: WORK(:),AB(:,:)
    double complex Eigenvec(Ndim,Ndim),A(Ndim,Ndim)
! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   JOBZ='V'
   UPLO='U'
   KD=1
   LDAB=KD+1
   LDZ=max(1,Ndim)
   INFO=0
   allocate(AB(LDAB,Ndim),WORK(Ndim),RWORK(max(1,3*Ndim-2)))

   call coeffi(A,Ndim)
    	do i=1,Ndim
		  do j=i,Ndim
		   if(max(1,j-kd).le.i) then
            AB(KD+1+i-j,j)=A(i,j)
		   end if
          enddo
        enddo

   call ZHBEV (JOBZ, UPLO, Ndim, KD, AB, LDAB, Eigenvalue, Eigenvec, LDZ, WORK, RWORK,INFO)

   deallocate(RWORK,WORK,AB)

   end subroutine eigen


!**************************************************************************************************************************
!   [4] The subroutine to calculate the Wigner-d matrix and its first-order derivative with various mv and nv for a given {j,theta}
!****************************************************************************************************************************
	subroutine Wigner_dmatrix(jmq,theta,wd_matrix,diffwd_matrix)
	integer(ik) Ndim,ixx,iyy,mu
    real(rk) jmq, mvar, nvar,theta, wd, wdderivative, wd_matrix(int(2*jmq+1),int(2*jmq+1)), diffwd_matrix(int(2*jmq+1),int(2*jmq+1))
	double complex  inum
	double precision,allocatable :: Evalue(:),Eigenvalue(:)
    double complex,allocatable :: coeffimatrix(:,:), Eigenvector(:,:)
! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++	
	inum=(0.d0,1.d0)
    Ndim=int(2.d0*jmq+1.d0)

    allocate(Evalue(Ndim),coeffimatrix(Ndim,Ndim),Eigenvector(Ndim,Ndim),Eigenvalue(Ndim))

    call coeffi(coeffimatrix,Ndim)	 
    call eigen(coeffimatrix,Eigenvalue,Eigenvector,Ndim)

     do ixx=1,Ndim
	   do iyy=1,Ndim
         
	      wd=0.d0
		  wdderivative=0.d0
		 
 	      do mu=1,Ndim
		     if(dble(int(jmq)).eq.jmq) then
                Evalue(mu)=dble(floor(Real(Eigenvalue(mu))+0.5d0))
			 else
                Evalue(mu)=dble(floor(Real(Eigenvalue(mu)))+0.5d0)
			 endif
                wd=wd+exp(-inum*Evalue(mu)*theta)*Eigenvector(ixx,mu)*dconjg(Eigenvector(iyy,mu))
                wdderivative=wdderivative+(-inum*Evalue(mu))*exp(-inum*Evalue(mu)*theta)*Eigenvector(ixx,mu)*dconjg(Eigenvector(iyy,mu))
		  enddo    
		  
		  wd_matrix(ixx,iyy)=wd
		  diffwd_matrix(ixx,iyy)=wdderivative
		end do
	 end do

    deallocate(Evalue, coeffimatrix, Eigenvector,Eigenvalue)

    end subroutine Wigner_dmatrix


end module wigner_d

