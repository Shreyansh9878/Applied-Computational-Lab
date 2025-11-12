!----------------------------------------------!
!---Santosh Mogurampelly; santosh@iitj.ac.in---!
!----------------------------------------------!
program vacfprogram
implicit none

  integer:: i,j,k,dummy
  real:: ke,ken
  integer,parameter:: nt=10000,nh=5,nmon=100,taumax=2000
  real(kind=8),dimension(:,:):: vx(nmon,nt*nh),vy(nmon,nt*nh),vz(nmon,nt*nh)
  real(kind=8),dimension(:,:):: vacfmon(nmon,taumax),normmon(nmon,taumax)
  real(kind=8),dimension(:):: vacf(taumax),norm(taumax)

    open(12,file='fort.9',action='read')
    open(13,file='vacf_md.dat',action='write')
!read and store velocities in v
    do i=1,nt*nh
     do j=1,nmon
     read(12,'(f22.11,f22.11,f22.11)')vx(j,i),vy(j,i),vz(j,i) 
     if((i.eq.nt*nh).and.(j.eq.nmon))write(*,'(i7,f22.11,f22.11,f22.11)')dummy,vx(j,i),vy(j,i),vz(j,i)
     enddo
    enddo
  
!initialize vacf and norm
    do i=1,taumax
     vacf(i)=0.0d0
     norm(i)=0.0d0
     do j=1,nmon
      vacfmon(j,i)=0.0d0
      normmon(j,i)=0.0d0
     enddo
    enddo
!    ke=0.0d0
!    ken=0.0d0
    do i=1,nt*nh
     do j=i,nt*nh
      do k=1,nmon
       if((j-i+1).le.taumax) then
        vacfmon(k,j-i+1)=vacfmon(k,j-i+1)+vx(k,i)*vx(k,j)+vy(k,i)*vy(k,j)+vz(k,i)*vz(k,j)
!        ke=ke+vx(k,j)*vx(k,j)+vy(k,j)*vy(k,j)+vz(k,j)*vz(k,j)
!        ken=ken+1.0d0
        normmon(k,j-i+1)=normmon(k,j-i+1)+1.0d0
       endif
      enddo
     enddo
    enddo
!normalize vacfmon
    do i=1,nmon
     do j=1,taumax
      if(normmon(i,j).gt.0.0d0)vacfmon(i,j)=vacfmon(i,j)/normmon(i,j)
     enddo
    enddo

    do i=1,taumax
     do j=1,nmon
      vacf(i)=vacf(i)+vacfmon(j,i)
      norm(i)=norm(i)+1.0d0
     enddo
    enddo

    do i=1,taumax
     if(norm(i).gt.0.0d0)vacf(i)=vacf(i)/norm(i)
     write(13,'(3f22.11)')i*0.001,vacf(i),vacfmon(4,i)
    enddo
!    write(*,*)"KE=",0.50d0*5*ke/(3.0d0*ken)
    close(12)
    close(13)

end program vacfprogram
