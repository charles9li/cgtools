!======MODULE CODE FOR geomlib======


function ClipTrig(x)
    implicit none
    real(8) :: ClipTrig
    real(8), intent(in) :: x
    ClipTrig = min(1.D0, max(-1.D0, x))
end function

function NormRad(Rad)
    implicit none
    real(8) :: NormRad
    real(8), parameter :: pi = 3.1415926535897931D0
    real(8), parameter :: pi2 = 2.D0*pi
    real(8), intent(in) :: Rad
    NormRad = mod(Rad + pi, pi2) - pi
    if (NormRad < -pi) NormRad = NormRad + pi2
end function

function NormDeg(Deg)
    implicit none
    real(8) :: NormDeg
    real(8), intent(in) :: Deg
    NormDeg = mod(Deg + 180.D0, 360.D0) - 180.D0
    if (NormDeg < -180.D0) NormDeg = NormDeg + 360.D0
end function

subroutine Centroid(Pos, Ret, N, Dim)
    implicit none
    integer, intent(in) :: N, Dim
    real(8), dimension(N,Dim), intent(in) :: Pos
    real(8), dimension(Dim), intent(out) :: Ret
    Ret = sum(Pos, 1) / real(N)
end subroutine

subroutine CentroidInd(Pos, AtomInd, Ret, N, Dim, NAtomInd)
    implicit none
    integer, intent(in) :: N, Dim, NAtomInd
    real(8), dimension(N,Dim), intent(in) :: Pos
    integer, dimension(NAtomInd), intent(in) :: AtomInd
    real(8), dimension(Dim), intent(out) :: Ret
    integer :: i
    Ret = 0.
    do i = 1, NAtomInd
        Ret = Ret + Pos(AtomInd(i) + 1, :)
    enddo
    Ret = Ret / dble(NAtomInd)
end subroutine

real(8) function Length(Vec, Dim)
    implicit none
    integer, intent(in) :: Dim
    real(8), dimension(Dim), intent(in) :: Vec
    Length = sqrt(sum(Vec*Vec))
end function

subroutine UnitVec(Vec, Ret, Dim)
    implicit none
    integer, intent(in) :: Dim
    real(8), dimension(Dim), intent(in) :: Vec
    real(8), dimension(Dim), intent(out) :: Ret
    Ret = Vec / sqrt(sum(Vec*Vec))
end subroutine

subroutine CrossProd3(r1, r2, Ret, Dim)
    integer, intent(in) :: Dim
    real(8), dimension(Dim), intent(in) :: r1, r2
    real(8), dimension(Dim), intent(out) :: Ret
    if (Dim /= 3) then
        print *, 'Expecting three dimensions and found', Dim
        stop
    endif
    Ret(1) = r1(2)*r2(3)-r1(3)*r2(2)
    Ret(2) = r1(3)*r2(1)-r1(1)*r2(3)
    Ret(3) = r1(1)*r2(2)-r1(2)*r2(1)
end subroutine

real(8) function RadFromTrig(SinVal, CosVal)
    implicit none
    real(8), parameter :: pi = 3.1415926535897931D0
    real(8), intent(in) :: SinVal, CosVal
    real(8), external :: ClipTrig
    RadFromTrig = acos(ClipTrig(CosVal))
    if (SinVal < 0.) RadFromTrig = 2.*pi - RadFromTrig
end function

real(8) function DegFromTrig(SinVal, CosVal)
    implicit none
    real(8), parameter :: pi = 3.1415926535897931D0
    real(8), parameter :: DegPerRad = 180.D0/pi
    real(8), intent(in) :: SinVal, CosVal
    real(8), external :: ClipTrig, RadFromTrig
    DegFromTrig = DegPerRad * RadFromTrig(SinVal, CosVal)
end function

real(8) function Angle3Rad(Pos1, Pos2, Pos3, Dim)
    implicit none
    real(8), parameter :: pi = 3.1415926535897931D0
    real(8), parameter :: DegPerRad = 180.D0/pi
    integer, intent(in) :: Dim
    real(8), dimension(Dim), intent(in) :: Pos1, Pos2, Pos3
    real(8), dimension(Dim) :: Vec21, Vec23
    real(8) :: Norm, Phi
    real(8), external :: ClipTrig, NormRad
    if (Dim /= 3) then
        print *, 'Expecting three dimensions and found', Dim
        stop
    endif
    if (all(Pos1 == Pos2) .or. all(Pos2 == Pos3)) then
        Angle3Rad = 0.
        return
    endif
    Vec21 = Pos1 - Pos2
    Vec23 = Pos3 - Pos2
    Norm = sqrt(sum(Vec21*Vec21)*sum(Vec23*Vec23))
    Phi = ClipTrig(dot_product(Vec21, Vec23) / Norm)
    Angle3Rad = acos(Phi)
end function


real(8) function Dihedral3Rad(Pos1, Pos2, Pos3, Pos4, Dim)
    implicit none
    real(8), parameter :: pi = 3.1415926535897931D0
    real(8), parameter :: DegPerRad = 180.D0/pi
    integer, intent(in) :: Dim
    real(8), dimension(Dim), intent(in) :: Pos1, Pos2, Pos3, Pos4
    real(8), dimension(Dim) :: Vec12, Vec23, Vec34
    real(8), dimension(Dim) :: Norm12, Norm34
    real(8) :: Norm, Phi
    real(8), external :: ClipTrig, NormRad
    external :: CrossProd3
    if (Dim /= 3) then
        print *, 'Expecting three dimensions and found', Dim
        stop
    endif
    Vec12 = Pos2 - Pos1
    Vec23 = Pos3 - Pos2
    Vec34 = Pos4 - Pos3
    call CrossProd3(Vec12, Vec23, Norm12, Dim)
    call CrossProd3(Vec23, Vec34, Norm34, Dim)
    Norm = sqrt( sum(Norm12*Norm12) * sum(Norm34*Norm34) )
    Phi = ClipTrig(dot_product(Norm12, Norm34) / Norm)
    Phi = acos(Phi)
    if (dot_product(Vec12, Norm34) < 0.) Phi = -Phi
    Dihedral3Rad = NormRad(Phi)
end function


subroutine GetVecMappingRad(Vec1, Vec2, Vec, Ang, Dim)
    implicit none
    integer, intent(in) :: Dim
    real(8), dimension(Dim), intent(in) :: Vec1, Vec2
    real(8), dimension(Dim), intent(out) :: Vec
    real(8), intent(out) :: Ang
    real(8), dimension(Dim) :: v1, v2
    real(8) :: CosAng
    real(8), external :: ClipTrig
    external :: CrossProd3, UnitVec
    call UnitVec(Vec1, v1, Dim)
    call UnitVec(Vec2, v2, Dim)
    CosAng = ClipTrig(dot_product(v1, v2))
    Ang = acos(CosAng)
    if (CosAng == 1.) then
        Vec = v1
    elseif (CosAng == -1.) then
        Vec(2:Dim) = v1(1:Dim-1)
        Vec(1) = v1(Dim)
        Vec = Vec - dot_product(Vec, v1)
    else
        call CrossProd3(v1, v2, Vec, Dim)
    endif
end subroutine


subroutine RotateArrayAboutPoint(Pos, RotMat, Point, Ret, N, Dim)
    integer, intent(in) :: N, Dim
    real(8), dimension(N, Dim), intent(in) :: Pos
    real(8), dimension(Dim, Dim), intent(in) :: RotMat
    real(8), dimension(Dim), intent(in) :: Point
    real(8), dimension(N, Dim), intent(out) :: Ret
    integer :: i
    do i = 1, N
        Ret(i,:) = Pos(i,:) - Point
    enddo
    Ret = matmul(Ret, RotMat)
    do i = 1, N
        Ret(i,:) = Ret(i,:) + Point
    enddo
end subroutine

subroutine RotatePointAboutPoint(Pos, RotMat, Point, Ret, Dim)
    integer, intent(in) :: Dim
    real(8), dimension(Dim), intent(in) :: Pos
    real(8), dimension(Dim, Dim), intent(in) :: RotMat
    real(8), dimension(Dim), intent(in) :: Point
    real(8), dimension(Dim), intent(out) :: Ret
    Ret = matmul(Pos - Point, RotMat) + Point
end subroutine


subroutine RotMat3Rad(Vec, Ang, Ret, Dim)
    implicit none
    integer, intent(in) :: Dim
    real(8), dimension(Dim), intent(in) :: Vec
    real(8), intent(in) :: Ang
    real(8), dimension(Dim,Dim), intent(out) :: Ret
    real(8), dimension(Dim) :: v
    real(8) :: rcos, rsin
    external :: UnitVec
    if (Dim /= 3) then
        print *, 'Expecting three dimensions and found', Dim
        stop
    endif
    Ret = 0.
    rcos = cos(Ang)
    rsin = sin(Ang)
    call UnitVec(Vec, v, 3)
    Ret(1,1) =         rcos + v(1)*v(1)*(1-rcos)
    Ret(2,1) =  v(3) * rsin + v(2)*v(1)*(1-rcos)
    Ret(3,1) = -v(2) * rsin + v(3)*v(1)*(1-rcos)
    Ret(1,2) = -v(3) * rsin + v(1)*v(2)*(1-rcos)
    Ret(2,2) =         rcos + v(2)*v(2)*(1-rcos)
    Ret(3,2) =  v(1) * rsin + v(3)*v(2)*(1-rcos)
    Ret(1,3) =  v(2) * rsin + v(1)*v(3)*(1-rcos)
    Ret(2,3) = -v(1) * rsin + v(2)*v(3)*(1-rcos)
    Ret(3,3) =         rcos + v(3)*v(3)*(1-rcos)
end subroutine


subroutine RotMat3EulerRad(Phi, Theta, Psi, Ret)
    real(8), dimension(3,3), intent(out) :: Ret
    real(8), intent(in) :: Phi, Theta, Psi
    real(8) :: Sin1, Cos1, Sin2, Cos2, Sin3, Cos3
    Sin1 = sin(Phi)
    Cos1 = cos(Phi)
    Sin2 = sin(Theta)
    Cos2 = cos(Theta)
    Sin3 = sin(Psi)
    Cos3 = cos(Psi)
    Ret(1,1) = Cos1*Cos3 - Sin1*Cos2*Sin3
    Ret(1,2) = Sin1*Cos3 + Cos1*Cos2*Sin3
    Ret(1,3) = Sin2*Sin3
    Ret(2,1) = -Cos1*Sin3 - Sin1*Cos2*Cos3
    Ret(2,2) = -Sin1*Sin3 + Cos1*Cos2*Cos3
    Ret(2,3) = Sin2*Cos3
    Ret(3,1) = Sin1*Sin2
    Ret(3,2) = -Cos1*Sin2
    Ret(3,3) = Cos2
end subroutine


subroutine RotMat3Q(q0, q1, q2, q3, Ret)
    implicit none
    real(8), dimension(3,3), intent(out) :: Ret
    real(8), intent(in) :: q0, q1, q2, q3
    Ret(1,1) = q0*q0 + q1*q1 - q2*q2 - q3*q3
    Ret(1,2) = 2.*(q1*q2 + q0*q3)
    Ret(1,3) = 2.*(q1*q3 - q0*q2)
    Ret(2,1) = 2.*(q1*q2 - q0*q3)
    Ret(2,2) = q0*q0 - q1*q1 + q2*q2 - q3*q3
    Ret(2,3) = 2.*(q2*q3 + q0*q1)
    Ret(3,1) = 2.*(q1*q3 + q0*q2)
    Ret(3,2) = 2.*(q2*q3 - q0*q1)
    Ret(3,3) = q0*q0 - q1*q1 - q2*q2 + q3*q3
end subroutine


subroutine EulerFromRotMat3Rad(RotMat, Phi, Theta, Psi, Dim)
    implicit none
    integer, intent(in) :: Dim
    real(8), dimension(Dim,Dim), intent(in) :: RotMat
    real(8), intent(out) :: Phi, Theta, Psi
    real(8) :: sphi
    real(8), external :: RadFromTrig
    if (Dim /= 3) then
        print *, 'Expecting three dimensions and found', Dim
        stop
    endif
    Theta = acos(min(1., max(-1., RotMat(3,3))))
    if (RotMat(3,3)==1. .or. (RotMat(3,1)==0. .and. RotMat(3,2)==0.) &
        & .or. (RotMat(1,3)==0. .and. RotMat(2,3)==0.)) then
        Psi = 0.
        Phi = RadFromTrig(RotMat(1,2), RotMat(1,1))
    else
        sphi = sqrt(1.-RotMat(3,3)*RotMat(3,3))
        Phi = RadFromTrig(RotMat(3,1)/sphi, -RotMat(3,2)/sphi)
        Psi = RadFromTrig(RotMat(1,3)/sphi, RotMat(2,3)/sphi)
    endif
end subroutine


subroutine SpherePoints(N, Points)
    implicit none
    integer, intent(in) :: N
    real(8), dimension(N,3), intent(out) :: Points
    real(8) :: off, y, phi, r
    real(8) :: inc 
    real(8), parameter :: pi = 3.1415926535897931D0 
    integer :: k
    inc = pi * (3. - sqrt(5.))
    Points = 0.
    off = 2. / real(N)
    do k = 1, N
        y = real(k-1) * off - 1. + (off * 0.5)
        r = sqrt(max(1. - y*y, 0.))
        phi = real(k-1) * inc
        Points(k,1) = cos(phi)*r
        Points(k,2) = y
        Points(k,3) = sin(phi)*r
    enddo
end subroutine


subroutine SphereSurfaceAreas(Pos, Radii, Points, BoxL, Areas, NSphere, NPoints, Dim)
    implicit none
    integer, intent(in) :: Dim
    real(8), dimension(NSphere, Dim), intent(in) :: Pos
    real(8), dimension(NSphere), intent(in) :: Radii
    real(8), dimension(NPoints, Dim), intent(in) :: Points
    real(8), dimension(Dim), intent(in) :: BoxL
    real(8), dimension(NSphere), intent(out) :: Areas
    real(8), parameter :: pi = 3.141592653589D0 
    integer, intent(in) :: NSphere, NPoints
    integer :: i, j, k
    real(8), dimension(NPoints,Dim) :: ThisPoints
    real(8) :: AreaPerPoint
    logical, dimension(NPoints) :: Exposed
    real(8), dimension(NSphere) :: RadiiSq
    real(8), dimension(Dim) :: iPos, jPos
    real(8), dimension(Dim) :: iBoxL, Dist
    
    if (Dim /= 3) then
        print *, 'Expecting three dimensions and found', Dim
        stop
    endif
    
    iBoxL = merge(1.d0/BoxL, 0.d0, BoxL > 0.d0)
    Areas = 0.
    RadiiSq = Radii*Radii
    do i = 1, NSphere
        iPos = Pos(i,:)
        AreaPerPoint = 4.*pi*Radii(i)**2 / real(NPoints)
        Exposed = .true.
        do k = 1, NPoints
            ThisPoints(k,:) = Points(k,:) * Radii(i) + iPos
        enddo
        do j = 1, NSphere
            if (i == j) cycle
            jPos = Pos(j,:)
            Dist = jPos - iPos
            Dist = Dist - BoxL * anint(Dist * iBoxL)
            jPos = Dist + iPos
            if (.not. any(Exposed)) exit
            !first check if spheres are far from each other
            if (sum(Dist**2) > (Radii(i) + Radii(j))**2) cycle
            do k = 1, NPoints
                if (.not. Exposed(k)) cycle
                if (sum((ThisPoints(k,:) - jPos)**2) < RadiiSq(j)) Exposed(k) = .false.
            enddo
        enddo
        Areas(i) = AreaPerPoint * real(count(Exposed))
    enddo
end subroutine


subroutine SphereVolumes(Pos, Radii, dx, Volumes, NSphere, Dim)
    implicit none
    integer, intent(in) :: Dim
    real(8), dimension(NSphere, Dim), intent(in) :: Pos
    real(8), dimension(NSphere), intent(in) :: Radii
    real(8),  intent(in) :: dx
    real(8), dimension(NSphere), intent(out) :: Volumes
    integer, intent(in) :: NSphere
    real(8), dimension(NSphere) :: RadiiSq
    real(8) :: minDistSq, DistSq, dV
    integer :: i,j
    real(8), dimension(Dim) :: Pos2, minPos, maxPos
    if (Dim /= 3) then
        print *, 'Expecting three dimensions and found', Dim
        stop
    endif
    RadiiSq = Radii*Radii
    Volumes = 0.
    dV = dx*dx*dx
    minPos = (/minval(Pos(:,1) - Radii), minval(Pos(:,2) - Radii), minval(Pos(:,3) - Radii)/)
    maxPos = (/maxval(Pos(:,1) + Radii), maxval(Pos(:,2) + Radii), maxval(Pos(:,3) + Radii)/)
    maxPos = maxPos + dx * 0.5    
    !first do a coarse grid check to see which spheres are where
    Pos2 = minPos
    do while (all(Pos2 < maxPos))
        j = 0
        minDistSq = huge(1.d0)
        do i = 1, NSphere
            DistSq = sum((Pos(i,:) - Pos2)**2)
            if (DistSq < minDistSq .and. DistSq < RadiiSq(i)) then
                minDistSq = DistSq
                j = i
            endif
        enddo
        if (j > 0) Volumes(j) = Volumes(j) + dV
        Pos2(1) = Pos2(1) + dx
        do i = 1, 2
            if (Pos2(i) >= maxPos(i)) then
                Pos2(i) = minPos(i)
                Pos2(i+1) = Pos2(i+1) + dx
            endif
        enddo
    enddo   
end subroutine


!======== POSITIONS ========


subroutine cubiclattice(N, Dim, Pos)
    integer, intent(in) :: N, Dim
    real(8), dimension(N, Dim), intent(out) :: Pos
    integer :: i, j
    integer, dimension(Dim) :: Counts, MaxCounts
    real(8), dimension(Dim) :: Spc
    real(8) :: Distrib
    OffSet = 0
    Distrib = dble(N)**(1.d0/dble(Dim))
    if (abs(Distrib-int(Distrib))>1.d-12) then
        Distrib = dble(int(Distrib)) + 1.d0
    endif
    MaxCounts = dnint(Distrib)
    Spc = 1.d0 / Distrib
    Counts = 0
    do i = 1, N
        Pos(i,:) = dble(Counts)*Spc - 0.5d0
        Counts(1) = Counts(1) + 1
        do j = 1, (Dim-1), 1
            if (Counts(j) == MaxCounts(j)) then
                Counts(j) = 0
                Counts(j+1) = Counts(j+1) + 1
            endif
        enddo
    enddo
end subroutine 


subroutine fcclattice(N, Dim, Pos)
    integer, intent(in) :: N, Dim
    real(8), dimension(N, Dim), intent(out) :: Pos
    integer :: i, j
    integer :: OffSet
    integer, dimension(Dim) :: Counts, MaxCounts
    real(8), dimension(Dim) :: Spc
    real(8) :: Distrib
    OffSet = 0
    Distrib = (2.d0*dble(N))**(1.d0/dble(Dim))
    if (abs(Distrib-int(Distrib))>1.d-12) then
        Distrib = dble(int(Distrib)) + 1.d0
    endif
    MaxCounts = dnint(Distrib)
    MaxCounts(1) = dnint(Distrib/2.d0)
    Spc = 1.d0 / Distrib
    Counts = 0
    do i = 1, N
        Pos(i,:) = dble(Counts)*Spc - 0.5d0
        Pos(i,1) = dble(2*Counts(1))*Spc(1) - 0.5d0 + dble(OffSet)*Spc(1)
        Counts(1) = Counts(1) + 1
        do j = 1, (Dim-1), 1
            if (Counts(j) == MaxCounts(j)) then
                Counts(j) = 0
                Counts(j+1) = Counts(j+1) + 1
                Offset = 1 - OffSet
            endif
        enddo
    enddo
end subroutine 


subroutine bcclattice(N, Dim, Pos)
    !only works for 3D
    integer, intent(in) :: N, Dim
    real(8), dimension(N, Dim), intent(out) :: Pos
    integer :: i, j
    integer, dimension(Dim) :: OffSet
    integer, dimension(Dim) :: Counts, MaxCounts
    real(8), dimension(Dim) :: Spc
    real(8) :: Distrib
    OffSet = 0
    Distrib = (4.d0 * dble(N))**(1.d0/dble(Dim))
    if (abs(Distrib-int(Distrib))>1.d-12) then
        Distrib = dble(int(Distrib)) + 1.d0
    endif
    MaxCounts = anint(Distrib) 
    MaxCounts(1:2) = anint(Distrib/2.d0)
    Spc = 1.d0 / Distrib
    Counts = 0
    do i = 1, N
        Pos(i,:) = dble(2.d0*Counts)*Spc - 0.5d0 + dble(OffSet)*Spc
        Pos(i,Dim) = dble(Counts(Dim))*Spc(Dim) - 0.5d0
        Counts(1) = Counts(1) + 1
        do j = 1, (Dim-1), 1
            if (Counts(j) == MaxCounts(j)) then
                Counts(j) = 0
                Counts(j+1) = Counts(j+1) + 1
                if (j==Dim-1) Offset(1:Dim-1) = 1. - Offset(1:Dim-1)
            endif
        enddo
    enddo
end subroutine


subroutine minimizepos(Pos, NSteps, ds, N, Dim)
    !only works for 3D
    integer, intent(in) :: N, Dim, NSteps
    real(8), intent(in) :: ds
    real(8), dimension(N, Dim), intent(inout) :: Pos
!f2py intent(in,out,inplace) :: Pos
    real(8), dimension(N, Dim) :: Force
    real(8), dimension(Dim) :: Fij, rij, Posmin, Posmax, Poscen
    real(8) :: dij2
    integer :: i, j, k
    external :: Centroid
    !normalize
    call Centroid(Pos, Poscen, N, Dim)
    do i = 1, N
        Pos(i,:) = Pos(i,:) - PosCen
    enddo
    Posmin = minval(Pos, 1)
    Posmax = maxval(Pos, 1)
    do i = 1, N
        Pos(i,:) = (Pos(i,:) - PosMin) / (PosMax - PosMin) - 0.5d0
    enddo
    !minimize
    do k = 1, NSteps
        !compute forces
        Forces = 0.d0
        do i = 1, N
            !pair forces, as -1/r
            do j = i+1, N
                rij = Pos(j,:) - Pos(i,:)
                rij = rij - dnint(rij)
                dij2 = dot_product(rij, rij)
                Fij = -rij / (dij2 + 0.001d0)
                Force(i,:) = Force(i,:) + Fij
                Force(j,:) = Force(j,:) - Fij
            enddo
        enddo
        !update
        Pos = Pos + ds * Force
    enddo
    !check bounds
    Pos = Pos - dnint(Pos)
end subroutine



!======== RMSD ========
function drms(Pos1, Pos2, NPos, Dim)
    implicit none
    real(8) :: dRMS
    integer, intent(in) :: NPos, Dim
    real(8), dimension(NPos, Dim), intent(in) :: Pos1, Pos2
    integer :: i,j
    real(8) :: d1, d2
    dRMS = 0.
    do i = 1, NPos
        do j = i+1, NPos
            d1 = sqrt(sum((Pos1(i,:)-Pos1(j,:))**2))
            d2 = sqrt(sum((Pos2(i,:)-Pos2(j,:))**2))
            dRMS = dRMS + (d1-d2)**2
         enddo
    enddo
    dRMS = sqrt(dRMS * 2.d0 / (NPos * (NPos - 1)))
end function




!======== CONTACT MAPS AND CLUSTERING ========

subroutine contactmap(Pos, BoxL, Cutoff, CMap, NPos, Dim)
    implicit none
    integer, intent(in) :: NPos, Dim
    real(8), dimension(NPos, Dim), intent(in) :: Pos
    real(8), dimension(Dim), intent(in) :: BoxL
    real(8), intent(in) :: Cutoff
    integer, dimension(NPos, NPos), intent(out) :: CMap
    integer :: i, j
    real(8), dimension(Dim) :: distvec, iBoxL
    real(8) :: distsq, cutsq 
    iBoxL = merge(1.d0/BoxL, 0.d0, BoxL > 0.d0)
    cutsq = Cutoff * Cutoff
    CMap = 0
    do i = 1, NPos
        do j = i+1, NPos
            distvec = Pos(i,:) - Pos(j,:)
            distvec = distvec - BoxL * anint(distvec * iBoxL)
            distsq = sum(distvec * distvec)
            if (distsq <= cutsq) then
                CMap(i,j) = 1.d0
                CMap(j,i) = 1.d0
            endif
        enddo
    enddo            
end subroutine



!======== MINIMUM IMAGING ========

subroutine minimage(Pos, BoxL, ReimagedPos, NPos, Dim)
    implicit none
    integer, intent(in) :: NPos, Dim
    real(8), dimension(NPos, Dim), intent(in) :: Pos
    real(8), dimension(Dim), intent(in) :: BoxL
    real(8), dimension(NPos, Dim), intent(out) :: ReimagedPos
    real(8), dimension(Dim) :: iBoxL
    integer :: i
    iBoxL = merge(1.d0/BoxL, 0.d0, BoxL > 0.d0)
    do i = 1, NPos
        ReimagedPos(i,:) = Pos(i,:) - BoxL * anint(Pos(i,:) * iBoxL)
    enddo            
end subroutine

subroutine minimagen(Pos, BoxL, n, NPos, Dim)
    implicit none
    integer, intent(in) :: NPos, Dim
    real(8), dimension(NPos, Dim), intent(in) :: Pos
    real(8), dimension(Dim), intent(in) :: BoxL
    real(8), dimension(NPos, Dim), intent(out) :: n
    real(8), dimension(Dim) :: iBoxL
    integer :: i
    iBoxL = merge(1.d0/BoxL, 0.d0, BoxL > 0.d0)
    do i = 1, NPos
        n(i,:) = anint(Pos(i,:) * iBoxL)
    enddo            
end subroutine

subroutine reimage(Pos, RefPos, BoxL, ReimagedPos, NPos, Dim)
    implicit none
    integer, intent(in) :: NPos, Dim
    real(8), dimension(NPos, Dim), intent(in) :: Pos
    real(8), dimension(Dim), intent(in) :: RefPos
    real(8), dimension(Dim), intent(in) :: BoxL
    real(8), dimension(NPos, Dim), intent(out) :: ReimagedPos
    integer :: i
    real(8), dimension(Dim) :: distvec, iBoxL
    real(8) :: distsq, cutsq 
    iBoxL = merge(1.d0/BoxL, 0.d0, BoxL > 0.d0)
    do i = 1, NPos
        distvec = Pos(i,:) - RefPos
        distvec = distvec - BoxL * anint(distvec * iBoxL)
        ReimagedPos(i,:) = RefPos + distvec
    enddo            
end subroutine

