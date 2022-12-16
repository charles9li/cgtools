!======MODULE CODE FOR maplib======


subroutine Map(AtomMap1, Weight, StartInd1, NAtom2, Pos1, BoxL, Pos2, &
    & NAtomMap1, NStartInd1, Dim, NAtom1)
    implicit none
    integer, intent(in) :: NAtomMap1, NStartInd1, Dim, NAtom1, NAtom2
    integer, dimension(0:NAtomMap1-1), intent(in) :: AtomMap1
    real(8), dimension(0:NAtomMap1-1), intent(in) :: Weight
    integer, dimension(0:NStartInd1-1), intent(in) :: StartInd1
    real(8), dimension(0:NAtom1-1, 0:Dim-1), intent(in) :: Pos1
    real(8), dimension(0:Dim-1), intent(in) :: BoxL
    real(8), dimension(0:NAtom2-1, 0:Dim-1), intent(out) :: Pos2
    real(8), dimension(Dim) :: iBoxL, Pos1reimaged, Pos1ref
    integer :: i, j, ind1, ind2
    Pos2 = 0.d0
    iBoxL = merge(1.d0/BoxL, 0.d0, BoxL > 0.d0)
    do i = 0, NStartInd1 - 2
        ind1 = StartInd1(i)
        ind2 = StartInd1(i+1) - 1
        if (ind1 < 0 .or. ind1 > NAtomMap1 - 1 .or. &
          & ind2 < 0 .or. ind2 > NAtomMap1 - 1) cycle
        do j = ind1, ind2
            Pos1reimaged = Pos1(AtomMap1(j), :)
            if (j == ind1) then
                Pos1ref = Pos1reimaged
            else
                !reimage all other positions with respect to the last one
                Pos1reimaged = Pos1reimaged - Pos1ref 
                Pos1reimaged = Pos1reimaged - BoxL * anint(Pos1reimaged * iBoxL)
                Pos1reimaged = Pos1reimaged + Pos1ref
            endif
            Pos2(i,:) = Pos2(i,:) + Weight(j) * Pos1reimaged
        enddo
        Pos2(i, :) = Pos2(i,:) - BoxL * anint(Pos2(i,:) * iBoxL)
    enddo
end subroutine
