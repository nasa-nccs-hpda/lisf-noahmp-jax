"""Debug hook (not part of LISF): dump the forcing perturbation arrays after every get_pert call of the GMAO scheme."""
import sys
BUILD = '/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_build/lis/dataassim/perturb/gmaopert/gmaopert_Mod.F90'
src = open(BUILD).read()
if 'jdp_dump' in src:
    sys.exit('already applied')
sub = '''
    subroutine jdp_dump(npert, nx, ny, ne, pert, ntrm, rseed)
      implicit none
      integer, intent(in) :: npert, nx, ny, ne
      real, intent(in) :: pert(npert,nx,ny,ne), ntrm(npert,nx,ny,ne)
      integer, intent(in) :: rseed(NRANDSEED,nx,ny,ne)
      integer, save :: u = 0
      logical, save :: opened = .false.
      if (.not. opened) then
        open(newunit=u, file='jaxdump_pert.bin', access='stream', form='unformatted', status='replace')
        write(u) npert, nx, ny, ne
        opened = .true.
      end if
      write(u) pert
      write(u) ntrm
      write(u) rseed(1,:,:,:)
      flush(u)
    end subroutine jdp_dump

'''
k = src.index('    subroutine gmaoPert_init(index)')
src = src[:k] + sub + src[k:]
# after the setup get_pert (initialize_ntrmdt=.true.) for forcing
mk = 'initialize_rseed=.true.,    &\n                    initialize_ntrmdt=.true.            )\n               deallocate(init_Pert_rseed)'
assert mk in src
src = src.replace(mk, mk.replace('deallocate(init_Pert_rseed)', 'call jdp_dump(LIS_rc%nforcepert, LIS_rc%lnc(nest), LIS_rc%lnr(nest), LIS_rc%nensem(nest), forcPert(nest,k)%Forcepert, forcPert(nest,k)%Forcepert_ntrmdt, forcPert(nest,k)%Forcepert_rseed)\n               deallocate(init_Pert_rseed)'), 1)
# in gmaoperturb id==1 non-xy branch
mk2 = '                 forcPert(n,k)%Forcepert)\n            \n            do i=1,objcount'
assert mk2 in src
src = src.replace(mk2, '                 forcPert(n,k)%Forcepert)\n            call jdp_dump(LIS_rc%nforcepert, LIS_rc%lnc(n), LIS_rc%lnr(n), LIS_rc%nensem(n), forcPert(n,k)%Forcepert, forcPert(n,k)%Forcepert_ntrmdt, forcPert(n,k)%Forcepert_rseed)\n            do i=1,objcount', 1)
open(BUILD, 'w').write(src)
print('ok')
