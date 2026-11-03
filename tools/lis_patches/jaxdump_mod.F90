! Debug/validation hook (not part of LISF): writes one record per Noah-3.9 SFLX call so the JAX
! port can replay real LIS inputs. Added by the LISF-to-JAX project; see jax-port/lis_patches/.
module jaxdump_mod
  implicit none
  integer, parameter :: JD_NHDR = 9, JD_NPRE = 100, JD_NPOST = 60
  integer :: jd_unit = 0
  logical :: jd_open = .false.
  logical :: jd_checked = .false., jd_only_glacier = .false.
  integer :: jd_maxtile = huge(1)
contains
  subroutine jaxdump_write(hdr, pre, post)
    integer, intent(in) :: hdr(JD_NHDR)
    real, intent(in) :: pre(JD_NPRE), post(JD_NPOST)
    character(len=16) :: ev
    integer :: ln
    if (.not. jd_checked) then
      call get_environment_variable('JAXDUMP_ONLY_GLACIER', ev, ln)
      jd_only_glacier = (ln > 0)
      call get_environment_variable('JAXDUMP_MAXTILE', ev, ln)
      if (ln > 0) read(ev, *) jd_maxtile
      jd_checked = .true.
    end if
    if (jd_only_glacier .and. hdr(6) /= 99) return
    if (hdr(7) > jd_maxtile) return
    if (.not. jd_open) then
      open(newunit=jd_unit, file='jaxdump_noah39.bin', access='stream', form='unformatted', status='replace')
      write(jd_unit) JD_NHDR, JD_NPRE, JD_NPOST
      jd_open = .true.
    end if
    write(jd_unit) hdr, pre, post
  end subroutine jaxdump_write
end module jaxdump_mod
