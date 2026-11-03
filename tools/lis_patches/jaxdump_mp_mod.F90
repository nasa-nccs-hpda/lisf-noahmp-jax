! Debug/validation hook (not part of LISF): records, per Noah-MP 4.0.1 NOAHMP_SFLX call, the inputs/states
! (pre record) and the states/outputs (post record) as float32, with a self-describing layout file.
! Added by the LISF-to-JAX project; see jax-port/lis_patches/.
module jaxdump_mp_mod
  implicit none
  integer, parameter :: JDM_MAX = 4000
  real    :: jdm_buf(JDM_MAX)
  character(len=24) :: jdm_nm(JDM_MAX)
  integer :: jdm_sz(JDM_MAX)
  integer :: jdm_n = 0, jdm_pos = 0
  integer :: jdm_unit = 0
  logical :: jdm_open = .false., jdm_layout_done(2) = .false., jdm_keep = .true.
  interface jdm_add
    module procedure jdm_add_r, jdm_add_i, jdm_add_l, jdm_add_rv
  end interface
contains
  logical function jdm_noveg()
    logical, save :: checked = .false., val = .false.
    character(len=8) :: ev
    integer :: ln
    if (.not. checked) then
      call get_environment_variable('JAXDUMP_NOVEG', ev, ln)
      val = (ln > 0)
      checked = .true.
    end if
    jdm_noveg = val
  end function jdm_noveg
  logical function jdm_snowonly()
    logical, save :: checked = .false., val = .false.
    character(len=8) :: ev
    integer :: ln
    if (.not. checked) then
      call get_environment_variable('JAXDUMP_SNOWONLY', ev, ln)
      val = (ln > 0)
      checked = .true.
    end if
    jdm_snowonly = val
  end function jdm_snowonly
  subroutine jdm_begin()
    jdm_n = 0; jdm_pos = 0
  end subroutine jdm_begin
  subroutine jdm_push(name, v, k)
    character(len=*), intent(in) :: name
    real, intent(in) :: v(:)
    integer, intent(in) :: k
    jdm_n = jdm_n + 1
    jdm_nm(jdm_n) = name
    jdm_sz(jdm_n) = size(v)
    jdm_buf(jdm_pos+1:jdm_pos+size(v)) = v
    jdm_pos = jdm_pos + size(v)
  end subroutine jdm_push
  subroutine jdm_add_r(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x
    call jdm_push(name, (/ x /), 1)
  end subroutine jdm_add_r
  subroutine jdm_add_i(name, x)
    character(len=*), intent(in) :: name
    integer, intent(in) :: x
    call jdm_push(name, (/ real(x) /), 1)
  end subroutine jdm_add_i
  subroutine jdm_add_l(name, x)
    character(len=*), intent(in) :: name
    logical, intent(in) :: x
    call jdm_push(name, (/ merge(1.0, 0.0, x) /), 1)
  end subroutine jdm_add_l
  subroutine jdm_add_rv(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x(:)
    call jdm_push(name, x, 1)
  end subroutine jdm_add_rv
  ! which = 1 (pre) or 2 (post)
  subroutine jdm_flush(which)
    integer, intent(in) :: which
    integer :: i, lu
    if (.not. jdm_keep) return
    if (.not. jdm_open) then
      open(newunit=jdm_unit, file='jaxdump_mp.bin', access='stream', form='unformatted', status='replace')
      jdm_open = .true.
    end if
    if (.not. jdm_layout_done(which)) then
      open(newunit=lu, file=merge('jaxdump_mp_pre.layout ', 'jaxdump_mp_post.layout', which == 1), status='replace')
      do i = 1, jdm_n
        write(lu, '(a,1x,i0)') trim(jdm_nm(i)), jdm_sz(i)
      end do
      close(lu)
      jdm_layout_done(which) = .true.
    end if
    write(jdm_unit) which, jdm_pos
    write(jdm_unit) jdm_buf(1:jdm_pos)
  end subroutine jdm_flush
end module jaxdump_mp_mod
