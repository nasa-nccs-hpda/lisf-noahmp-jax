! Debug/validation hook (not part of LISF): records, per Noah-MP 4.0.1 NOAHMP_SFLX call, the inputs/states
! (pre record) and the states/outputs (post record) as float32, with a self-describing layout file.
! Added by the LISF-to-JAX project; see jax-port/lis_patches/.
module jaxdump_gla_mod
  implicit none
  integer, parameter :: JDG_MAX = 4000
  real    :: jdg_buf(JDG_MAX)
  character(len=24) :: jdg_nm(JDG_MAX)
  integer :: jdg_sz(JDG_MAX)
  integer :: jdg_n = 0, jdg_pos = 0
  integer :: jdg_unit = 0
  logical :: jdg_open = .false., jdg_layout_done(2) = .false., jdg_keep = .true.
  interface jdg_add
    module procedure jdg_add_r, jdg_add_i, jdg_add_l, jdg_add_rv
  end interface
contains
  logical function jdg_snowonly()
    logical, save :: checked = .false., val = .false.
    character(len=8) :: ev
    integer :: ln
    if (.not. checked) then
      call get_environment_variable('JAXDUMP_SNOWONLY', ev, ln)
      val = (ln > 0)
      checked = .true.
    end if
    jdg_snowonly = val
  end function jdg_snowonly
  subroutine jdg_begin()
    jdg_n = 0; jdg_pos = 0
  end subroutine jdg_begin
  subroutine jdg_push(name, v, k)
    character(len=*), intent(in) :: name
    real, intent(in) :: v(:)
    integer, intent(in) :: k
    jdg_n = jdg_n + 1
    jdg_nm(jdg_n) = name
    jdg_sz(jdg_n) = size(v)
    jdg_buf(jdg_pos+1:jdg_pos+size(v)) = v
    jdg_pos = jdg_pos + size(v)
  end subroutine jdg_push
  subroutine jdg_add_r(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x
    call jdg_push(name, (/ x /), 1)
  end subroutine jdg_add_r
  subroutine jdg_add_i(name, x)
    character(len=*), intent(in) :: name
    integer, intent(in) :: x
    call jdg_push(name, (/ real(x) /), 1)
  end subroutine jdg_add_i
  subroutine jdg_add_l(name, x)
    character(len=*), intent(in) :: name
    logical, intent(in) :: x
    call jdg_push(name, (/ merge(1.0, 0.0, x) /), 1)
  end subroutine jdg_add_l
  subroutine jdg_add_rv(name, x)
    character(len=*), intent(in) :: name
    real, intent(in) :: x(:)
    call jdg_push(name, x, 1)
  end subroutine jdg_add_rv
  ! which = 1 (pre) or 2 (post)
  subroutine jdg_flush(which)
    integer, intent(in) :: which
    integer :: i, lu
    if (.not. jdg_keep) return
    if (.not. jdg_open) then
      open(newunit=jdg_unit, file='jaxdump_gla.bin', access='stream', form='unformatted', status='replace')
      jdg_open = .true.
    end if
    if (.not. jdg_layout_done(which)) then
      open(newunit=lu, file=merge('jaxdump_gla_pre.layout ', 'jaxdump_gla_post.layout', which == 1), status='replace')
      do i = 1, jdg_n
        write(lu, '(a,1x,i0)') trim(jdg_nm(i)), jdg_sz(i)
      end do
      close(lu)
      jdg_layout_done(which) = .true.
    end if
    write(jdg_unit) which, jdg_pos
    write(jdg_unit) jdg_buf(1:jdg_pos)
  end subroutine jdg_flush
end module jaxdump_gla_mod
