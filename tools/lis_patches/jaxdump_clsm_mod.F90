! Debug/validation hook (not part of LISF): records, for a stride-sampled set of catchment tiles, every argument of the
! CLSM F2.5 core call (clsmf25) before (tag 0) and after (tag 1) the call, as big-endian float32 stream records.
! Added by the LISF-to-JAX project; see jax-port/lis_patches/make_clsm_hook.py.
module jaxdump_clsm_mod
  implicit none
  integer :: jdc_unit = 0, jdc_step = 0, jdc_ns = 0, jdc_maxstep = 200
  integer, allocatable :: jdc_idx(:)
  logical :: jdc_active = .false., jdc_ready = .false.
contains
  subroutine jdc_begin(ncat, tag)
    integer, intent(in) :: ncat, tag
    character(len=16) :: ev
    integer :: ln, stride, k
    if (.not. jdc_ready) then
      jdc_ready = .true.
      call get_environment_variable('JAXDUMP_CLSM_STRIDE', ev, ln)
      stride = 25
      if (ln > 0) read(ev,*) stride
      call get_environment_variable('JAXDUMP_CLSM_MAXSTEP', ev, ln)
      if (ln > 0) read(ev,*) jdc_maxstep
      jdc_ns = (ncat - 1) / stride + 1
      allocate(jdc_idx(jdc_ns))
      do k = 1, jdc_ns
        jdc_idx(k) = 1 + (k-1)*stride
      end do
      open(newunit=jdc_unit, file='jaxdump_clsm.bin', form='unformatted', access='stream', status='replace')
      write(jdc_unit) jdc_ns
      write(jdc_unit) jdc_idx
      jdc_active = .true.
    end if
    if (tag == 0) jdc_step = jdc_step + 1
    if (jdc_step > jdc_maxstep) jdc_active = .false.
    if (jdc_active) write(jdc_unit) tag, jdc_step
  end subroutine jdc_begin
  subroutine jdc_s(x)
    real, intent(in) :: x
    if (jdc_active) write(jdc_unit) x
  end subroutine jdc_s
  subroutine jdc_v1(a)
    real, intent(in) :: a(:)
    if (jdc_active) write(jdc_unit) a(jdc_idx)
  end subroutine jdc_v1
  subroutine jdc_i1(a)
    integer, intent(in) :: a(:)
    if (jdc_active) write(jdc_unit) real(a(jdc_idx))
  end subroutine jdc_i1
  subroutine jdc_k(a)     ! (k, ncat)
    real, intent(in) :: a(:,:)
    if (jdc_active) write(jdc_unit) a(:, jdc_idx)
  end subroutine jdc_k
  subroutine jdc_t(a)     ! (ncat, k)
    real, intent(in) :: a(:,:)
    if (jdc_active) write(jdc_unit) a(jdc_idx, :)
  end subroutine jdc_t
end module jaxdump_clsm_mod
