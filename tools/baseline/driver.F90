! Standalone Fortran driver for the Noah 3.9 column physics (SFLX) from LISF.
! Reduced-scope baseline for the JAX port: no LIS framework, no I/O layer.
! It mirrors the per-tile preparation done in lis/surfacemodels/land/noah.3.9/noah39_main.F90
! (forcing floors, FFROZP, CALHUM, SFCDIF_OFF, LWDN*EMISS, SOLNET) and calls SFLX.
! Assumptions recorded in jax-port/GOAL.md: opt_thcnd=1, aoasis=1, iz0tlnd=0, ua_phys=false, ice=0.
program noah39_driver
  use module_sf_noah39lsm, only : SFLX, SFCDIF_OFF
  implicit none
  include 'spec_idx.inc'
  real, parameter :: dt = 1800.0, zlvl = 10.0, t0 = 273.15
  integer :: nc, nt, ns, u, c, k, step, rep, nrep, isurban, iz0tlnd
  integer, allocatable :: vegt(:), soiltyp(:), nroot(:)
  real, allocatable :: P(:,:), S0(:,:), L0(:,:,:), F(:,:,:), O(:,:,:)
  real, allocatable :: st(:,:), lay(:,:,:), z0old(:)
  character(len=256) :: fin, fout, arg
  character(len=256) :: llanduse, lsoil
  integer(8) :: c0, c1, cr
  real :: sldpth(4), stc(4), smc(4), sh2o(4), et(4), smav(4)
  real :: ffrozp, lwdn, soldn, solnet, sfcprs, prcp, sfctmp, q2, sfcspd, cosz, prcprain, solardirect
  real :: th2, q2sat, dqsdt2, es, t1v, th2v, ustar
  real :: slope, shdfac, shdmin, shdmax, alb, snoalb, tbot, z0brd, z0, emissi, embrd
  real :: cmc, t1, snowh, sneqv, albedo, ch, cm, evp, eta, sheat, eta_kinematic, fdown
  real :: ec, edir, ett, esnow, drip, dew, beta, etp, ssoil, flx1, flx2, flx3, flx4
  real :: fvb, fbur, fgsn, ztopv, zbotv, snomlt, sncovr, runoff1, runoff2, runoff3
  real :: rc, pc, rsmin, xlai, rcs, rct, rcq, rcsoil, soilw, soilm, q1, snotime1, ribb
  real :: smcwlt, smcdry, smcref, smcmax, hs, rgl, snup, cmcmax, rsmax, topt, bexp, dksat
  real :: dwsat, f1, quartz, psisat, czil, sbeta, fxexp, csoil, salp, kdt, cfactr, zbot
  real :: refkdt, ptu, frzx, sndens, lvcoef, tsoil, sfhead1rt, infxs1rt, etpnd1, aoasis
  real :: xsda_qfx, hfx_phy, qfx_phy, xqnorm, hcpct_fasdas
  integer :: nrootc, vegtc, soiltypc, opt_thcnd, fasdas
  logical :: local, ua_phys, rdlai2d, usemonalb
  real, parameter :: a2 = 17.67, a3 = 273.15, a4 = 29.65, elwv = 2.501e6, e0 = 611.0, rv = 461.0, eps = 0.622

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, arg)
  nrep = 1
  if (len_trim(arg) > 0) read(arg, *) nrep
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  read(u) nc, nt, ns
  if (ns /= 4) stop 'driver is written for 4 soil layers'
  allocate(vegt(nc), soiltyp(nc), nroot(nc), P(nc, n_s), S0(nc, n_i), L0(nc, ns, 3))
  allocate(F(nc, nt, n_f), O(nc, n_o, nt), st(nc, n_i), lay(nc, ns, 3), z0old(nc))
  read(u) vegt
  read(u) soiltyp
  read(u) nroot
  do k = 1, n_s;  read(u) P(:, k);  end do
  do k = 1, n_i;  read(u) S0(:, k); end do
  do k = 1, 3;    read(u) L0(:, :, k); end do   ! stc, smc, sh2o
  do k = 1, n_f
    read(u) F(:, :, k)
  end do
  close(u)
  llanduse = 'USGS'; lsoil = 'STAS'
  local = .false.; ua_phys = .false.; rdlai2d = .true.; usemonalb = .true.
  isurban = 1; iz0tlnd = 0; opt_thcnd = 1; fasdas = 0; aoasis = 1.0
  sldpth = (/0.1, 0.3, 0.6, 1.0/)

  open(newunit=u, file=trim(fout), access='stream', form='unformatted', status='replace')
  call system_clock(c0, cr)
  do rep = 1, nrep
    st = S0; lay = L0; z0old = S0(:, i_z0)
    do step = 1, nt
      do c = 1, nc
        sfctmp = F(c, step, f_sfctmp); q2 = F(c, step, f_q2); soldn = F(c, step, f_soldn)
        sfcspd = F(c, step, f_sfcspd); sfcprs = F(c, step, f_sfcprs); prcp = F(c, step, f_prcp)
        emissi = st(c, i_emissi)
        lwdn = F(c, step, f_lwdn) * emissi
        ffrozp = 0.0
        if (prcp > 0.0 .and. sfctmp < t0) ffrozp = 1.0
        if (sfcspd <= 0.01) sfcspd = 0.01
        if (q2 < 0.1e-5) q2 = 0.1e-5
        es = e0 * exp(elwv / rv * (1.0 / a3 - 1.0 / sfctmp))
        q2sat = eps * es / (sfcprs - (1 - eps) * es)
        dqsdt2 = q2sat * a2 * (a3 - a4) / (sfctmp - a4)**2
        th2 = sfctmp + 0.0098 * zlvl
        t1 = st(c, i_t1)
        t1v = t1 * (1.0 + 0.61 * q2)
        th2v = th2 * (1.0 + 0.61 * q2)
        ch = st(c, i_ch); cm = st(c, i_cm)
        vegtc = vegt(c); soiltypc = soiltyp(c); nrootc = nroot(c)
        czil = P(c, s_czil)
        call SFCDIF_OFF(zlvl, zlvl, z0old(c), t1v, th2v, sfcspd, czil, cm, ch, vegtc, isurban, iz0tlnd, ustar)
        albedo = st(c, i_albedo)
        solnet = soldn * (1.0 - albedo)
        cosz = 0.0; prcprain = 0.0; solardirect = 0.0; ribb = 0.0; ptu = 0.0
        slope = P(c, s_slope); shdfac = P(c, s_shdfac); shdmin = P(c, s_shdmin); shdmax = P(c, s_shdmax)
        alb = P(c, s_alb); snoalb = P(c, s_snoalb); tbot = P(c, s_tbot); z0brd = P(c, s_z0brd)
        z0 = st(c, i_z0); embrd = P(c, s_embrd)
        cmc = st(c, i_cmc); snowh = st(c, i_snowh); sneqv = st(c, i_sneqv)
        stc = lay(c, :, 1); smc = lay(c, :, 2); sh2o = lay(c, :, 3)
        sncovr = st(c, i_sncovr); snotime1 = st(c, i_snotime1); q1 = st(c, i_q1); xlai = st(c, i_xlai)
        ztopv = P(c, s_ztopv); zbotv = P(c, s_zbotv)
        smcwlt = P(c, s_smcwlt); smcdry = P(c, s_smcdry); smcref = P(c, s_smcref); smcmax = P(c, s_smcmax)
        hs = P(c, s_hs); rgl = P(c, s_rgl); snup = P(c, s_snup); cmcmax = P(c, s_cmcmax); rsmax = P(c, s_rsmax)
        topt = P(c, s_topt); bexp = P(c, s_bexp); dksat = P(c, s_dksat); dwsat = P(c, s_dwsat)
        f1 = P(c, s_f1); quartz = P(c, s_quartz); psisat = P(c, s_psisat); sbeta = P(c, s_sbeta)
        fxexp = P(c, s_fxexp); csoil = P(c, s_csoil); salp = P(c, s_salp); kdt = P(c, s_kdt)
        cfactr = P(c, s_cfactr); zbot = P(c, s_zbot); refkdt = P(c, s_refkdt); frzx = P(c, s_frzx)
        lvcoef = P(c, s_lvcoef); rsmin = P(c, s_rsmin)
        sfhead1rt = 0.0; infxs1rt = 0.0; etpnd1 = 0.0
        xsda_qfx = 0.0; hfx_phy = 0.0; qfx_phy = 0.0; xqnorm = 0.0
        sndens = 0.0
        call SFLX(1, 1, ffrozp, isurban, dt, zlvl, ns, sldpth, local, llanduse, lsoil, &
             lwdn, soldn, solnet, sfcprs, prcp, sfctmp, q2, sfcspd, cosz, prcprain, solardirect, &
             th2, q2sat, dqsdt2, vegtc, soiltypc, slope, shdfac, shdmin, shdmax, &
             alb, snoalb, tbot, z0brd, z0, emissi, embrd, &
             cmc, t1, stc, smc, sh2o, snowh, sneqv, albedo, ch, cm, &
             evp, eta, sheat, eta_kinematic, fdown, ec, edir, et, ett, esnow, drip, dew, &
             beta, etp, ssoil, flx1, flx2, flx3, flx4, fvb, fbur, fgsn, ua_phys, ztopv, zbotv, &
             snomlt, sncovr, runoff1, runoff2, runoff3, rc, pc, rsmin, xlai, rcs, rct, rcq, rcsoil, &
             soilw, soilm, q1, smav, rdlai2d, usemonalb, snotime1, ribb, &
             smcwlt, smcdry, smcref, smcmax, nrootc, &
             hs, rgl, snup, cmcmax, rsmax, topt, bexp, dksat, dwsat, f1, quartz, psisat, czil, &
             sbeta, fxexp, csoil, salp, kdt, cfactr, zbot, refkdt, ptu, frzx, sndens, &
             lvcoef, tsoil, sfhead1rt, infxs1rt, etpnd1, opt_thcnd, aoasis, &
             xsda_qfx, hfx_phy, qfx_phy, xqnorm, fasdas, hcpct_fasdas)
        z0old(c) = z0
        ! carry state to next step
        st(c, i_cmc) = cmc; st(c, i_t1) = t1; st(c, i_snowh) = snowh; st(c, i_sneqv) = sneqv
        st(c, i_albedo) = albedo; st(c, i_ch) = ch; st(c, i_cm) = cm; st(c, i_sncovr) = sncovr
        st(c, i_z0) = z0; st(c, i_snotime1) = snotime1; st(c, i_q1) = q1
        st(c, i_emissi) = emissi; st(c, i_xlai) = xlai
        lay(c, :, 1) = stc; lay(c, :, 2) = smc; lay(c, :, 3) = sh2o
        if (.true.) then
          O(c, o_cmc, step) = cmc; O(c, o_t1, step) = t1
          O(c, o_stc1, step) = stc(1); O(c, o_stc2, step) = stc(2); O(c, o_stc3, step) = stc(3); O(c, o_stc4, step) = stc(4)
          O(c, o_smc1, step) = smc(1); O(c, o_smc2, step) = smc(2); O(c, o_smc3, step) = smc(3); O(c, o_smc4, step) = smc(4)
          O(c, o_sh2o1, step) = sh2o(1); O(c, o_sh2o2, step) = sh2o(2); O(c, o_sh2o3, step) = sh2o(3); O(c, o_sh2o4, step) = sh2o(4)
          O(c, o_snowh, step) = snowh; O(c, o_sneqv, step) = sneqv; O(c, o_albedo, step) = albedo; O(c, o_ch, step) = ch
          O(c, o_cm, step) = cm; O(c, o_sncovr, step) = sncovr; O(c, o_z0, step) = z0; O(c, o_snotime1, step) = snotime1
          O(c, o_q1, step) = q1; O(c, o_emissi, step) = emissi; O(c, o_xlai, step) = xlai; O(c, o_shdfac, step) = shdfac
          O(c, o_eta, step) = eta; O(c, o_sheat, step) = sheat; O(c, o_ssoil, step) = ssoil; O(c, o_etp, step) = etp
          O(c, o_edir, step) = edir; O(c, o_ec, step) = ec; O(c, o_ett, step) = ett; O(c, o_esnow, step) = esnow
          O(c, o_runoff1, step) = runoff1; O(c, o_runoff2, step) = runoff2; O(c, o_runoff3, step) = runoff3
          O(c, o_snomlt, step) = snomlt; O(c, o_drip, step) = drip; O(c, o_dew, step) = dew; O(c, o_beta, step) = beta
          O(c, o_fdown, step) = fdown; O(c, o_rc, step) = rc; O(c, o_pc, step) = pc; O(c, o_soilm, step) = soilm
          O(c, o_soilw, step) = soilw; O(c, o_tsoil, step) = tsoil
        end if
      end do
    end do
  end do
  call system_clock(c1)
  do step = 1, nt
    write(u) O(:, :, step)
  end do
  close(u)
  write(*, '(a,i0,a,i0,a,i0,a,f12.6,a)') 'columns=', nc, ' steps=', nt, ' reps=', nrep, &
       ' wall_total=', real(c1 - c0) / real(cr), ' s'
  write(*, '(a,f14.9,a)') 'wall_per_day=', real(c1 - c0) / real(cr) / real(nrep), ' s'
end program noah39_driver
