Noah-MP 3.6 port (2026-10-22). Owner: G. Tamkin. Written by a Claude Code session. DRAFT. Project-local.

Method: noahmp36_jax.py is a copy of the Noah-MP 4.0.1 port (noahmp401_jax.py) edited routine by routine against
lis/surfacemodels/land/noahmp.3.6/module_sf_noahmplsm_36.F90. The differences were found with a normalised per-routine diff (comments and
blanks removed, PARAMETERS% and (VEGTYP) indexing removed, upper case); about 40 of the 50 routines are identical after that.

Real differences of 3.6 relative to 4.0.1 that the port implements
  driver      No QSNOW/FWET/CANLIQ/CANICE update before ENERGY: ENERGY uses the values of the previous step (state); TV is clamped to TG + 20 K
              before the call; Julian day for the declination uses the whole hour only (calc_declin_36); no first-step EAH/TAH/CM/CH fix-ups.
  ATM         QPRECC = 0.1 PRCP, QPRECL = 0.9 PRCP; no rain/snow partition, no OPT_SNF 4 (hail/graupel).
  CANWATER    Does the partition (FPICE thresholds TFRZ+2.5/+0.5/+2, BDFALL = MAX(120, ...)), the canopy interception, throughfall and drip
              (no buried-canopy case, no ice drip term) and returns QRAIN, QSNOW, SNOWHIN, FPICE. No PRECIP_HEAT, no PAH terms.
  PHENOLOGY   no PGS, no crops; FB without the SNOWHC > 1e-6 guard.
  ENERGY      FVEG floor 0.01 (4.0.1: 0.05); bare-ground roughness 0.01 (0.002); FSNO without the FMELT < 1e-6 branch; melt factor M = 2.5;
              one RSURF formula; no PET calls (a single VEGE_FLUX call supplies T2MV, Q2V, CHV2, CHLEAF, CHUC, RSSUN, RSSHA).
  VEGE_FLUX   UC = UR LOG(HCAN/Z0M)/LOG(ZLVL/Z0M) without the displacement height; RAGRB without the RB >= 100 s/m bound.
  ALBEDO      CLASS snow albedo: 0.55 + (ALBOLD - 0.55) EXP(-0.01 DT/3600), then + MIN(QSNOW DT, SWEMX) (0.84 - ALB)/SWEMX.
  SOILWATER   SMC is not updated inside the fine sub-steps (conductivities use the SMC of the start of the step); SSTEP does not redistribute
              excess water upwards (the excess left in layer 1 goes to the surface runoff).
  SNOWH2O, SNOWWATER, PHASECHANGE   snow depth bookkeeping (BDSNOW based updates; SNOWH recomputed from the layers; density cap).
Not ported: ICE = 1 branch, lake (IST = 2), OPT_STC 2 and 3, OPT_SNF 4, runoff options 2-5, crop/GECROS, carbon (dveg 2, 5), albedo DA (UPD_ALB_FLAG).
Parameters: MPTABLE.TBL (noah_mp_<scheme>_parameters), VEGPARM.TBL (NROOT RS RGL HS TOPT RSMAX), SOILPARM.TBL.WRF-3.9, GENPARM.TBL; every parameter
entry (about 100 per tile) of the 76,180 CONUS tiles is bit-identical to the Fortran (FRZX differs only for soil type 14, where the Fortran keeps a stale value
that is never used); the coldstart state is identical.
Fortran quirks: the LSM id string is "NoahMP.3.6" while the configuration keys start with "Noah-MP.3.6"; the supplied output list needs
Albedo, Snowcover, RelSMC, RHMin (and Qs, Qsb with kg/m2 units) removed (MODEL_OUTPUT_LIST.TBL.noahmp36 in the case directory).
Run recipe: dev/lisf_run/conus_mp36 (6 h), conus_mp36_2d (2 d pair), conus_mp36_dump (3 h call dump, 1.4 GB, hook lis_patches/make_noahmp36_hook.py);
replay: MPVER=36 python3 lis/replay_mp.py DUMPDIR [n] [start].
