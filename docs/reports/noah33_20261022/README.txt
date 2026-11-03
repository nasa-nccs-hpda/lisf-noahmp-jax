Noah 3.3 port: checks made on 2026-10-22. Owner: G. Tamkin. Written by a Claude Code session. DRAFT. CONUS NLDAS-2, 0.125 deg, 76,180 tiles, Feb 2017.

Port: Noah 3.3 = the Noah 3.6 path of noah39_jax.sflx (v36) with v33=True: no UA snow physics, canopy resistance computed whenever SHDFAC > 0
(RC = inf and PC = 0 for LAI = 0), no land-ice branch (ice only for water tiles / xice = 1, none in this domain), CZIL and CMCMAX per vegetation class
from the vegetation table (columns 16-17), FXEXP and SBETA per soil class from the soil table (columns 11-12). Tables: /css/lisdata-pub/data/PARAMETERS/noah33_parms.
Setup: same LDT file as Noah 3.9 (dev/lisf_run/conus/out/lis_input.conus.noah39.d01.nc); Fortran reference dev/lisf_run/conus_n33 (6 h), conus_n33_2d (2 days).

1. Closed loop vs real LIS (JAX CPU): 6 h Qle/Qh/Qg p99 0.052/0.16/0.22 W/m2; 2 days Qle/Qh/Qg p99 0.19/0.57/0.84 W/m2 (max 4.2/11/16), SWE p99 8.7e-5 kg/m2,
   soil moisture p99 5.8e-5, soil temperature p99 0.020 K. Tiles with Qh error > 1 W/m2 at hour 48: 525 of 76,180 (0.7%), mostly open shrubland (veg 7) and grassland (veg 10), 12% under snow.
2. Open-loop replay of the real calls (replay33.py; dump from make_noah33_hook.py; 304,720 calls each for steps 69-72 of the cold run and 25-28, 45-48 of a restart run):
   state and fluxes at float32 rounding (sheat p99 1.8e-3, max 0.72 W/m2 on one call; t1 p99 3e-5 K; eta p99 5e-5). RC and PC (diagnostics, not used afterwards when LAI = 0) differ on a few tiles.
3. Sensitivity (sensitivity_run.py): JAX rerun with a 1e-6 relative perturbation of the initial soil and skin temperatures and moistures vs the unperturbed JAX run:
   Qle/Qh/Qg p99 0.17/0.51/0.75 W/m2 over the 2 days, the same size as JAX vs Fortran. So the closed-loop difference equals the model's own sensitivity to rounding-level changes.

Limits: one 2-day window; Noah 3.3 water tiles (ice = 1) and xice not exercised; coldstart only (no restart read of the Fortran Noah 3.3 restart file tested in JAX).

Noah 3.2 (same day, same setup; Fortran references dev/lisf_run/conus_n32 (6 h) and conus_n32_2d (2 d), tables /css/lisdata-pub/data/PARAMETERS/noah32_parms):
  Noah 3.2 = the Noah 3.3 path (v33) with: constant CZIL (general table) and CMCMAX (vegetation table tail) instead of per-class columns; and the wind handling of noah32_main.F90:
  when the temperature height zh (2 m) is below the wind height zm (10 m) the wind speed is scaled by ZFAC = ln((zh+z0)/z0) / ln((zm+z0)/z0) (z0 = current roughness length), floored at 0.01 m/s,
  and SFCDIF_OFF is called with one height. Without the ZFAC step the 6 h mismatch was Qle p99 57 W/m2; with it Qle/Qh/Qg p99 0.049/0.16/0.21 W/m2 (6 h) and 0.19/0.58/0.84 W/m2 (2 d), the Noah 3.3 level.
  Not tested: the veg-table "MODI" section of noah32_parms was used as it is; no replay of real Noah 3.2 calls was made (the Noah 3.3 hook was not ported to noah32_main).
