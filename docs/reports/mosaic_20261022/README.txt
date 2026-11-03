Mosaic LSM port: setup and checks of 2026-10-22. Owner: G. Tamkin. Written by a Claude Code session. DRAFT. CONUS NLDAS-2, 0.125 deg, 76,064 tiles, Feb 2017.

Files here: ldt.config and lis.config (copies from dev/lisf_run/conus_mos), make_mos_hook.py (dump hook for the build copy of mos_main.F90), replay_mos.py (open-loop replay).
Code: mosaic_jax.py (MOSTILE and callees, umd_sibalb, m_pmonth), lisjax/run_mos.py (setup, ASTRO zenith angle, forcing, step, history), lisjax/mosaic_sibalb.npy (SiB albedo coefficient table made by running mapsib2umd.F90 and dumping sib%ALVDR... : 6 x 13 x 2 x 14).

Reference setup (all public LIS data, nothing changed in the LIS source):
  - LDT: Land surface model Mosaic; UMD dominant landcover and mask from NLDAS_0.125/umdveg_dom_nldas (source AVHRR, classification UMD); STATSGO-FAO 25 km texture;
    AVHRR LAI and SAI from NLDAS_0.125; FAO porosity (3 layers) from UMD/1KM/porosity_FAO.L1-L3 (unzipped to dev/lisf_data/FAO_porosity, 6.5 GB, outside the repository).
    Two edits of the LDT output: 643 land cells with texture class 16 (land ice) set to class 6 because the 13-class Mosaic soil table has no class 16. (A first attempt with a 2-D constant porosity gave NaN soil moisture in layers 2 and 3 in the real LIS: LIS reads 3 porosity layers.)
  - LIS: Mosaic tables mosaic_vegparms_umd.txt, mosaic_monthlyvegparms_umd.txt, mosaic.soilparms.nldas (13 classes), albedo from the SiB coefficients (Albedo data source none), output table MODEL_OUTPUT_LIST.TBL.mosaic. Qg is written with the table's sign convention (DN) = -ghflux.
  - ACond output is not written: mos_main.F90 outputs the uninitialised variable `tempac`; BR/EF are ratios that explode when the latent heat is near zero. Not compared.

Checks:
  1. Parameters built by SetupMos equal the dump of the real LIS exactly (vegt, vegp(24), soilp(10); first 2 steps, 76,064 tiles).
  2. Time-dependent inputs: greenness, LAI+SAI and the four monthly-table parameters bit-identical to the dump (steps 1, 6, 24); cosine of zenith angle (ASTRO in single precision) max 9e-8.
  3. Open-loop replay (replay_mos.py, 2 steps, Fortran state in): the arguments of MOSTILE built by the JAX preparation and all outputs at float32 rounding (e.g. Qle p99 6e-3 W/m2 of 396 range after one call, hlwup 2e-4, ghflux 3e-3, water layers 1e-6).
  4. Closed loop 6 h and 2 d vs real LIS (CPU JAX): see STATUS_20261022.md section 2 and Reports/BENCHMARKS.md.
Limits: one setup (NLDAS-2, UMD classes, SiB albedo, no albedo/LAI DA), Feb 2017; no restart or DA ported; Mosaic output variables not listed above are not written.
