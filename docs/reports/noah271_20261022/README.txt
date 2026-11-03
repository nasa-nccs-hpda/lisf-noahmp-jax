Noah 2.7.1 port (2026-10-22). Owner: G. Tamkin. Written by a Claude Code session. DRAFT. Project-local.

Files
  make_noah271_tables.py   builds the Noah 2.7.1 vegetation and soil parameter files from the public Noah 3.3 tables
  (code)                   noah271_jax.py (SFLX and the routines it calls), lisjax/params271.py, lisjax/model271.py,
                           lis/replay271.py (open-loop replay), lis_patches/make_noah271_hook.py (Fortran dump hook, build copy only)
  run directory            dev/lisf_run/conus_n271 (6 h), conus_n271_2d (2 d), conus_n271_dump (full-domain call dump, 6 h, 1.2 GB; not in the repository)

What is NOT the real LIS data set
  The public LIS parameter set has no Noah 2.7.1 tables. The tables here are DERIVED from the Noah 3.3 tables
  (MODIFIED_IGBP_MODIS_NOAH vegetation: NROOT RSMIN RGL HS SNUP; Z0 = mean of Z0MIN and Z0MAX; LAI = mean of LAIMIN and LAIMAX;
  STAS soil: MAXSMC SATPSI SATDK BB QTZ ...; CZIL 0.1). The Fortran reference and the JAX port read the same files, so the
  comparison tests the code, not the parameter values. Results are not valid as a Noah 2.7.1 simulation of the real world.

A Fortran quirk reproduced on purpose
  noah271_f2t.F90 reads Wind_E and Wind_N at index t (tile number) and tests precipitation validity at index t, but reads all other
  forcing at tile_id (grid cell number). With a land mask tile_id differs from t, so tile t gets the wind of grid cell number t
  (row-major in the 224 x 464 grid; undefined = -9999 over the ocean, which gives a wind speed near 14,000 m/s and large exchange
  coefficients), and tile t gets no precipitation when grid cell t is outside the NLDAS-2 mask. Found from the dump (record shows
  tid 103 for tile 1 and a wind speed of 10,854 m/s). The JAX port reproduces this (NLDAS2(alt=...) in lisjax/forcing.py;
  Setup271.f2t_alt). Any real Noah 2.7.1 run on a masked domain in this LIS version has the same problem.

Method and results: see STATUS_20261022.md and LOG.md (entry "NOAH 2.7.1").
