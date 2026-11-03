CLSM F2.5: how the large end-to-end maxima were localised (2026-10-22). Owner: G. Tamkin. Written by a Claude Code session. DRAFT.

Question: why does the 6 h CONUS CLSM run differ from the Fortran by up to 112 W/m2 in latent heat when the core agrees at float32 rounding?

Data: full-domain call dump of the real LIS (76,088 tiles, steps 1-3) made with the hooks lis_patches/make_clsm_hook.py (call arguments before/after clsmf25) and make_clsm_inter_hook.py (50 intermediate arrays after the three energy balances). Run directory dev/lisf_run/conus_clsm/d6i (not in the repository, 367 MB).

Scripts here (copies of the session's scratch scripts; run from jax-port/ with JAX_PLATFORMS=cpu):
  cmp_inter.py DIR STEP   JAX core on the Fortran pre-state; compares 50 intermediates and the final state/fluxes; lists worst tiles.
  sens.py                 same step-2 state with uniform noise of +-1 and +-8 J/m2 added to the 6 ground-heat layers; reports the change of tc2, hlatn, shflux, tsurf.
  (the driver-input comparison used a scratch variant of the driver; see LOG.md 2026-10-22 ~08:30)

Results (single run each):
  - cmp_inter.py step 2: all intermediates agree at float32 rounding (largest: tc2sf 5.5e-4 K, shflux2 0.12 W/m2 on one tile; ratios of huge resistances excluded).
  - sens.py: +-1 J/m2 noise: tc2 change up to 3.4 K on 199 tiles above 0.05 K; +-8 J/m2: up to 2.9 K, 748 tiles; hlatn max 116 W/m2, 756 tiles above 2 W/m2; shflux max 82 W/m2; tsurf max 2.8 K on 65 tiles.
  - end-to-end 6 h run: Qle max 112 W/m2, 1,339 tiles above 2 W/m2.

Limits: step 2 only; CLSM only; noise is uniform, not the true rounding pattern; the sensitive tiles are cold (tc2 258-262 K) with ground heat near 3.1e6 J/m2; why that branch amplifies is not analysed. Not tested for Noah-MP.
