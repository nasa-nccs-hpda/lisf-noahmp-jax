#!/bin/bash
# Noah-MP 5.0 restart round trip on CONUS (76,180 tiles incl. land ice): JAX writes a restart after day 1, Fortran LIS and JAX continue from it for 6 h; reference = one continuous Fortran run to Feb 2 06:00.
#SBATCH --job-name=mp5_rt
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=$LISFJAX_REPO/gpu_runs/conus_mp5_rt_%j.out
JP=$LISFJAX_REPO
RUN=$LISF_RUN/conus_mp5_rt
CONDA=$CONDA_BIN
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
LIS=$LISF_RUN/LIS_mp_clean
jax() { (cd $JP; $CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/$1 | grep -E "TIMING|rror" | cut -c1-200); }
echo "=== JAX day 1 with restart output"; jax cfg.j_a; ls $RUN/j_a/SURFACEMODEL/201702/
echo "=== JAX from its own restart, 6 h"; jax cfg.j_c
echo "=== Fortran from the JAX restart, 6 h"
( source $LISF_DEPS/env_lis.sh; cd $RUN; $LIS -f cfg.f_b > f_b.out 2>&1; tail -2 f_b/lislog.0000 )
echo "=== Fortran continuous to Feb 2 06:00"
( source $LISF_DEPS/env_lis.sh; cd $RUN; $LIS -f cfg.f_full > f_full.out 2>&1; tail -2 f_full/lislog.0000 )
echo "=== compare (hours 1-6 of Feb 2): Fortran-from-JAX-restart vs continuous; JAX-from-JAX-restart vs continuous"
