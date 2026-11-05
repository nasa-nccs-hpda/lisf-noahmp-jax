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
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax/gpu_runs/conus_mp5_rt_%j.out
JP=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp5_rt
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
LIS=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/LIS_mp_clean
jax() { (cd $JP; $CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/$1 | grep -E "TIMING|rror" | cut -c1-200); }
echo "=== JAX day 1 with restart output"; jax cfg.j_a; ls $RUN/j_a/SURFACEMODEL/201702/
echo "=== JAX from its own restart, 6 h"; jax cfg.j_c
echo "=== Fortran from the JAX restart, 6 h"
( source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_lis.sh; cd $RUN; $LIS -f cfg.f_b > f_b.out 2>&1; tail -2 f_b/lislog.0000 )
echo "=== Fortran continuous to Feb 2 06:00"
( source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_lis.sh; cd $RUN; $LIS -f cfg.f_full > f_full.out 2>&1; tail -2 f_full/lislog.0000 )
echo "=== compare (hours 1-6 of Feb 2): Fortran-from-JAX-restart vs continuous; JAX-from-JAX-restart vs continuous"
