#!/bin/bash
# CONUS Noah-MP, 12-member ensemble with GMAO forcing perturbation, 1 simulated day: Fortran (serial) timing + reference, then Python/JAX on the V100.
#SBATCH --job-name=lis_conus_ens
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=$LISFJAX_REPO/gpu_runs/conus_ens_%j.out
JP=$LISFJAX_REPO
RUN=$LISF_RUN
CONDA=$CONDA_BIN
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $JP
echo "=== JAX on V100, 12 members, 1 simulated day"
$CONDA run --no-capture-output -n graphcast-env python -u src/lisf_noahmp_jax/driver/bench_conus.py $RUN/conus_mpe $RUN/conus_mpe/jax1 1
echo "=== Fortran LIS, 12 members, 1 simulated day (serial)"
source $LISF_DEPS/env_lis.sh
cd $RUN/conus_mpe && mkdir -p lisout
/usr/bin/time -v $RUN/LIS_mp_clean -f lis.config > lis.out 2> lis.time
grep -E "Elapsed|Maximum resident|User time" lis.time
