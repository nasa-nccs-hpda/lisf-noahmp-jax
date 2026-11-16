#!/bin/bash
# CONUS 1 simulated day, Noah-MP 5.0 (NLDAS-2, 15-min step, 2017-02-01): Fortran LIS (GNU 14, serial) then Python/JAX on a V100 (closed loop, same node).
# Submit: ssh gpulogin1 "cd <this dir> && sbatch gpu_bench_conus_mp5.sbatch"
#SBATCH --job-name=mp5_conus
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=$LISFJAX_REPO/gpu_runs/conus_mp5_%j.out
JP=$LISFJAX_REPO
RUN=$LISF_RUN/conus_mp5
CONDA=$CONDA_BIN
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
cd $JP
echo "=== JAX on V100"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
/usr/bin/time -v $CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/lis.config.jax > $RUN/jax.out 2> $RUN/jax.time
grep -E "Elapsed|Maximum resident" $RUN/jax.time; tail -3 $RUN/jax.out
echo "=== Fortran LIS (GNU 14, serial)"
source $LISF_DEPS/env_lis.sh
cd $RUN
/usr/bin/time -v $LISF_RUN/LIS_mp_clean -f lis.config.fortran > lis.fortran.out 2> lis.fortran.time
grep -E "Elapsed|Maximum resident|User time" lis.fortran.time
tail -1 lisout/lislog.0000
