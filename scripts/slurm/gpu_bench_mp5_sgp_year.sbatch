#!/bin/bash
# Noah-MP 5.0, SGP 27x22 (616 tiles), 2017 full year, 15-min step, 6-hourly history: Fortran LIS (GNU 14, serial) vs Python/JAX on a V100. Configs in lisf_run/mp5_sgp_year.
#SBATCH --job-name=mp5_sgp_year
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --output=$LISFJAX_REPO/gpu_runs/mp5_sgp_year_%j.out
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclusive
RUN=$LISF_RUN/mp5_sgp_year
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $RUN
echo "=== Fortran LIS (Noah-MP 5.0, GNU 14 serial, one core), SGP, 2017 full year"
( source $LISF_DEPS/env_lis.sh; OMP_NUM_THREADS=1 /usr/bin/time -v $LISF_RUN/LIS_mp_clean -f lis.config.fort > lis.fort.out 2> lis.fort.time
  grep -E "Elapsed|Maximum resident|User time" lis.fort.time; tail -2 fort/lislog.0000 )
echo "=== JAX on the same node"
cd $LISFJAX_REPO
$CONDA_BIN run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/lis.config.jax | grep -v "wrote history" | cut -c1-300
