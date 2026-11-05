#!/bin/bash
# Noah-MP 5.0, SGP 27x22 (616 tiles), 2017 full year, 15-min step, 6-hourly history: Fortran LIS (GNU 14, serial) vs Python/JAX on a V100. Configs in lisf_run/mp5_sgp_year.
#SBATCH --job-name=mp5_sgp_year
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax/gpu_runs/mp5_sgp_year_%j.out
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --exclusive
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/mp5_sgp_year
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $RUN
echo "=== Fortran LIS (Noah-MP 5.0, GNU 14 serial, one core), SGP, 2017 full year"
( source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_lis.sh; OMP_NUM_THREADS=1 /usr/bin/time -v /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/LIS_mp_clean -f lis.config.fort > lis.fort.out 2> lis.fort.time
  grep -E "Elapsed|Maximum resident|User time" lis.fort.time; tail -2 fort/lislog.0000 )
echo "=== JAX on the same node"
cd /panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax
/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/lis.config.jax | grep -v "wrote history" | cut -c1-300
