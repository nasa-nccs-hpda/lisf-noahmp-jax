#!/bin/bash
# CONUS CLSM F2.5 (76,088 catchment tiles), 1 simulated day: Fortran LIS (GNU 14, serial) vs Python/JAX on a V100 node.
#SBATCH --job-name=lisjax_clsm
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax/gpu_runs/clsm_bench_%j.out
JP=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
cd $RUN/conus_clsm1d
echo "=== Fortran LIS, 1 day (dump hook disabled with JAXDUMP_CLSM_MAXSTEP=0)"
( source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_lis.sh
  mkdir -p lisout; export JAXDUMP_CLSM_MAXSTEP=0
  /usr/bin/time -v $RUN/LIS_clsm_dump2 -f lis.config > lis.out 2> lis.time
  grep -E "Elapsed|Maximum resident|User time" lis.time; tail -1 lisout/lislog.0000 )
echo "=== JAX on V100, 1 day"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $JP
$CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.run_clsm $RUN/conus_clsm1d $RUN/conus_clsm1d/jaxout lis.config 2017020100 2017020200
echo "=== JAX second run (warm compile cache not shared; timing repeat)"
$CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.run_clsm $RUN/conus_clsm1d $RUN/conus_clsm1d/jaxout2 lis.config 2017020100 2017020200
