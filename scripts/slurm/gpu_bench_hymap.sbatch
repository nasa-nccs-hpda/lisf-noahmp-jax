#!/bin/bash
# CONUS Noah-MP 4.0.1 + HyMAP2 routing (76,180 LSM tiles, 78,554 river cells): Python/JAX on a V100, 1 and 2 simulated days.
#SBATCH --job-name=lisjax_hymap
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port/gpu_runs/hymap_jax_%j.out
JP=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $JP
echo "=== JAX on V100, 1 simulated day (cold)"
$CONDA run --no-capture-output -n graphcast-env python -u lisjax/bench_conus.py $RUN/conus_hy $RUN/conus_hy/bench_gpu1 1
echo "=== JAX on V100, 2 simulated days (these are the files compared with the Fortran run)"
$CONDA run --no-capture-output -n graphcast-env python -u lisjax/bench_conus.py $RUN/conus_hy $RUN/conus_hy/bench_gpu2 2
