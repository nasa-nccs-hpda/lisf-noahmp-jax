#!/bin/bash
# CONUS benchmark, Noah-MP 4.0.1 (76,180 land tiles, NLDAS-2, 15-min step): Python/JAX on a V100 and on the node CPU (Fortran timing: gpu_bench_conus_mp_fortran.sbatch).
# Submit: ssh gpulogin1 "cd <this dir> && sbatch gpu_bench_conus_mp.sbatch"
#SBATCH --job-name=lisjax_conus_mp
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=$LISFJAX_REPO/gpu_runs/conus_mp_jax_%j.out
JP=$LISFJAX_REPO
RUN=$LISF_RUN
CONDA=$CONDA_BIN
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $JP
echo "=== JAX on V100, 1 simulated day (cold: includes compile)"
$CONDA run --no-capture-output -n graphcast-env python -u src/lisf_noahmp_jax/driver/bench_conus.py $RUN/conus_mp $RUN/conus_mp/bench_gpu1 1
echo "=== JAX on V100, 2 simulated days"
$CONDA run --no-capture-output -n graphcast-env python -u src/lisf_noahmp_jax/driver/bench_conus.py $RUN/conus_mp $RUN/conus_mp/bench_gpu2 2
echo "=== JAX on CPU of this node (JAX_PLATFORMS=cpu), 1 simulated day"
JAX_PLATFORMS=cpu $CONDA run --no-capture-output -n graphcast-env python -u src/lisf_noahmp_jax/driver/bench_conus.py $RUN/conus_mp $RUN/conus_mp/bench_cpu1 1
