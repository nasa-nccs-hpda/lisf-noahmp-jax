#!/bin/bash
# CONUS Noah-MP 4.0.1, 7 simulated days on a V100 (compare with the Fortran reference dev/lisf_run/conus_mp7/lisout).
#SBATCH --job-name=lisjax_conus_mp7
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=$LISFJAX_REPO/gpu_runs/conus_mp7_jax_%j.out
JP=$LISFJAX_REPO
RUN=$LISF_RUN
CONDA=$CONDA_BIN
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $JP
$CONDA run --no-capture-output -n graphcast-env python -u src/lisf_noahmp_jax/driver/bench_conus.py $RUN/conus_mp7 $RUN/conus_mp7/jax7 7
