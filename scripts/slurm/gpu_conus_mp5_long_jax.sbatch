#!/bin/bash
# Noah-MP 5.0 CONUS Feb 1 - Jul 1 2017 (150 d) in four JAX segments chained by restart files (Feb1-Apr1, Apr1-May1, May1-Jun1, Jun1-Jul1); timed.
#SBATCH --job-name=mp5_long_jax
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=gpu_runs/conus_mp5_long_jax_%j.out
JP=$LISFJAX_REPO
D=$LISF_RUN/conus_mp5_long
CONDA=$CONDA_BIN
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd $JP; export OMP_NUM_THREADS=1; unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
T0=$(date +%s)
for i in 1 2 3 4; do
  \rm -rf $D/j$i
  $CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $D $D/cfg.j$i 2> $D/j$i.err | grep -E "TIMING|rror" | cut -c1-160
  echo "segment $i done at +$(( $(date +%s) - T0 )) s"
done
echo "TOTAL JAX wall $(( $(date +%s) - T0 )) s"
