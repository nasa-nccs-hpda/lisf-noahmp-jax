#!/bin/bash
# Noah-MP 5.0, CONUS 60 simulated days (2017-02-01 to 04-02, NLDAS-2, 15-min step, 6-hourly history): Python/JAX on a V100 (timed, own job).
#SBATCH --job-name=mp5_60d_jax
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=gpu_runs/conus_mp5_60d_jax_%j.out
JP=$LISFJAX_REPO
RUN=$LISF_RUN/conus_mp5_60d
CONDA=$CONDA_BIN
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd $JP; export OMP_NUM_THREADS=1; unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
\rm -rf $RUN/jaxout
/usr/bin/time -v $CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/lis.config.jax 2> $RUN/jax.time | grep -v "wrote history" > $RUN/jax.out
grep -E "Elapsed|Maximum resident" $RUN/jax.time; grep TIMING $RUN/jax.out | cut -c1-160
