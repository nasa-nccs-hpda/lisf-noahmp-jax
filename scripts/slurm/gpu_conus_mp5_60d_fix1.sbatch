#!/bin/bash
# Noah-MP 5.0 CONUS 60 d JAX rerun after the snow liquid-water hydrology fix (SNOW_RET_FAC, SNLIQMAXFRAC, layer thickness).
#SBATCH --job-name=mp5_60d_fix1
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=$LISFJAX_REPO/gpu_runs/conus_mp5_60d_fix1_%j.out
JP=$LISFJAX_REPO
RUN=$LISF_RUN/conus_mp5_60d
CONDA=$CONDA_BIN
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd $JP; export OMP_NUM_THREADS=1; unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
\rm -rf $RUN/jaxout_fix1
/usr/bin/time -v $CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/lis.config.jax_fix1 2> $RUN/jax_fix1.time | grep -v "wrote history" > $RUN/jax_fix1.out
grep -E "Elapsed|Maximum resident" $RUN/jax_fix1.time; grep TIMING $RUN/jax_fix1.out | cut -c1-160
