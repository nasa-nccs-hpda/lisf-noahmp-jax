#!/bin/bash
# Noah-MP 5.0 CONUS 60 d JAX rerun with instantaneous Qle/Qh/Qg/Evap/Swnet/Lwnet written (closed-loop flux comparison, roadmap item 10).
#SBATCH --job-name=mp5_60d_flux
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=gpu_runs/conus_mp5_60d_flux_%j.out
JP=$LISFJAX_REPO
RUN=$LISF_RUN/conus_mp5_60d
CONDA=$CONDA_BIN
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd $JP; export OMP_NUM_THREADS=1 PYTHONPATH=$JP/src; unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
\rm -rf $RUN/jaxout_flux
/usr/bin/time -v $CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/lis.config.jax_flux 2> $RUN/jax_flux.time | grep -v "wrote history" > $RUN/jax_flux.out
grep -E "Elapsed|Maximum resident" $RUN/jax_flux.time; grep TIMING $RUN/jax_flux.out | cut -c1-160
