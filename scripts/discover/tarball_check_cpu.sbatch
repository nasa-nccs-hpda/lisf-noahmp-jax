#!/bin/bash
# Discover step 2a: full test suite of the extracted tarball on a CPU node (about 30 min on 16 cores). Run tarball_check_setup.sh first.
#SBATCH --job-name=tb_cpu
#SBATCH --partition=compute
#SBATCH --account=ilab
#SBATCH --constraint=mil
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/discover/nobackup/gtamkin/tbcheck/tb_cpu_%j.out
W=/discover/nobackup/gtamkin/tbcheck; cd $W/lisf-noahmp-jax
echo "host: $(hostname)"; export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
/usr/bin/time -v $W/uvenv/bin/python -m pytest -q --durations=8 -p no:cacheprovider
