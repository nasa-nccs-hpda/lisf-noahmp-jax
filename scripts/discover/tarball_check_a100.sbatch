#!/bin/bash
# Discover step 2b: the 5-day example on one A100 (submit with sbatch -C rome; the partition needs a time limit above 1 h). Run tarball_check_setup.sh first.
# The example tolerances were set on CPU; a tolerance failure on GPU is a result to report, not necessarily a bug.
#SBATCH --job-name=tb_a100
#SBATCH --partition=gpu_a100
#SBATCH --account=ilab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/discover/nobackup/gtamkin/tbcheck/tb_a100_%j.out
W=/discover/nobackup/gtamkin/tbcheck; cd $W/lisf-noahmp-jax
echo "host: $(hostname)"; nvidia-smi -L; export JAX_PLATFORMS=cuda
$W/uvenv/bin/python -c "import jax; print(jax.devices())"
/usr/bin/time -v $W/uvenv/bin/python -m pytest -q tests/test_example_sgp.py -p no:cacheprovider
