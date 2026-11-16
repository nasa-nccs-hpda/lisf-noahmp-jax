#!/bin/bash
#SBATCH --job-name=jaxr1d
#SBATCH --partition=gpu_a100
#SBATCH --account=ilab
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
module load python/GEOSpyD/24.11.3-0/3.12
J=$DISCOVER_WORK/jaxtest
R=$R_CASE
rm -rf $J/out_1d; mkdir -p $J/out_1d
python3 $J/mkcfg.py $R/lis.config.nldas2.Basic.OL $J/cfg_1d 2008 10 2 $J/out_1d
nvidia-smi -L
export LISJAX_FORCING_PROCS=8 JAX_PLATFORMS=cuda
cd $J/lisf-noahmp-jax
/usr/bin/time -v $J/venv2/bin/python -m lisf_noahmp_jax.cli -C $R $J/cfg_1d --hours 1
ls $J/out_1d/SURFACEMODEL/*/ | head -3
