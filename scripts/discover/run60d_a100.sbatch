#!/bin/bash
#SBATCH --job-name=jaxr60a100
#SBATCH --partition=gpu_a100
#SBATCH --account=ilab
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=$DISCOVER_WORK/jaxtest/r60a100_%j.out
module load python/GEOSpyD/24.11.3-0/3.12
J=$DISCOVER_WORK/jaxtest
R=$R_CASE
rm -rf $J/out_60d_a100; mkdir -p $J/out_60d_a100
python3 $J/mkcfg.py $R/lis.config.nldas2.Basic.OL $J/cfg_60d_a100 2008 11 30 $J/out_60d_a100
nvidia-smi -L
export LISJAX_FORCING_PROCS=8 JAX_PLATFORMS=cuda
cd $J/lisf-noahmp-jax
/usr/bin/time -v $J/venv2/bin/python -m lisf_noahmp_jax.cli -C $R $J/cfg_60d_a100 --hours 3
ls $J/out_60d_a100/SURFACEMODEL/*/ | head -3
