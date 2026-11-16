#!/bin/bash
#SBATCH --job-name=jaxr1dcpu
#SBATCH --partition=compute
#SBATCH --account=ilab
#SBATCH --constraint=mil
#SBATCH --qos=serial
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
module load python/GEOSpyD/24.11.3-0/3.12
J=$DISCOVER_WORK/jaxtest
R=$R_CASE
rm -rf $J/out_1d_cpu; mkdir -p $J/out_1d_cpu
python3 $J/mkcfg.py $R/lis.config.nldas2.Basic.OL $J/cfg_1d_cpu 2008 10 2 $J/out_1d_cpu
export LISJAX_FORCING_PROCS=8 JAX_PLATFORMS=cpu
cd $J/lisf-noahmp-jax
/usr/bin/time -v $J/venv2/bin/python -m lisf_noahmp_jax.cli -C $R $J/cfg_1d_cpu --hours 1
ls $J/out_1d_cpu/SURFACEMODEL/*/ | head -3
