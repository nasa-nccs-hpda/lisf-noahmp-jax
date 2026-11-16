#!/bin/bash
# Noah-MP 5.0 CONUS Fortran LIS segment 3 (starts from the JAX restart of the segment start; compare with the JAX continuation).
#SBATCH --job-name=mp5_long_f3
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=gpu_runs/conus_mp5_long_f3_%j.out
D=$LISF_RUN/conus_mp5_long
echo "host: $(hostname) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
source $LISF_DEPS/env_lis.sh
cd $D; \rm -rf f3
/usr/bin/time -v $LISF_RUN/LIS_mp_clean -f cfg.f3 > f3.out 2> f3.time
grep -E "Elapsed|Maximum resident" f3.time; tail -1 f3/lislog.0000
