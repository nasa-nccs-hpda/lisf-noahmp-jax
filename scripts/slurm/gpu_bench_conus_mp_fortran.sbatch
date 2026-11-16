#!/bin/bash
# Fortran LIS (GNU 14, serial) Noah-MP 4.0.1 timing for the CONUS 1-day benchmark on the same node type as the V100 JAX run.
#SBATCH --job-name=lis_fortran_conus_mp
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=$LISFJAX_REPO/gpu_runs/conus_mp_fortran_%j.out
RUN=$LISF_RUN
echo "host: $(hostname) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
source $LISF_DEPS/env_lis.sh
rm -rf $RUN/conus_mp_fbench/lisout; mkdir -p $RUN/conus_mp_fbench/lisout && cd $RUN/conus_mp_fbench && ln -sfn ../INPUT INPUT && ln -sfn ../conus_mp/out out
cp $RUN/conus_mp/bench/lis.config.1day lis.config
/usr/bin/time -v $RUN/LIS_mp_clean -f lis.config > lis.out 2> lis.time
grep -E "Elapsed|Maximum resident|User time" lis.time
tail -1 lisout/lislog.0000
