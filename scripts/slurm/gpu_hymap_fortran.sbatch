#!/bin/bash
# Fortran LIS (GNU 14, serial) Noah-MP 4.0.1 + HyMAP2 router, CONUS 0.125 deg, 2 days: reference output + timing.
#SBATCH --job-name=lis_fortran_hymap
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=$LISFJAX_REPO/gpu_runs/hymap_fortran_%j.out
RUN=$LISF_RUN
echo "host: $(hostname) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
source $LISF_DEPS/env_lis.sh
cd $RUN/conus_hy && rm -rf lisout_f && mkdir -p lisout_f
sed 's#\./lisout#./lisout_f#g' lis.config > lis.config.f
/usr/bin/time -v $RUN/LIS_mp_hy -f lis.config.f > lis.f.out 2> lis.f.time
grep -E "Elapsed|Maximum resident|User time" lis.f.time
tail -1 lisout_f/lislog.0000
