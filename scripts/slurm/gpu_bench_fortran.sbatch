#!/bin/bash
# Fortran LIS (GNU 14, serial) timing for the CONUS 1-day benchmark on the same node type as the V100 JAX run.
#SBATCH --job-name=lis_fortran_conus
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port/gpu_runs/conus_fortran_%j.out
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run
echo "host: $(hostname) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_lis.sh
mkdir -p $RUN/conus_bench2/lisout && cd $RUN/conus_bench2 && ln -sfn ../INPUT INPUT && ln -sfn ../conus/out out
cp $RUN/conus/bench/lis.config.1day lis.config && sed -i 's#./lisout#./lisout#' lis.config
/usr/bin/time -v $RUN/LIS -f lis.config > lis.out 2> lis.time
grep -E "Elapsed|Maximum resident|User time" lis.time
tail -1 lisout/lislog.0000
