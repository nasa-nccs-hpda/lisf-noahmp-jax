#!/bin/bash
# Noah-MP 5.0 CONUS Fortran LIS segment 4 (starts from the JAX restart of the segment start; compare with the JAX continuation).
#SBATCH --job-name=mp5_long_f4
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax/gpu_runs/conus_mp5_long_f4_%j.out
D=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp5_long
echo "host: $(hostname) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_lis.sh
cd $D; \rm -rf f4
/usr/bin/time -v /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/LIS_mp_clean -f cfg.f4 > f4.out 2> f4.time
grep -E "Elapsed|Maximum resident" f4.time; tail -1 f4/lislog.0000
