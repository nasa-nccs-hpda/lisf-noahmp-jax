#!/bin/bash
# Noah-MP 5.0, CONUS 60 simulated days (2017-02-01 to 04-02): Fortran LIS (GNU 14, serial, 1 core), same node type as the JAX job.
#SBATCH --job-name=mp5_60d_f
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax/gpu_runs/conus_mp5_60d_fortran_%j.out
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp5_60d
echo "host: $(hostname) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_lis.sh
cd $RUN; \rm -rf lisout
/usr/bin/time -v /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/LIS_mp_clean -f lis.config.fortran > lis.fortran.out 2> lis.fortran.time
grep -E "Elapsed|Maximum resident|User time" lis.fortran.time; tail -1 lisout/lislog.0000
