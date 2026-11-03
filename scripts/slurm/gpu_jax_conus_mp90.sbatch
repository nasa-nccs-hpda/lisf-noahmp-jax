#!/bin/bash
# CONUS 89 simulated days (2017-02-01 to 05-01), Noah-MP 4.0.1, JAX on a V100 only, timed (Fortran reference: lisf_run/conus_mp90/lisout_f2, 59 d in 10:34:55).
# Submit: ssh gpulogin1 "sbatch <this file>"
#SBATCH --job-name=mp90_jax_t
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port/gpu_runs/conus_mp90/jax90_timed_%j.out
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp90
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd /panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port
export OMP_NUM_THREADS=1
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
\rm -rf $RUN/jaxout_t
/usr/bin/time -v $CONDA run --no-capture-output -n graphcast-env python -u -m lisjax.lis -C $RUN $RUN/lis.config.jax_t 2> $RUN/jax_t.time | grep -v "wrote history" > $RUN/jax_t.out
grep -E "Elapsed|Maximum resident" $RUN/jax_t.time; grep TIMING $RUN/jax_t.out | cut -c1-120
