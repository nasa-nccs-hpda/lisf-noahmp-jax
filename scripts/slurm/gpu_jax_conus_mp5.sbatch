#!/bin/bash
# CONUS 1 simulated day, Noah-MP 5.0 (NLDAS-2, 15-min step, 2017-02-01): Python/JAX on a V100 only (rerun after the output fixes; Fortran reference in lisout from gpu_bench_conus_mp5.sbatch).
# Submit: ssh gpulogin1 "cd <this dir> && sbatch gpu_bench_conus_mp5.sbatch"
#SBATCH --job-name=mp5_conus_jax
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port/gpu_runs/conus_mp5_jax_%j.out
JP=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp5
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
cd $JP
echo "=== JAX on V100"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
/usr/bin/time -v $CONDA run --no-capture-output -n graphcast-env python -u -m lisjax.lis -C $RUN $RUN/lis.config.jax > $RUN/jax.out 2> $RUN/jax.time
grep -E "Elapsed|Maximum resident" $RUN/jax.time; tail -3 $RUN/jax.out
