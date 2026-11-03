#!/bin/bash
# V100 benchmark of the JAX Noah 3.9 port. Submit: ssh gpulogin1 "cd <this dir> && sbatch gpu_bench.sbatch"
# Same partition/constraint pattern as the Rocke3D V100 recipe (projects/imvi/rocke3d_jax/gpu/v100_step6_graphcast.sbatch).
#SBATCH --job-name=noah39_jax_v100
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclusive
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port/gpu_runs/bench_%j.out
cd /panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
export OMP_NUM_THREADS=1
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
for n in 2000 20000 262144; do
  $CONDA run --no-capture-output -n graphcast-env python -u bench.py baseline/case_$n.bin 5 || echo "FAILED n=$n"
done
