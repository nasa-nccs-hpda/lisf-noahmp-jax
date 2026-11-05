#!/bin/bash
# CONUS benchmark (76,180 land tiles, NLDAS-2, Noah.3.9, 15-min step): Fortran LIS (1 core) vs Python/JAX on a V100.
# Submit: ssh gpulogin1 "cd <this dir> && sbatch gpu_bench_conus.sbatch"
#SBATCH --job-name=lisjax_conus_bench
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax/gpu_runs/conus_bench_%j.out
JP=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $JP
echo "=== JAX on V100, 1 simulated day (cold: includes compile)"
$CONDA run --no-capture-output -n graphcast-env python -u src/lisf_noahmp_jax/driver/bench_conus.py $RUN/conus $JP/run/conus/bench_gpu1 1
echo "=== JAX on V100, 2 simulated days"
$CONDA run --no-capture-output -n graphcast-env python -u src/lisf_noahmp_jax/driver/bench_conus.py $RUN/conus $JP/run/conus/bench_gpu2 2
echo "=== JAX on CPU of this node (JAX_PLATFORMS=cpu), 1 simulated day"
JAX_PLATFORMS=cpu $CONDA run --no-capture-output -n graphcast-env python -u src/lisf_noahmp_jax/driver/bench_conus.py $RUN/conus $JP/run/conus/bench_cpu1 1
echo "=== Fortran LIS (GNU 14, serial, 1 core) 1 simulated day on this node"
mkdir -p $RUN/conus_bench/lisout && cd $RUN/conus_bench && ln -sfn ../INPUT INPUT && ln -sfn ../conus/out out
cp $RUN/conus/bench/lis.config.1day lis.config
export LD_LIBRARY_PATH=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/install/esmf/lib/libO/Linux.gfortran.64.mpiuni.default:/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/install/lib:$LD_LIBRARY_PATH
/usr/bin/time -v $RUN/LIS -f lis.config > lis.out 2> lis.time
grep -E "Elapsed|Maximum resident|User time" lis.time
tail -1 lisout/lislog.0000
