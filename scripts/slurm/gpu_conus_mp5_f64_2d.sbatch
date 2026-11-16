#!/bin/bash
# Float64 sensitivity check: the Noah-MP 5.0 CONUS run for 2017-02-01 00Z to 02-03 00Z with every float32 in a scratch copy of src switched to float64
# (sed jnp.float32/np.float32 -> float64, JAX_ENABLE_X64=1). Compared with the float32 run and Fortran by tools/compare/swe_divergence.py.
#SBATCH --job-name=mp5_f64
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax/gpu_runs/conus_mp5_f64_%j.out
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp5_60d
S64=/panfs/ccds02/nobackup/people/gtamkin/.nccstmp/f64
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd $S64; export OMP_NUM_THREADS=1 PYTHONPATH=$S64/src JAX_ENABLE_X64=1; unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
\rm -rf $RUN/jaxout_f64
/usr/bin/time -v $CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/lis.config.jax_f64 2> $RUN/jax_f64.time | grep -v "wrote history" > $RUN/jax_f64.out
grep -E "Elapsed|Maximum resident" $RUN/jax_f64.time; tail -3 $RUN/jax_f64.out | cut -c1-200
