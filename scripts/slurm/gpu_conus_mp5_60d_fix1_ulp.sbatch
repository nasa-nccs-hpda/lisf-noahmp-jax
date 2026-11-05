#!/bin/bash
# Same 60-day Noah-MP 5.0 CONUS JAX run with air temperature moved one float32 ulp (LISJAX_ULP_TAIR=1): sensitivity test.
#SBATCH --job-name=mp5_60d_fix1ulp
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax/gpu_runs/conus_mp5_60d_fix1ulp_%j.out
JP=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lisf-noahmp-jax
RUN=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp5_60d
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
echo "host: $(hostname) gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd $JP; export OMP_NUM_THREADS=1; export LISJAX_ULP_TAIR=1; unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
\rm -rf $RUN/jaxout_fix1_ulp
/usr/bin/time -v $CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $RUN $RUN/lis.config.jax_fix1_ulp 2> $RUN/jax_fix1_ulp.time | grep -v "wrote history" > $RUN/jax_fix1_ulp.out
grep -E "Elapsed|Maximum resident" $RUN/jax_fix1_ulp.time; grep TIMING $RUN/jax_fix1_ulp.out | cut -c1-160
