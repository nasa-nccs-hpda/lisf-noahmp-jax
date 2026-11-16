#!/bin/bash
# Open-loop replay on deep Mar-1 snow: JAX runs Feb 1 - Mar 1 (restart written), the Fortran dump LIS starts from that restart for 4 h (snow tiles), JAX replays the dumped calls.
#SBATCH --job-name=mp5_mar1_rep
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=$LISFJAX_REPO/gpu_runs/conus_mp5_mar1_replay_%j.out
JP=$LISFJAX_REPO
D=$LISF_RUN/mp50_rst_mar1
CONDA=$CONDA_BIN
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $JP; \rm -rf $D/jx
$CONDA run --no-capture-output -n graphcast-env python -u -m lisf_noahmp_jax.driver.lis -C $D $D/cfg.jx | grep -E "TIMING|rror" | cut -c1-120
RST=$(ls $D/jx/SURFACEMODEL/*/LIS_RST_NoahMP50_201703010000.d01.nc); echo "restart: $RST"
sed -i "s#RSTFILE#$RST#" $D/lis.config
source $LISF_DEPS/env_lis.sh
cd $D; \rm -f jaxdump*; \rm -rf lisout; mkdir lisout
JAXDUMP_SNOWONLY=1 JAXDUMP_MAXSTEP=16 $LISF_RUN/LIS_mp50_dump -f lis.config > lis.out 2>&1; tail -1 lisout/lislog.0000
cd $JP; MPVER=50 JAX_PLATFORMS=cpu $CONDA run --no-capture-output -n graphcast-env python lis/replay_mp50.py $D 300000 0 rand:20000 > $D/replay.txt 2>&1; tail -40 $D/replay.txt
