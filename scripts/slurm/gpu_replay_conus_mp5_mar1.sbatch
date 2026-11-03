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
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port/gpu_runs/conus_mp5_mar1_replay_%j.out
JP=/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/jax-port
D=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/mp50_rst_mar1
CONDA=/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
cd $JP; \rm -rf $D/jx
$CONDA run --no-capture-output -n graphcast-env python -u -m lisjax.lis -C $D $D/cfg.jx | grep -E "TIMING|rror" | cut -c1-120
RST=$(ls $D/jx/SURFACEMODEL/*/LIS_RST_NoahMP50_201703010000.d01.nc); echo "restart: $RST"
sed -i "s#RSTFILE#$RST#" $D/lis.config
source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_lis.sh
cd $D; \rm -f jaxdump*; \rm -rf lisout; mkdir lisout
JAXDUMP_SNOWONLY=1 JAXDUMP_MAXSTEP=16 /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/LIS_mp50_dump -f lis.config > lis.out 2>&1; tail -1 lisout/lislog.0000
cd $JP; MPVER=50 JAX_PLATFORMS=cpu $CONDA run --no-capture-output -n graphcast-env python lis/replay_mp50.py $D 300000 0 rand:20000 > $D/replay.txt 2>&1; tail -40 $D/replay.txt
