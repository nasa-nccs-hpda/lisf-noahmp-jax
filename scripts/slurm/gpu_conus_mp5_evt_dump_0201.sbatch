#!/bin/bash
# Cold-start Fortran dump LIS, 2017-02-01 00Z-18Z, snow tiles, steps 1-72: replay of where the JAX free-run snow first diverges (original rain-snow option 1).
#SBATCH --job-name=mp5_evt_dump0201
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=$LISF_RUN/mp50_evt0201/dump_%j.out
D=$LISF_RUN/mp50_evt0201
cd $D
source $LISF_DEPS/env_lis.sh
\rm -f jaxdump*; \rm -rf lisout; mkdir lisout
JAXDUMP_SNOWONLY=1 JAXDUMP_MAXSTEP=72 $LISF_RUN/LIS_mp50_dump -f lis.config > lis.out 2>&1; tail -1 lisout/lislog.0000; ls -la jaxdump_mp.bin
