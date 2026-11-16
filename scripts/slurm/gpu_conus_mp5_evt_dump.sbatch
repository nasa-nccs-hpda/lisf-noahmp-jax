#!/bin/bash
# Stage 2 of the flux-outlier replay: Fortran dump LIS (serial) restarted from the real-LIS 2017-03-06 00Z state (stage 1), 18 h, snow tiles, steps 1-72 (15-minute steps; the step ending 18Z is #72).
#SBATCH --job-name=mp5_evt_dump
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
D=$LISF_RUN/mp50_evt0306
RST=$(ls $LISF_RUN/conus_mp5_mpi/lisout_evt/SURFACEMODEL/*/LIS_RST_NoahMP50_201703060000.d01.nc); echo "restart: $RST"
cd $D; sed -i "s#^Noah-MP.5.0 restart file:.*#Noah-MP.5.0 restart file: $RST#" lis.config
source $LISF_DEPS/env_lis.sh
\rm -f jaxdump*; \rm -rf lisout; mkdir lisout
JAXDUMP_SNOWONLY=1 JAXDUMP_MAXSTEP=72 $LISF_RUN/LIS_mp50_dump -f lis.config > lis.out 2>&1; tail -1 lisout/lislog.0000; ls -la jaxdump_mp.bin
