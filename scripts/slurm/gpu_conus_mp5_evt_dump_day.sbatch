#!/bin/bash
# Fortran dump LIS (serial) for one March-2017 day, restarted from the real-LIS restart of that day 00Z (stage 1b), snow tiles, 72 steps of 15 min (to 18Z).
# usage: sbatch --export=ALL,DAY=24 gpu_conus_mp5_evt_dump_day.sbatch
#SBATCH --job-name=mp5_evt_dd
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
R=$LISF_RUN
D=$R/mp50_evt03$DAY
RST=$(ls $R/conus_mp5_mpi/lisout_evt2/SURFACEMODEL/201703/LIS_RST_NoahMP50_201703${DAY}0000.d01.nc); echo "restart: $RST"
mkdir -p $D; cd $D; ln -sfn ../INPUT INPUT; ln -sfn ../ldt_mp5c ldt_mp5c
sed -e "s/^Starting day:.*/Starting day: $DAY/; s/^Ending day:.*/Ending day: $DAY/; s#^Noah-MP.5.0 restart file:.*#Noah-MP.5.0 restart file: $RST#" $R/mp50_evt0306/lis.config > lis.config
source $LISF_DEPS/env_lis.sh
\rm -f jaxdump*; \rm -rf lisout; mkdir lisout
JAXDUMP_SNOWONLY=1 JAXDUMP_MAXSTEP=72 $R/LIS_mp50_dump -f lis.config > lis.out 2>&1; tail -1 lisout/lislog.0000; ls -la jaxdump_mp.bin
