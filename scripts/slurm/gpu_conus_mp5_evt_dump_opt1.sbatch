#!/bin/bash
# Stage 2 of the flux-outlier replay: Fortran dump LIS (serial) restarted from the real-LIS 2017-03-06 00Z state (stage 1), 18 h, snow tiles, steps 1-72 (15-minute steps; the step ending 18Z is #72).
#SBATCH --job-name=mp5_evt_dumpb
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/mp50_evt0306b/dump_%j.out
D=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/mp50_evt0306b
RST=$(ls /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp5_mpi/lisout_evt/SURFACEMODEL/*/LIS_RST_NoahMP50_201703060000.d01.nc); echo "restart: $RST"
cd $D; sed -i "s#^Noah-MP.5.0 restart file:.*#Noah-MP.5.0 restart file: $RST#" lis.config
source /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps/env_lis.sh
\rm -f jaxdump*; \rm -rf lisout; mkdir lisout
JAXDUMP_SNOWONLY=1 JAXDUMP_MAXSTEP=72 /panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/LIS_mp50_dump -f lis.config > lis.out 2>&1; tail -1 lisout/lislog.0000; ls -la jaxdump_mp.bin
