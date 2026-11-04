#!/bin/bash
# Noah-MP 5.0 CONUS 5 simulated days with the MPI LIS, DIFFERENT domain decompositions.
# Purpose: the 8x5 (40-rank) run disagreed with serial Fortran in one rank block (x-block 0, y-block 2). If the bad block moves with the layout, it is a parallel-path problem.
#SBATCH --job-name=mp5_mpi_dec
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp5_mpi/decomp_%j.out
D=/panfs/ccds02/nobackup/people/gtamkin/dev
RUN=$D/lisf_run/conus_mp5_mpi
source $D/lisf_deps/env_gnu14.sh
M=$D/lisf_deps_mpi/install
export PATH=$M/bin:$PATH
export LD_LIBRARY_PATH=$M/lib:$M/esmf/lib/libO/Linux.gfortran.64.mpich.default:$D/lisf_deps/install/lib:$D/lisf_deps/install/lib64:$LD_LIBRARY_PATH
cd $RUN
for cfg in "40 5 8" "40 10 4" "20 4 5"; do
  set -- $cfg; N=$1; X=$2; Y=$3; TAG=n${N}_${X}x${Y}
  \rm -rf lisout_$TAG; mkdir lisout_$TAG
  sed -e "s#^Ending day:.*#Ending day: 6#; s#^Ending month:.*#Ending month: 2#; s#^Number of processors along x:.*#Number of processors along x: $X#; s#^Number of processors along y:.*#Number of processors along y: $Y#; s#^Output directory:.*#Output directory: ./lisout_$TAG#" lis.config.base > lis.config.$TAG
  echo "=== $TAG start $(date +%T)"
  /usr/bin/time -v mpiexec -n $N $D/lisf_run/LIS_mp50_mpi -f lis.config.$TAG > lis.$TAG.out 2> lis.$TAG.time
  echo "exit $?"; grep -E "Elapsed" lis.$TAG.time
done
