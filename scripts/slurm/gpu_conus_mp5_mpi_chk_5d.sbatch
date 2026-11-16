#!/bin/bash
# Production -O2 MPI LIS with per-rank forcing checksums (FCHK lines) in read_nldas20a, 5 days at 32 and 40 ranks.
# behind the bad rank blocks at 32/40 ranks. 5 simulated days, 8x4 = 32 ranks, then 8x5 = 40 ranks. Output goes to the .out file with rank prefixes.
#SBATCH --job-name=mp5_mpi_chk
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=$LISF_RUN/conus_mp5_mpi/chk_%j.out
D=$LISF_DEV
RUN=$D/lisf_run/conus_mp5_mpi
source $D/lisf_deps/env_gnu14.sh
M=$D/lisf_deps_mpi/install
export PATH=$M/bin:$PATH
export LD_LIBRARY_PATH=$M/lib:$M/esmf/lib/libO/Linux.gfortran.64.mpich.default:$D/lisf_deps/install/lib:$D/lisf_deps/install/lib64:$LD_LIBRARY_PATH
cd $RUN
for cfg in "32 8 4" "40 8 5"; do
  set -- $cfg; N=$1; X=$2; Y=$3; TAG=chk$N
  \rm -rf lisout_$TAG; mkdir -p lisout_$TAG
  sed -e "s#^Ending day:.*#Ending day: 6#; s#^Ending month:.*#Ending month: 2#; s#^Number of processors along x:.*#Number of processors along x: $X#; s#^Number of processors along y:.*#Number of processors along y: $Y#; s#^Output directory:.*#Output directory: ./lisout_$TAG#; s#^Diagnostic output file:.*#Diagnostic output file: ./lisout_$TAG/lislog#" lis.config.base > lis.config.$TAG
  echo "=== $TAG start $(date +%T)"
  /usr/bin/time -v mpiexec -prepend-rank -n $N $D/lisf_run/LIS_mp50_mpi_chk -f lis.config.$TAG > lis.$TAG.out 2> lis.$TAG.time
  echo "exit $?"; grep -E "Elapsed" lis.$TAG.time
  grep -hE "Program received|Fortran runtime error|Backtrace|#[0-9]+ .* at |SIGFPE|bound" lis.$TAG.out lis.$TAG.time | head -40
done
