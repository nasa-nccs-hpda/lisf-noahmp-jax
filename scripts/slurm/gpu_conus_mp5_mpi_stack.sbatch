#!/bin/bash
# Test for the bad rank blocks at 32/40 ranks: same 5-day case and layouts, but with unlimited stack and per-run log directories.
# Hypothesis: stack/automatic-array problem in the forcing reader (read_nldas20b.F90 puts varfield and dummy on the stack).
#SBATCH --job-name=mp5_mpi_stk
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_mp5_mpi/stack_%j.out
D=/panfs/ccds02/nobackup/people/gtamkin/dev
RUN=$D/lisf_run/conus_mp5_mpi
source $D/lisf_deps/env_gnu14.sh
M=$D/lisf_deps_mpi/install
export PATH=$M/bin:$PATH
export LD_LIBRARY_PATH=$M/lib:$M/esmf/lib/libO/Linux.gfortran.64.mpich.default:$D/lisf_deps/install/lib:$D/lisf_deps/install/lib64:$LD_LIBRARY_PATH
cd $RUN
echo "stack limit before: $(ulimit -s)"; ulimit -s unlimited; echo "after: $(ulimit -s)"
for cfg in "32 8 4" "40 8 5"; do
  set -- $cfg; N=$1; X=$2; Y=$3; TAG=n${N}_stk
  \rm -rf lisout_$TAG; mkdir -p lisout_$TAG
  sed -e "s#^Ending day:.*#Ending day: 6#; s#^Ending month:.*#Ending month: 2#; s#^Number of processors along x:.*#Number of processors along x: $X#; s#^Number of processors along y:.*#Number of processors along y: $Y#; s#^Output directory:.*#Output directory: ./lisout_$TAG#; s#^Diagnostic output file:.*#Diagnostic output file: ./lisout_$TAG/lislog#" lis.config.base > lis.config.$TAG
  echo "=== $TAG start $(date +%T)"
  /usr/bin/time -v mpiexec -n $N $D/lisf_run/LIS_mp50_mpi -f lis.config.$TAG > lis.$TAG.out 2> lis.$TAG.time
  echo "exit $?"; grep -E "Elapsed" lis.$TAG.time
done
