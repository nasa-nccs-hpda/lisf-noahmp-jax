#!/bin/bash
# Stage 1 of the flux-outlier replay: real LIS (MPI, 20 ranks, bitwise clean) Noah-MP 5.0 CONUS from 2017-02-01 to 03-06 06Z with daily restarts (need 03-06 00Z).
#SBATCH --job-name=mp5_evt_rst
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
D=$LISF_DEV
RUN=$D/lisf_run/conus_mp5_mpi
source $D/lisf_deps/env_gnu14.sh
M=$D/lisf_deps_mpi/install
export PATH=$M/bin:$PATH
export LD_LIBRARY_PATH=$M/lib:$M/esmf/lib/libO/Linux.gfortran.64.mpich.default:$D/lisf_deps/install/lib:$D/lisf_deps/install/lib64:$LD_LIBRARY_PATH
cd $RUN; \rm -rf lisout_evt; mkdir lisout_evt
echo "start $(date +%T)"
/usr/bin/time -v mpiexec -n 20 $D/lisf_run/LIS_mp50_mpi -f lis.config.evt_rst > lis.evt_rst.out 2> lis.evt_rst.time
echo "exit $?"; grep -E "Elapsed" lis.evt_rst.time; find lisout_evt -name "LIS_RST*" | sort | tail -3
