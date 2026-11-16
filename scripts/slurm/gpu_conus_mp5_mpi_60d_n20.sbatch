#!/bin/bash
# Noah-MP 5.0, CONUS 60 simulated days (2017-02-01 to 04-02): real LIS with MPI, 20 ranks (4 x 5), the largest layout found bitwise clean vs serial Fortran, one exclusive node of the same type as the JAX run.
#SBATCH --job-name=mp5_mpi60_n20
#SBATCH --partition=compute
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=$LISF_RUN/conus_mp5_mpi/60d_n20_%j.out
D=$LISF_DEV
RUN=$D/lisf_run/conus_mp5_mpi
echo "host: $(hostname) cpu: $(lscpu | grep 'Model name' | sed 's/.*: *//') cores: $(nproc)"
source $D/lisf_deps/env_gnu14.sh
M=$D/lisf_deps_mpi/install
export PATH=$M/bin:$PATH
export LD_LIBRARY_PATH=$M/lib:$M/esmf/lib/libO/Linux.gfortran.64.mpich.default:$D/lisf_deps/install/lib:$D/lisf_deps/install/lib64:$LD_LIBRARY_PATH
cd $RUN; \rm -rf lisout_60d_n20; mkdir lisout_60d_n20
sed -e "s#^Number of processors along x:.*#Number of processors along x: 4#; s#^Number of processors along y:.*#Number of processors along y: 5#; s#^Output directory:.*#Output directory: ./lisout_60d_n20#" lis.config.base > lis.config.60d_n20
echo "start $(date +%T)"
/usr/bin/time -v mpiexec -n 20 $D/lisf_run/LIS_mp50_mpi -f lis.config.60d_n20 > lis.60d_n20.out 2> lis.60d_n20.time
echo "exit $?"; grep -E "Elapsed|Maximum resident" lis.60d_n20.time
