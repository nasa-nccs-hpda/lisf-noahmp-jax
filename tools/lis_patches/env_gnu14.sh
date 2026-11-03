# GNU 14 toolchain for building LISF (written 2026-10-21). Source this file.
source /usr/share/lmod/lmod/init/bash
module purge 2>/dev/null
module load gcc/14.2.0 cmake/3.24.2
# serial build: no MPI (the cluster's OpenMPI 5 module needs libhwloc.so.15 which is not on the system path)
export DEPS=/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_deps
export PATH=$DEPS/install/bin:$PATH
export LD_LIBRARY_PATH=$DEPS/install/lib:$DEPS/install/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH
export CC=gcc FC=gfortran CXX=g++
