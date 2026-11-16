# Environment to configure/compile/run LIS with GNU 14 (serial, ESMF mpiuni). Source after env_gnu14.sh.
source $LISF_DEPS/env_gnu14.sh
export LIS_ARCH=linux_gfortran
export LIS_FC=gfortran
export LIS_CC=gcc
export LIS_NETCDF=$DEPS/install
export LIS_HDF5=$DEPS/install
export LIS_ECCODES=$DEPS/install
export ESMFMKFILE=$(find $DEPS/install/esmf -name esmf.mk | head -1)
export LD_LIBRARY_PATH=$DEPS/install/esmf/lib/libO/Linux.gfortran.64.mpiuni.default:$LD_LIBRARY_PATH
