module load gcc
module load mpich
module load cmake
module load netcdf
module load cuda/10.1-k3n6svu
ncxx=/lore/gopan/install/build-netcdfcxx431/install
export MY_NCXX=$ncxx
libconf=/lore/gopan/install/build-libconfig-1.7.2/install
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${libconf}/lib64/pkgconfig
#:${ncxx}/lib64/pkgconfig
#export CPATH=${CPATH}:${libconf}/include
#export PATH=${PATH}:${libconf}/bin
pp=/lore/gopan/install/build-pumipic-cuda-rhel7-blockade/install
export LD_LIBRARY_PATH=${ncxx}/lib64:${libconf}/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${pp}:$ncxx:$ncxx/lib64/cmake/netCDFCxx:${libconf}:$CMAKE_PREFIX_PATH
kksrc=/lore/gopan/install/kokkos
export MPICH_CXX=${kksrc}/bin/nvcc_wrapper


