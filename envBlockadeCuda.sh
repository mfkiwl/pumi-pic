module load gcc
module load mpich
module load cmake
module load netcdf
module load cuda/10.1-k3n6svu
ncxx=/lore/gopan/install/build-netcdfcxx431/install
export MY_NCXX=$ncxx
pp=/lore/gopan/install/build-pumipic-cuda-rhel7-blockade/install
export LD_LIBRARY_PATH=$ncxx/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${pp}:$ncxx:$ncxx/lib64/cmake/netCDFCxx:$CMAKE_PREFIX_PATH
kksrc=/lore/gopan/install/kokkos
export MPICH_CXX=${kksrc}/bin/nvcc_wrapper


