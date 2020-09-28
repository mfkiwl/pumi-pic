module load gcc
module load mpich
module load cmake/3.15.4-kgiql7d
module load netcdf
module load cuda/10.1-k3n6svu

ncxx=/lore/gopan/install/build-netcdfcxx431/install
#libconf=/lore/gopan/install/build-libconfig-1.7.2/install
#export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${libconf}/lib64/pkgconfig
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${ncxx}/lib64/pkgconfig
pp=/lore/gopan/install/build-pumipic-cuda-rhel7-blockade-gitrm/install
export CMAKE_PREFIX_PATH=${pp}:${ncxx}:$CMAKE_PREFIX_PATH
#:${libconf}
kksrc=/lore/gopan/install/kokkos
export MPICH_CXX=${kksrc}/bin/nvcc_wrapper

