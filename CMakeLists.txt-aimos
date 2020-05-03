cmake_minimum_required(VERSION 3.13.0)
project(gitrm VERSION 1.0.0 LANGUAGES CXX)

include(cmake/bob.cmake)
bob_begin_package()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(NetCDF REQUIRED IMPORTED_TARGET netcdf>4.7.0)
pkg_check_modules(NetCDF-CXX4 REQUIRED IMPORTED_TARGET netcdf-cxx4>4.3.0)

find_package(pumipic REQUIRED)
set(gitrm_USE_pumipic_DEFAULT ON)
message(STATUS "Found PUMIpic: ${pumipic_VERSION}")
bob_public_dep(pumipic)

add_subdirectory(src)
add_executable(GITRm src/GITRm.cpp ${SOURCES})
target_link_libraries(GITRm 
  pumipic::pumipic
  PkgConfig::NetCDF
  PkgConfig::NetCDF-CXX4)

#Settings options for testing
enable_testing()
include(CTest)

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

#get the mpirun binary/script
get_filename_component(COMPILER_DIR "${CMAKE_CXX_COMPILER}" PATH)
find_program(MPIRUN NAMES mpirun PATHS "${COMPILER_DIR}")
set(MPIRUN_PROCFLAG "-np" CACHE STRING
  "the command line flag to give process count to MPIRUN")

if(IS_TESTING)
  add_subdirectory(test)
endif()

bob_end_package()
