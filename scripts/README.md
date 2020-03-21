
## processing scripts

The scripts are for processing files from either GITR or GITRm, except when noted.

### make_vtk_nc_compare

 To convert particle history NC file to legacy VTK, or to compare 
 two history NC files.

 Two streams of use:

* (a) compare 2 NC file particle paths 
* (b) makevtk for 1 file
    
 The stream (a) is given first in the <1st_stream/2nd_stream>

 If no split in <>, then that is only for 2nd stream.
 
```
(a) /path/<exe> <ncfile> <ncfile2> 1.0e-3 10000
 
(b) /path/<exe> <ncfile> 1 5 # for 10 ptcls starting 5th
  
Use:
  <exe> <NCfile/NCfile> [ <NCfile2/numPtcls> <tolerance/pstart#> \  
       <numTimesteps/numTimesteps> \
       <tstart#> <nP-name> <nT-name> [<x-name> <y-name> <z-name>]]  
  where the 3rd line starting tstart are only for stream2.

Example:
 ./make_vtk <ncfile> [10 0 1000 0  "nP" "nT" "x" "y" "z"]

Compile:
  module load gcc mpich netcdf;
  g++ --std=c++11 <thisfile>.cpp -o makevtk -I/<ncxx>/install/include \
   -lnetcdf -L/<nccxx>/install/lib64/ -lnetcdf-cxx4

Before run:
  export LD_LIBRARY_PATH=/lore/gopan/install/build-netcdfcxx431/install/lib64:$LD_LIBRARY_PATH
   
```

### GITR_pisces_result_count.sh 

  To count/print particles from positions.m on piseces tower.

### script_get_num.sh 

  To retrieve nth entry from ncdump/text file.

### gitr_class_all_edit.py  (From GITR)

  This is copied from GITR with edits to plot surface model results.

### make_vtk_nc_2dmesh.cpp

  Convert GITR surface mesh (2d) NC file to VTK format. The NC file 
is to be created by running GITR code.
