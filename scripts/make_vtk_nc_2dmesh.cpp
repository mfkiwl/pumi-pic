#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <netcdf>

// module load gcc mpich netcdf;
// export LD_LIBRARY_PATH=/lore/gopan/install/build-netcdfcxx431/install/lib64:$LD_LIBRARY_PATH
// g++ --std=c++11 <thisfile>.cpp -o makevtk -I/lore/gopan/install/build-netcdfcxx431/install/include 
// -lnetcdf -L/lore/gopan/install/build-netcdfcxx431/install/lib64/ -lnetcdf-cxx4


int main(int argc, char* argv[]) {

  if(argc < 2) {
    // convert 2D mesh to legacy vtk. Input= 3 points per cell
    // /path/to/makevtk_mesh ncfile.nc   - "X" "Y" "Z"
    std::cout << argv[0] << " <NCfile> [<cells-name> <x-name> <y-name> <z-name>]\n"
              << "Example: \n"
              << "./makevtk_mesh <ncfile> [<ncells> \"x\" \"y\" \"z\"] \n";
    exit(1);
  }
  std::string ncFileName = argv[1];

  std::string ncellsName = "ncells";
  std::string xName = "x";
  std::string yName = "y";
  std::string zName = "z";
  if(argc > 2)
    ncellsName = argv[2];
  if(argc > 3) {
    xName = argv[3];
    yName = argv[4];
    zName = argv[5];
  }
  std::vector<double> xdata, ydata, zdata;
  long int ncells = 0;
  long int size = 0;
  try {
    netCDF::NcFile ncf(ncFileName, netCDF::NcFile::read);
    netCDF::NcDim ncCells(ncf.getDim(ncellsName));
    ncells = ncCells.getSize();
    size = 3*ncells;
    xdata.resize(size); 
    ydata.resize(size); 
    zdata.resize(size); 
    netCDF::NcVar ncvar1(ncf.getVar(xName));
    ncvar1.getVar(&(xdata[0]));
    netCDF::NcVar ncvar2(ncf.getVar(yName));
    ncvar2.getVar(&(ydata[0]));
    netCDF::NcVar ncvar3(ncf.getVar(zName));
    ncvar3.getVar(&(zdata[0]));
    std::cout << "Done reading " << ncFileName << "\n";
  } catch (netCDF::exceptions::NcException &e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  //write vtk file
  auto points = size; //4th to close-out
  std::vector<unsigned long int>cellLinks(4*ncells);
  std::vector<int>cellCodes(ncells, 3);
  long int cumul = 0;
  for(auto i=0; i<ncells; ++i) {
    cellLinks[i*4] = 3;
    cellLinks[i*4+1] = cumul;
    cellLinks[i*4+2] = cumul+1;
    cellLinks[i*4+3] = cumul+2;
    cumul += 3;
  }
  
  std::string outFileName = ncFileName.substr(0, ncFileName.find_last_of(".")) + ".vtk";
  outFileName = outFileName.substr(ncFileName.find_last_of("/")+1);
  std::cout << "Writing VTK file ./" << outFileName << "\n";
  std::ofstream outf;
  outf.open(outFileName);

  outf << "# vtk DataFile Version 2.0\n" 
       << "Mesh 2D\n";
  outf << "ASCII\n"; 
  outf << "DATASET UNSTRUCTURED_GRID\n"
       << "POINTS " << points << " double\n";
  std::cout << "Writing x,y,z \n";

  for(auto i = 0; i < size; ++i)
    outf << xdata[i] << " " << ydata[i] << " " << zdata[i] << "\n";

  
  std::cout << "Writing cell links \n";
  outf << "CELLS " << ncells << " " << 4*ncells << "\n";
  for(auto i=0; i< ncells; ++i)
    outf << cellLinks[4*i] << " " <<  cellLinks[4*i+1] << " " 
         << cellLinks[4*i+2] << " " <<  cellLinks[4*i+3] << "\n";
  
  std::cout << "Writing cell-code 5 \n";
  outf << "CELL_TYPES " << ncells << "\n";
  for(auto i=0; i< ncells; ++i)
    outf << "5\n";

  outf.close();
  std::cout << "Done !\n";  
  return 0;
}
