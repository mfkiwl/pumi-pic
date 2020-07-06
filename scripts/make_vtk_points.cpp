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
// NC=/lore/gopan/install/build-netcdfcxx431/install
// g++ --std=c++11 <thisfile>.cpp -o makevtk -I${NC}/include -lnetcdf -L${NC}/lib64/ -lnetcdf-cxx4
// export LD_LIBRARY_PATH=$NC/lib64:$LD_LIBRARY_PATH

int main(int argc, char* argv[]) {

  if(argc < 2) {
    // convert points NC file to legacy vtk. Input=x,y,z separate arrays
    std::cout << argv[0] << " <NCfile> [<outname> <nP-name> <x-name> <y-name> <z-name>]\n"
              << "Example: \n"
              << "./makevtk_mesh <ncfile> positions.vtk";
    exit(1);
  }
  std::string ncFileName = argv[1];

  std::string nPname = "nP";
  std::string xName = "x";
  std::string yName = "y";
  std::string zName = "z";
  std::string outName{};
  if(argc > 2)
    outName = argv[2];
  if(argc > 3)
    nPname = argv[3];
  if(argc > 4) {
    xName = argv[4];
    yName = argv[5];
    zName = argv[6];
  }
  std::vector<double> xdata, ydata, zdata;
  long int nP = 0;
  long int size = 0;
  try {
    netCDF::NcFile ncf(ncFileName, netCDF::NcFile::read);
    netCDF::NcDim ncNP(ncf.getDim(nPname));
    nP = ncNP.getSize();
    size = nP;
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
  int nans = 0;
  for(int i=0; i<xdata.size(); ++i)
    if(std::isnan(xdata[i]) || std::isnan(ydata[i]) || std::isnan(zdata[i])) {
      ++nans;
      xdata[i] = ydata[i] = zdata[i] = 0;
    }
  if(nans)
    std::cout << "\n*******WARNING found "<< nans << " NaNs in data ******\n\n";

  bool replicateTo3d = false;
  double spread = 0.5;
  int factor = 100;
  if(replicateTo3d) {
    int old = nP;
    size = nP = nP * factor;
    for(int i=0; i<old; ++i) {
      
    }
  }

  std::string ext = ".vtk";
  std::string outFileName;
  if(!outName.empty()) {
    outFileName = outName;
    size_t pos = outFileName.find_last_of(".");
    if(pos == std::string::npos)
      outFileName += ext;
    else {
      std::string ex = outFileName.substr(pos, outFileName.size() - pos);
      if(ex != ext)
        outFileName += ext;
    }
  }
  else {
    outFileName = ncFileName.substr(0, ncFileName.find_last_of(".")) + ext;
    outFileName = outFileName.substr(ncFileName.find_last_of("/")+1);
  }
  std::cout << "Writing VTK file " << outFileName << "\n";
  std::ofstream outf;
  outf.open(outFileName);

  //write vtk file
  auto points = nP;
  outf << "# vtk DataFile Version 2.0\n" 
       << "Particle points\n";
  outf << "ASCII\n"; 
  outf << "DATASET UNSTRUCTURED_GRID\n"
       << "POINTS " << points << " double\n";
  std::cout << "Writing x,y,z \n";

  for(auto i = 0; i < size; ++i)
    outf << xdata[i] << " " << ydata[i] << " " << zdata[i] << "\n";

  std::cout << "Writing cell links \n";
  outf << "CELLS " << nP << " " << 3*nP << "\n";
  for(auto i=0; i< nP; ++i)
    outf << "2 " << i << " " << i << "\n";
  
  std::cout << "Writing cell-code 3 \n";
  outf << "CELL_TYPES " << nP << "\n";
  for(auto i=0; i< nP; ++i)
    outf << "3\n";

  outf.close();
  std::cout << "Done !\n";  
  return 0;
}
