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

//nP name = dimNames[0]; nT name = dimNames[1]
int readNetcdfData(const std::string& ncFileName,
   std::vector<std::string> dimNames, std::vector<std::string> varNames,
   std::vector<double>& xd, std::vector<double>& yd, std::vector<double>& zd,
   int& numPtcls, int& nTimeSteps) {

  int ncSizePerComp = 1;
  int nComp = varNames.size();
  // 1st is nP; 2nd is nT
  std::vector<int> dims;
  try {
    netCDF::NcFile ncf(ncFileName, netCDF::NcFile::read);
    for(int i=0; i< dimNames.size(); ++i) {
      netCDF::NcDim ncDimName(ncf.getDim(dimNames[i]));
      auto size = ncDimName.getSize();
      dims.push_back(size);
      if(i==0) { //numPtcls
      //  if(size < numPtcls)
         numPtcls = size;
      }
      if(i==1) { // nTimeSteps
        nTimeSteps = size;
      }
      std::cout << dimNames[i] << " : " << dims[i] << "\n";
      ncSizePerComp *= dims[i];
    }
    std::cout << "ncSizePerComp: " << ncSizePerComp << "; nComp " << nComp << "\n";

    xd.resize(ncSizePerComp); 
    yd.resize(ncSizePerComp); 
    zd.resize(ncSizePerComp); 
    // TODO use maxNPtcls and numPtclsRead
    netCDF::NcVar ncvar1(ncf.getVar(varNames[0]));
    ncvar1.getVar(&(xd[0]));
    netCDF::NcVar ncvar2(ncf.getVar(varNames[1]));
    ncvar2.getVar(&(yd[0]));
    netCDF::NcVar ncvar3(ncf.getVar(varNames[2]));
    ncvar3.getVar(&(zd[0]));
    std::cout << "Done reading " << ncFileName << "\n";
  } catch (netCDF::exceptions::NcException &e) {
    std::cout << e.what() << std::endl;
    return 1;
  }
      //const std::vector<size_t> start{(size_t)i};
      //const std::vector<size_t> count{(size_t)numPtclxHistn};
      //const std::vector<ptrdiff_t> stridep{dof};
      // http://unidata.github.io/netcdf-cxx4/ncVar_8cpp_source.html#l00788 line-1142
      // https://github.com/Unidata/netcdf-c/blob/master/ncdump/ref_ctest.c
      //ncVars[i].putVar(start, count, stridep, &(ptclsData[0]));
  // NaNs
  int nans = 0;
  for(int i=0; i<xd.size(); ++i)
    if(std::isnan(xd[i]) || std::isnan(yd[i]) || std::isnan(zd[i])) {
      ++nans;
    }
  if(nans)
    std::cout << "\n*******WARNING found "<< nans << " NaNs in data ******\n\n";
  assert(nans == 0);

  return 0;
}

// https://stackoverflow.com/questions/10913666/error-writing-binary-vtk-files
template <typename T>
void swapEnd(T& var) {
  char* varArray = reinterpret_cast<char*>(&var);
  for(long i = 0; i < static_cast<long>(sizeof(var)/2); i++)
    std::swap(varArray[sizeof(var) - 1 - i],varArray[i]);
}
// https://stackoverflow.com/questions/585257/\
// is-there-a-better-way-to-reverse-an-array-of-bytes-in-memory
void reverseBytes( void *start, int size) {
  char *istart = reinterpret_cast<char*>(start);
  char *iend = reinterpret_cast<char*>(istart + size);
  std::reverse(istart, iend);
}

bool close( double a, double b, double tol=1.0e-6) {
  /*
  double aa = std::abs(a);
  double bb = std::abs(b);
  double max = (aa > bb ) ? aa : bb;
  if(max < tol)
    return true;
  if(std::abs(a -b)/max <= tol)
  */
  if(std::abs(a -b) <= tol)    
    return true;
  return false;
}


int main(int argc, char* argv[]) {
  if(argc < 2) {
    //two streams of usage: (a) compare 2 NC file particle paths (b) makevtk for 1 file
    //1st stream is given first in <1st_stream/2nd_stream>
    //if no split in <>, then that is only for 2nd stream.
    // a:  ./bin <ncfile> <ncfile2> 1.0e-3 10000
    // b:  ./bin <ncfile> 1 5 # for 10 ptcls starting 5th
    std::cout << argv[0] << " <NCfile/NCfile> [ <NCfile2/numPtcls> <tolerance/pstart#> "
              << "<numTimesteps/numTimesteps> <tstart#>" 
              << " <nP-name> <nT-name> [<x-name> <y-name> <z-name>]]\n"
              << "Example: \n"
              << "./make_vtk <ncfile> [10 0 1000 0 \"nP\" \"nT\" \"x\" \"y\" \"z\"] \n";
    exit(1);
  }
  bool writeVTK = true;
  bool compare = false;
  long arg2 = 0;
  bool write_binary = false;  
  int debug = 0;
  std::string ncFileName = argv[1];
  std::string ncFileName2;

  int np = 0, nt = 0, pstart = 0, tstart = 0;
  double tol = 1.0e-6;
  if(argc > 2) {
    arg2 = strtol(argv[2], NULL, 0);
    if(arg2 <= 0) {
      compare = true;
      writeVTK = false;
      ncFileName2 = argv[2];
    } else
      np = atoi(argv[2]);
  }
  if(compare && argc > 3) {
    tol = atof(argv[3]);
    std::cout << "tolerance : " << tol << "\n";
  } else if(argc > 3)
    pstart = atoi(argv[3]);
  if(argc > 4)
    nt = atoi(argv[4]);
  if(argc > 5)
    tstart = atoi(argv[5]);

  std::string npName = "nP";
  std::string ntName = "nT";
  std::string xName = "x";
  std::string yName = "y";
  std::string zName = "z";

  if(argc > 6) 
    npName = argv[6];
  if(argc > 7)
    ntName = argv[7];
  if(argc > 10) {
    xName = argv[8];
    yName = argv[9];
    zName = argv[10];
  }

  std::vector<std::string> dimNames, varNames;
  dimNames.push_back(npName);
  dimNames.push_back(ntName);
  varNames.push_back(xName);
  varNames.push_back(yName);
  varNames.push_back(zName);
  std::vector<double> xdata, ydata, zdata;

  // Now reading all data. When NC can read subset, replace npFile & ntFile by np & nt
  int npFile = 0, ntFile = 0;
  readNetcdfData(ncFileName, dimNames, varNames, xdata,ydata, zdata, npFile, ntFile);

  if(np < 1 || np > npFile)
    np = npFile;
  if(nt < 1 || nt > ntFile)
    nt = ntFile;
  std::cout << "NcFile1:: nPFile: " << npFile << " nTFile: " << ntFile << "\n";
  std::cout << "Using:: nP: " << np << " nT: " << nt << "\n";  

  //2nd file
  std::vector<double> xdata2, ydata2, zdata2;
  int npFile2 = 0, ntFile2 = 0;
  if(compare) {
    std::cout << "\ncomparing with " << ncFileName2 << "\n";
    readNetcdfData(ncFileName2, dimNames, varNames, xdata2, ydata2, 
      zdata2, npFile2, ntFile2);
    if(np > npFile2)
      np = npFile2;
    if(nt > ntFile2)
      nt = ntFile2;
    std::cout << "NcFile2:: nPFile2: " << npFile2 << " nTFile2: " << ntFile2 << "\n";
    std::cout << "\nComparing with: nP " << np << " nT " << nt << "\n";
    long nmis = 0;
    long nptst = np*nt;
    std::vector<double> x1(nptst), y1(nptst), z1(nptst), x2(nptst), y2(nptst), z2(nptst);
    int k=0;
    for(int i=0, pf=pstart; i<np && pf<npFile; ++i, ++pf) {
      for(int j=0, tf=tstart; j<nt && tf<ntFile; ++j, ++tf) {
        x1[k] = xdata[pf*ntFile+tf];
        y1[k] = ydata[pf*ntFile+tf];
        z1[k] = zdata[pf*ntFile+tf];
        ++k;
      }
    }
    auto sizex1 = k;
    k = 0;
    for(int i=0, pf=pstart; i<np && pf<npFile2; ++i, ++pf) {
      for(int j=0, tf=tstart; j<nt && tf<ntFile2; ++j, ++tf) {    
        x2[k] = xdata2[pf*ntFile2+tf];
        y2[k] = ydata2[pf*ntFile2+tf];
        z2[k] = zdata2[pf*ntFile2+tf];      
        ++k;
      }
    }
    assert(sizex1 == k);
    int ptcl = -1, p = 0, ts=0;
    for(int i=0; i<nptst; ++i) {
      if(debug)
       std::cout << x1[i] << " " << x2[i] << " " << y1[i] << " " << y2[i]
                 <<  " " << z1[i] << " " << z2[i] << "\n"; 
      if(!(close(x1[i],x2[i], tol) && close(y1[i],y2[i],tol) && 
            close(z1[i],z2[i], tol))) {
        p = i/nt;
        ts = i - p*nt;
        if(ptcl != p) {
          ptcl = p;
          std::cout << "begin mismatch: ptcl " << ptcl << " step " << ts << " "  
            << x1[i] <<  " " << x2[i] << " "
            << y1[i] <<  " " << y2[i] << " "
            << z1[i] <<  " " << z2[i] << "\n";
        }
        ++nmis;
      }
    }
    std::cout << "Done comparing: " << nmis << " mismatches for tolerance " 
              << tol << "\n";
    // delta
    std::ofstream ofs("comparison.txt");
    ptcl = -1; p = 0; ts=0;
    for(int i=0; i<nptst; ++i) {
      p = i/nt;
      ts = i - p*nt;
      if(ptcl != p)
        ptcl = p;
      auto delta = std::sqrt((x1[i] - x2[i])*(x1[i] - x2[i]) + 
          (y1[i] - y2[i])*(y1[i] - y2[i]) + (z1[i] - z2[i])*(z1[i] - z2[i]));
      ofs << ptcl << " " << ts << " " << delta << "\n";
         //<< std::abs(x1[i] - x2[i]) << " "  << std::abs(y1[i] - y2[i]) << " " 
         //<<  std::abs(z1[i] - z2[i]) << "\n"; 
    }
  } //compare

  if(writeVTK) {
    //write vtk file for all particles
    int points = np*nt;
    int dataSize = 1;
    if(write_binary)
      dataSize = 3*points;
    std::vector<double> data(dataSize);
    int numCells = np*(nt-1);
    std::vector<unsigned long int>cellLinks(3*numCells);
    std::vector<int>cellCodes(numCells, 3);
    unsigned long int cumul = 0, cellN = 0;
    for(int i=0, pf=pstart; i<np && pf<npFile; ++i, ++pf) {
      for(int j=0, tf=tstart; j<nt && tf<ntFile; ++j, ++tf){
        if(write_binary) {
          data[(i*nt+j)*3] = xdata[pf*ntFile+tf];
          data[(i*nt+j)*3+1] = ydata[pf*ntFile+tf];
          data[(i*nt+j)*3+2] = zdata[pf*ntFile+tf];
        }
        if(j < nt-1) {
          cellLinks[3*cellN] = 2;
          cellLinks[3*cellN+1] = cumul;
          cellLinks[3*cellN+2] = cumul+1;
          ++cellN;
        }
        ++cumul;
      }
    }
    
    std::string outFileName = ncFileName.substr(0, ncFileName.find_last_of(".")) + ".vtk";
    outFileName = outFileName.substr(ncFileName.find_last_of("/")+1);
    std::cout << "Writing VTK file ./" << outFileName << "\n";
    std::ofstream outf;
    if(write_binary)
      outf.open(outFileName, std::ios::out | std::ios::binary);
    else
      outf.open(outFileName);

    outf << "# vtk DataFile Version 2.0\n" 
         << "particle paths\n";
    if(write_binary)  
       outf << "BINARY\n";
    else
       outf << "ASCII\n"; 
    outf << "DATASET UNSTRUCTURED_GRID\n"
         << "POINTS " << points << " double\n";
    std::cout << "Writing x,y,z \n";
    if(write_binary) {
      //TODO try swapping individual elements
      //https://stackoverflow.com/questions/55829282/write-vtk-file-in-binary-format
      reverseBytes(&data[0], 3*points*sizeof(double)); 
      reverseBytes(&cellLinks[0], 3*numCells*sizeof(unsigned long int)); 
      reverseBytes(&cellCodes[0], numCells*sizeof(int)); 
      outf.write((char*)&data[0], 3*points);
    } else {
      for(int i = pstart; i < pstart+np; ++i)
        for(int j = tstart; j < tstart+nt; ++j)
          outf << xdata[i*ntFile+j] << " " << ydata[i*ntFile+j] << " " << zdata[i*ntFile+j] << "\n";
    } 
    
    std::cout << "Writing cell links \n";
    outf << "CELLS " << numCells << " " << 3*numCells << "\n";
    if(write_binary)
      outf.write((char*)&cellLinks[0], 3*numCells);
    else
      for(int i=0; i< numCells; ++i)
        outf << cellLinks[3*i] << " " <<  cellLinks[3*i+1] << " " <<  cellLinks[3*i+2] << "\n";
    
    std::cout << "Writing cell-code 3 \n";
    outf << "CELL_TYPES " << numCells << "\n";
    if(write_binary)
      outf.write((char*)&cellCodes[0], numCells);
    else
      for(int i=0; i< numCells; ++i)
        outf << "3\n";

    outf.close();
    std::cout << "Done writing VTK !\n";  
  }
  return 0;
}
