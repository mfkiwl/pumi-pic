#include <fstream>
#include <cmath>
#include <utility>
#include <sstream>
#include "Omega_h_fail.hpp"
#include "Omega_h_array.hpp"
#include "Omega_h_array_ops.hpp"
#include <netcdf>
#include <netcdf_par.h>
#include <cstddef>
#include "GitrmInputOutput.hpp"

double Field3StructInput::getGridDelta(int ind) {
  OMEGA_H_CHECK(ind < MAX_SIZE && ind >=0);
  OMEGA_H_CHECK(nGridVec.size() > ind && nGridVec[ind] > 0);
  //return (getGridMax(ind) - getGridMin(ind))/(double)nGridVec[ind];
  auto second = (ind>0) ? ((ind>1)? (grid3[1]): grid2[1]) : grid1[1];
  return second - getGridMin(ind);
}

int Field3StructInput::getNumGrids(int ind) {
  OMEGA_H_CHECK(ind < MAX_SIZE && ind >=0);
  OMEGA_H_CHECK(nGridVec.size() > ind);
  return nGridVec[ind];
}

double Field3StructInput::getGridMin(int ind) {
  OMEGA_H_CHECK(ind >=0 && ind < MAX_SIZE);
  OMEGA_H_CHECK(MAX_SIZE ==3);
  return (ind>0) ? ((ind>1)? (grid3[0]): grid2[0]) : grid1[0];
}

double Field3StructInput::getGridMax(int ind) {
  OMEGA_H_CHECK(ind >=0 && ind < 3);
  OMEGA_H_CHECK(MAX_SIZE == 3);
  double max = 0;
  if(ind==0)
    max = grid1[grid1.size()-1];
  else if(ind==1)
    max = grid2[grid2.size()-1];
  else if(ind==2)
    max = grid3[grid3.size()-1];
  return max;
}

int Field3StructInput::getIntValueOf(std::string name) {
  for(int i=0; i< nVarNames.size(); ++i) {
    if(name == nVarNames[i]) {
      if(nVarVec.size() > i) {
        return nVarVec[i];
      }
    }
  }
  for(int i=0; i< nGridNames.size(); ++i) {
    if(name == nGridNames[i]) {
      if(nGridVec.size() > i) {
        return nGridVec[i];
      }
    }
  }
  Omega_h_fail("Error: getIntValueOf %s\n", name.c_str());
  return -1;
}

int verifyNetcdfFile(const std::string& ncFileName, int nc_err) {
  int status = 0;
  try {
    netCDF::NcFile ncFile(ncFileName, netCDF::NcFile::read);
  } catch (netCDF::exceptions::NcException &e) {
    std::cout << e.what() << std::endl;
    status = nc_err;
  }
  if(status)
    Omega_h_fail("Error: invalid NetCDF file %s\n", ncFileName.c_str());
  return status;
}


bool readParticleSourceNcFile(std::string ncFileName, o::HostWrite<o::Real>& data,
   int& numPtcls, size_t each_chunk, size_t each_chunk_pos, bool replaceNaN) {
    int myrank = -1, numranks = 0, ncid = 0, err = 0, np_id = 0, num_vrs = 0;
    char str_char[256]={};
    size_t np_length = 0;
    bool status = true;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    //printf("While reading num ranks is %d \n", numranks);

    size_t myrank_s = size_t(myrank);
    //std::cout<<"The passed string is"<<ncFileName<<std::endl;
    std::strcpy (str_char, ncFileName.c_str());

    char const *vrs[] = {"x","y","z","vx","vy","vz"};
    num_vrs = sizeof(vrs)/sizeof(vrs[0]);
    //printf("Number of variables in netCDF file %d \n",num_vrs);
    int* vrs_ind = nullptr;
    vrs_ind=(int *)calloc(num_vrs, sizeof(int));

    err=nc_open_par(str_char, NC_NETCDF4|NC_NOWRITE, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid);
      if (err!=0){
        printf("The error0 status should be 0 is %d \n" , err);
        printf("%s\n", nc_strerror(err));
        exit(-1);
      }

    err=nc_inq_dimid(ncid, "nP", &np_id);
     if (err!=0){
      printf("The error1 status should be 0 is %d \n" , err);
      printf("%s\n", nc_strerror(err));
      exit(-1);
    }

    err=nc_inq_dimlen(ncid, np_id, &np_length);
    if (err!=0){
      printf("The error2 status should be 0 is %d \n" , err);
      printf("%s\n", nc_strerror(err));
      exit(-1);
    }

    numPtcls=int(each_chunk);
    //double * nc_values;
    static size_t start[] = {myrank_s*each_chunk_pos};
    static size_t count[] = {each_chunk};
    //nc_values=(double*)calloc(num_vrs*num_ptcls, sizeof(double));
    data = o::HostWrite<o::Real>(o::Write<o::Real>(numPtcls*num_vrs, 0, "readData"));

    for (int i=0; i<num_vrs; i++){
      err=nc_inq_varid (ncid, vrs[i], vrs_ind+i);
      if (err!=0){
        printf("The error3 status should be 0 is %d \n" , err);
        printf("%s\n", nc_strerror(err));
        exit(-1);
      }
      err = nc_var_par_access(ncid, *(vrs_ind+i), NC_INDEPENDENT);
      if (err!=0){
        printf("The error4 status should be 0 is %d \n" , err);
        printf("%s\n", nc_strerror(err));
        exit(-1);
      }
      err=nc_get_vara_double(ncid, *(vrs_ind+i), start, count, &data[i*each_chunk]);
      if (err!=0){
        printf("The error5 status should be 0 is %d \n" , err);
        printf("%s\n", nc_strerror(err));
        exit(-1);
      }
    }

    err=nc_close(ncid);
      if (err!=0){
        printf("The error5 status should be 0 is %d \n" , err);
        printf("%s\n", nc_strerror(err));
        exit(-1);
      }

    if(replaceNaN) {

      long int nans = 0;
        for(auto i=0; i<data.size(); ++i)
          if(std::isnan(data[i])) {
            data[i] = 0;
            ++nans;
          }
      if(nans)
          printf("\n*******WARNING replaced %ld NaNs in ptclSrc *******\n\n", nans);
    }

    free (vrs_ind);
    return status;
}


bool readParticleSourceNcFile(std::string ncFileName, o::HostWrite<o::Real>& data,
  int& numPtcls, bool replaceNaN) {
    constexpr int dof = 6;
    bool status = true;
    verifyNetcdfFile(ncFileName);
    try {
      netCDF::NcFile ncf(ncFileName, netCDF::NcFile::read);
    netCDF::NcDim ncf_np(ncf.getDim("nP"));
    auto np = ncf_np.getSize();
    OMEGA_H_CHECK(numPtcls <= np);
    //if(numPtcls >0 && numPtcls < np)
    //  np = numPtcls;
    numPtcls = np;
    std::cout << " nPtcls in source file " << np << "\n";
    data = o::HostWrite<o::Real>(o::Write<o::Real>(np*dof, 0, "data"));
    netCDF::NcVar ncx(ncf.getVar("x"));
    netCDF::NcVar ncy(ncf.getVar("y"));
    netCDF::NcVar ncz(ncf.getVar("z"));
    netCDF::NcVar ncvx(ncf.getVar("vx"));
    netCDF::NcVar ncvy(ncf.getVar("vy"));
    netCDF::NcVar ncvz(ncf.getVar("vz"));
    ncx.getVar(&data[0]);
    ncy.getVar(&data[1*np]);
    ncz.getVar(&data[2*np]);
    ncvx.getVar(&data[3*np]);
    ncvy.getVar(&data[4*np]);
    ncvz.getVar(&data[5*np]);
    } catch (netCDF::exceptions::NcException &e) {
    std::cout << e.what() << std::endl;
    status = false;
    }
    printf("The starting data is %e %p \n", data[0], &data[0]);
    if(replaceNaN) {
    long int nans = 0;
    for(auto i=0; i<data.size(); ++i)
      if(std::isnan(data[i])) {
        data[i] = 0;
        ++nans;
      }
    if(nans)
      printf("\n*******WARNING replaced %ld NaNs in ptclSrc *******\n\n", nans);
      printf("The total size of data to be loaded in GPU is %d\n",data.size());
  }
  if(!status)
    Omega_h_fail("ERROR: reading file %s\n", ncFileName.c_str());
  return status;
}


//Reads from 0 to 3 grids having gridNames; .
int readInputDataNcFileFS3(const std::string& ncFileName,
  Field3StructInput& fs, bool debug) {
  int numInFile = 0;
  int numRead = 0;
  return readInputDataNcFileFS3(ncFileName, fs, numInFile, numRead, "NA", debug);
}

// numInFile updated if > that in file.
int readInputDataNcFileFS3(const std::string& ncFileName,
  Field3StructInput& fs, int& numInFile, int& numRead,
  std::string numStr, bool debug) {
  int ncSizePerComp = 1;
  int status = 0;
  //verifyNetcdfFile(ncFileName);
  if(debug)
    printf("Reading NC file %s\n", ncFileName.c_str());
  try {
    netCDF::NcFile ncf(ncFileName, netCDF::NcFile::read);
    int dimCount = ncf.getDimCount();
    int nameCount = fs.nVarNames.size() + fs.nGridNames.size();
    if(dimCount < nameCount)
      std::cout << " WARNING: mismatch between NC file #dims and #reading\n";
    for(int i=0; i< fs.nGridNames.size(); ++i) {
      netCDF::NcDim ncGridName(ncf.getDim(fs.nGridNames[i].c_str()));
      auto unlimit = ncGridName.isUnlimited();
      if(unlimit)
        std::cout << "\nWarning: value of " << fs.nGridNames[i]<< " set to 0 \n";
      int size = 0;
      if(!unlimit)
        size = ncGridName.getSize();
      fs.nGridVec.push_back(size);
      if(fs.nGridNames[i] == numStr) {
        numInFile = size;
        //if(size < numRead) //TODO handle it below
          numRead = size;
      }
      if(debug)
        std::cout << ncFileName << " : " << fs.nGridNames[i] << " : "
          << fs.nGridVec[i] << "\n";
      ncSizePerComp *= fs.nGridVec[i];
    }
    if(debug)
      std::cout << " ncSizePerComp: " << ncSizePerComp << " nComp " << fs.nComp << "\n";

    for(int i=0; i<fs.nGridRead && i<fs.gridNames.size(); ++i) {
      netCDF::NcVar ncvar(ncf.getVar(fs.gridNames[i].c_str()));
      if(debug)
        std::cout << "  reading:: " << fs.gridNames[i]  << "\n";
      if(i==0) {
        fs.grid1 = o::HostWrite<o::Real>(o::Write<o::Real>(fs.nGridVec[0], 0, "nGridVec0"));
        ncvar.getVar(&(fs.grid1[0]));
        if(debug)
          printf("  read: size %d %s %g %g ...\n", fs.nGridVec[0],
              fs.gridNames[i].c_str(), fs.grid1[0], fs.grid1[1]);
      }
      if(i==1) {
        fs.grid2 = o::HostWrite<o::Real>(o::Write<o::Real>(fs.nGridVec[1], 0, "nGridVec1"));
        ncvar.getVar(&(fs.grid2[0]));
        if(debug)
          printf("  read: size %d %s %g %g ...\n", fs.nGridVec[1],
              fs.gridNames[i].c_str(), fs.grid2[0], fs.grid2[1]);
      }
      if(i==2) {
        fs.grid3 = o::HostWrite<o::Real>(o::Write<o::Real>(fs.nGridVec[2], 0, "nGridVec2"));
        ncvar.getVar(&(fs.grid3[0]));
      }
    }
    fs.data = o::HostWrite<o::Real>(o::Write<o::Real>(ncSizePerComp*fs.nComp, 0, "fs.data"));
    for(int i=0; i<fs.nComp; ++i) {
      if(debug)
        std::cout << "comp " << i << " " << fs.compNames[i] << "\n";
      o::HostWrite<o::Real>temp(ncSizePerComp, "temp-data");
      netCDF::NcVar ncvar(ncf.getVar(fs.compNames[i].c_str()));
      //ncvar.getVar(&(fs.data[i*ncSizePerComp]));
      ncvar.getVar(&temp[0]);
      for(int j=0; j<ncSizePerComp; ++j) {
        fs.data[j*fs.nComp + i] = temp[j];
        if(debug && j <10) {
          printf(" i,j,dat %d %d %g ", i,j,fs.data[j*fs.nComp+i]);
          printf("\n");
        }
      }
    }
    for(int i=0; i< fs.nVarNames.size(); ++i) {
      netCDF::NcDim ncVarName(ncf.getDim(fs.nVarNames[i]));
      auto unlimit = ncVarName.isUnlimited();
      if(unlimit)
        std::cout << " Warning: value of " << ncVarName.getName() << " set to 0 \n";
      int size = 0;
      if(!unlimit)
        size = ncVarName.getSize();
      if(ncf.getDim(fs.nVarNames[i]).isNull())
        size = -1;
      if(debug)
        std::cout << " " << fs.nVarNames[i] << " " << ncVarName.getName()
          << " size " << size << " \n";
      fs.nVarVec.push_back(size);
    }

    ncf.close();
  } catch (netCDF::exceptions::NcException &e) {
    std::cout << e.what() << std::endl;
    status = 1;
  }

  // replace NaNs
  int nans = 0;
  for(int i=0; i<fs.data.size(); ++i)
    if(std::isnan(fs.data[i])) {
      fs.data[i] = 0;
      ++nans;
    }
  if(nans)
    printf("\n*******WARNING replaced %d NaNs in data ******\n\n", nans);

  if(status)
    Omega_h_fail("ERROR: failed reading file %s\n", ncFileName.c_str());
  if(debug)
    std::cout << " Done reading " << ncFileName << "\n";
  return status;
}

//TODO combine nc write functions
void writeOutBdryFaceCoordsNcFile(const std::string& fileName,
    o::Write<o::Real>& xd, o::Write<o::Real>& yd, o::Write<o::Real>& zd,
    const int nf) {
  auto xx = o::HostRead<o::Real>(xd);
  auto yy = o::HostRead<o::Real>(yd);
  auto zz = o::HostRead<o::Real>(zd);
  int status = 0;
  try {
    netCDF::NcFile ncMeshFile(fileName, netCDF::NcFile::replace);
    netCDF::NcDim nc_cells = ncMeshFile.addDim("ncells", nf);
    netCDF::NcDim nc_xnums = ncMeshFile.addDim("nx", nf*3);
    netCDF::NcDim nc_ynums = ncMeshFile.addDim("ny", nf*3);
    netCDF::NcDim nc_znums = ncMeshFile.addDim("nz", nf*3);
    std::vector<netCDF::NcDim> dims_ncx;
    dims_ncx.push_back(nc_xnums);
    std::vector<netCDF::NcDim> dims_ncy;
    dims_ncy.push_back(nc_ynums);
    std::vector<netCDF::NcDim> dims_ncz;
    dims_ncz.push_back(nc_znums);
    netCDF::NcVar ncvx = ncMeshFile.addVar("x", netCDF::ncDouble, dims_ncx);
    netCDF::NcVar ncvy = ncMeshFile.addVar("y", netCDF::ncDouble, dims_ncy);
    netCDF::NcVar ncvz = ncMeshFile.addVar("z", netCDF::ncDouble, dims_ncz);
    ncvx.putVar(&xx[0]);
    ncvy.putVar(&yy[0]);
    ncvz.putVar(&zz[0]);
  } catch (netCDF::exceptions::NcException& e) {
    std::cout << e.what() << "\n";
    status = 1;
  }
  if(status)
    Omega_h_fail("ERROR: failed writing file %s\n", fileName.c_str());
}


void writeOutputNcFile( o::HostWrite<o::Real>& ptclHistoryData, int numPtcls,
  long int totalPtcls, int nThistory, std::string outNcFileName) {

    int myrank, numranks, ncid, err, num_vrs, num_dim, each_chunk;
    char str_char[100];

    char const *vrs[]={"x","y","z","vx","vy","vz"};
    char const *dim[]={"nP", "nT"};
    num_vrs=sizeof(vrs)/sizeof(vrs[0]);
    num_dim=sizeof(dim)/sizeof(dim[0]);

    //printf("Number of variables in result netCDF file %d \n",num_vrs);
    //printf("Number of dimesnions in result netCDF file %d \n",num_dim);

    int* vrs_ind;
    int* dim_ind;
    long int* dim_count;

    vrs_ind=(int *)calloc(num_vrs, sizeof(int));
    dim_ind=(int *)calloc(num_dim, sizeof(int));
    dim_count=(long int *)calloc(num_dim, sizeof(long int));

    dim_count[0]=totalPtcls;
    dim_count[1]=nThistory;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    each_chunk=totalPtcls/numranks;
    size_t each_chunk_pos=each_chunk;
    if (myrank==numranks-1) each_chunk=each_chunk+ totalPtcls%numranks;
    //printf("Total ptcls is %d \n",totalPtcls);
    //printf("Total ranks is %d \n",numranks);
    //printf("Each chunk is %d \n", each_chunk);

    std::strcpy(str_char, outNcFileName.c_str());
    size_t myrank_s=size_t(myrank);
    //printf("Entering parallel netCDF routine \n");
    err=nc_create_par(str_char, NC_NETCDF4|NC_MPIIO, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid);
      if (err!=0){
        printf("The error0 status should be 0 is %d \n" , err);
        printf("%s\n", nc_strerror(err));
        exit(-1);
      }

    for (int i=0; i<num_dim;i++){
      err=nc_def_dim(ncid, dim[i], dim_count[i], dim_ind+i);
        if (err!=0){
          printf("The error1 status should be 0 is %d \n" , err);
          printf("%s\n", nc_strerror(err));
          exit(-1);
        }
    }

    for (int i=0; i<num_vrs;i++){
      err=nc_def_var(ncid, vrs[i], NC_DOUBLE, num_dim, dim_ind, vrs_ind+i);
        if (err!=0){
          printf("The error2 status should be 0 is %d \n" , err);
          printf("%s\n", nc_strerror(err));
          exit(-1);
        }
    }

    size_t start[2];
    size_t count[2];
    //printf("ptclHistoryData.size() is %d \n", ptclHistoryData.size());

    for (int i=0; i<num_vrs;i++){

      err=nc_var_par_access(ncid, *(vrs_ind+i), NC_INDEPENDENT);
        if (err!=0){
          printf("The error3 status should be 0 is %d \n" , err);
          printf("%s\n", nc_strerror(err));
          exit(-1);
      }

      //int ref = i*totalPtcls*nThistory+myrank_s*each_chunk*nThistory;
      int ref=i*each_chunk*nThistory;
      start[0]=myrank_s*(each_chunk_pos);
      //printf("Starting is %d \n", start[0]);
      start[1] = 0;
      count[0] = each_chunk;
      count[1] = nThistory;

      err=nc_put_vara_double(ncid, *(vrs_ind+i), start, count, &ptclHistoryData[ref]);
        if (err!=0){
          printf("The error4 status should be 0 is %d \n" , err);
          printf("%s\n", nc_strerror(err));
          exit(-1);
        }
    }

      err=nc_close(ncid);
        if (err!=0){
          printf("The error5 status should be 0 is %d \n" , err);
          printf("%s\n", nc_strerror(err));
          exit(-1);
        }

    free(vrs_ind);
    free(dim_ind);
    free(dim_count);

}

void writeOutputNcFile( o::HostWrite<o::Real>& ptclHistoryData, int numPtcls,
  int dof, OutputNcFileFieldStruct& st, std::string outNcFileName) {
  //if ext not nc,
  //outNcFileName = outNcFileName + std::to_string(i) + ".nc";
  OMEGA_H_CHECK(dof == st.fieldNames.size());
  OMEGA_H_CHECK(numPtcls == st.nDims[0]);
  int status = 0;
  try {
    netCDF::NcFile ncFile(outNcFileName, netCDF::NcFile::replace);
    std::vector<netCDF::NcDim> ncDims;
    for(int i=0; i< st.nDims.size(); ++i) {
      netCDF::NcDim dim = ncFile.addDim(st.numStrings[i], st.nDims[i]);
      ncDims.push_back(dim);
    }
    std::vector<netCDF::NcVar> ncVars;
    for(int i=0; i< st.fieldNames.size(); ++i) {
      netCDF::NcVar var = ncFile.addVar(st.fieldNames[i],
        netCDF::NcDouble(), ncDims);
      ncVars.push_back(var);
    }
    //stored in timestep order : numPtcls*dof * iHistStep + id*dof
    int nHist = (int)ptclHistoryData.size()/(dof*numPtcls);

    //each component is written as seperate Var
    for(int i=0; i< dof; ++i) {
      o::HostWrite<o::Real> dat(nHist*numPtcls);
      for(int j=0; j<numPtcls; ++j) {
        for(int k=0; k<nHist; ++k) {
          dat[j*nHist+k] = ptclHistoryData[j*dof + i + k*numPtcls*dof];
        }
      }
      ncVars[i].putVar(&(dat[0]));

      //const std::vector<size_t> start{(size_t)i};
      //const std::vector<size_t> count{(size_t)numPtclxHistn};
      //const std::vector<ptrdiff_t> stridep{dof};
      // http://unidata.github.io/netcdf-cxx4/ncVar_8cpp_source.html#l00788 line-1142
      // https://github.com/Unidata/netcdf-c/blob/master/ncdump/ref_ctest.c
      //ncVars[i].putVar(start, count, stridep, &(ptclHistoryData[0]));

    }
  } catch (netCDF::exceptions::NcException& e) {
    std::cout << e.what() << "\n";
    status = 1;
  }
  if(status)
    Omega_h_fail("ERROR: failed writing file %s \n", outNcFileName.c_str());
}

int readCsrFile(const std::string& ncFileName, const std::vector<std::string>& vars,
  const std::vector<std::string>& datNames, o::HostWrite<o::LO>& ptrs,
  o::HostWrite<o::LO>& data) {
  int status = 0;
  try {
    netCDF::NcFile ncf(ncFileName, netCDF::NcFile::read);
    netCDF::NcDim ncPtrName(ncf.getDim(vars[0].c_str()));
    auto psize = ncPtrName.getSize();
    std::cout << ncFileName << " : " << vars[0] << " : " << psize << "\n";

    ptrs = o::HostWrite<o::LO>(o::Write<o::LO>(psize, 0, "ptrs"));
    netCDF::NcVar ncp(ncf.getVar(datNames[0]));
    ncp.getVar(&(ptrs[0]));

    netCDF::NcDim ncDataName(ncf.getDim(vars[1]));
    auto dsize = ncDataName.getSize();
    std::cout << ncFileName << " : " << vars[1] << " : " << dsize << "\n";
    data = o::HostWrite<o::LO>(o::Write<o::LO>(dsize, 0, "data"));
    netCDF::NcVar ncd(ncf.getVar(datNames[1]));
    ncd.getVar(&(data[0]));

    int nans = 0;
    for(int i=0; i<data.size(); ++i)
      if(std::isnan(data[i]))
        ++nans;
    if(nans)
      printf("\n*******WARNING found %d NaNs in data ******\n\n", nans);
    std::cout << " Done reading " << ncFileName << "\n\n";
  } catch (netCDF::exceptions::NcException &e) {
    std::cout << e.what() << std::endl;
    status = 1;
  }
  if(status)
    Omega_h_fail("ERROR: failed reading file %s \n", ncFileName.c_str());
  return status;
}

void writeOutputCsrFile(const std::string& outFileName,
    const std::vector<std::string>& vars, const std::vector<std::string>& datNames,
    o::LOs& ptrs_d, o::LOs& data_d, int* valExtra) {
  auto data = o::HostRead<o::LO>(data_d);
  auto ptrs = o::HostRead<o::LO>(ptrs_d);
  int psize = ptrs.size();
  int dsize = data.size();
  int status = 0;
  try {
    netCDF::NcFile ncFile(outFileName, netCDF::NcFile::replace);
    //TODO pass values to remove ordering
    netCDF::NcDim dim1 = ncFile.addDim(vars[0], psize);
    netCDF::NcDim dim2 = ncFile.addDim(vars[1], dsize);
    for(auto i=2; i<vars.size(); ++i)
      netCDF::NcDim dext = ncFile.addDim(vars[i], valExtra[i-2]);
    netCDF::NcVar ncptrs = ncFile.addVar(datNames[0], netCDF::ncInt, dim1);
    ncptrs.putVar(&(ptrs[0]));
    netCDF::NcVar ncdata = ncFile.addVar(datNames[1], netCDF::ncInt, dim2);
    ncdata.putVar(&(data[0]));
  } catch (netCDF::exceptions::NcException& e) {
    std::cout << e.what() << "\n";
    status = 1;
  }
  if(status)
    Omega_h_fail("Error: writing NC file %s failed\n", outFileName.c_str());
}
