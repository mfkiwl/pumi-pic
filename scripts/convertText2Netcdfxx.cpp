#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <netcdf>

// module load gcc mpich netcdf;
// export LD_LIBRARY_PATH=/lore/gopan/install/build-netcdfcxx431/install/lib64:$LD_LIBRARY_PATH
// g++ --std=c++11 <this.cpp> -o <bin> -I/lore/gopan/install/build-netcdfcxx431/install/include
// -lnetcdf -L/lore/gopan/install/build-netcdfcxx431/install/lib64/ -lnetcdf-cxx4

using namespace netCDF;
#define NC_ERR 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(2);}


struct PtclStruct {
  PtclStruct(std::string n, std::string np, std::string nt, std::string x, 
  	std::string y, std::string z,std::string vx, std::string vy, 
  	std::string vz):
    name(n), nPname(np), nTname(nt), xName(x), yName(y), zName(z), 
    vxName(vx), vyName(vy), vzName(vz) {}
  std::string name;
  std::string nPname;// "nP"
  std::string nTname; //nT
  std::string xName;
  std::string yName;
  std::string zName;
  std::string vxName;
  std::string vyName;
  std::string vzName;
  int nComp = 6;  //pos, vel
  int nP = 0;
  int nT = 0; //sub cycling for history
};

void parseFileFieldData(std::stringstream& ss, std::string sFirst, 
  std::string fieldName, bool semi, std::vector<double>& data, int& indP,
  int& indT, bool& dataLine, std::set<int>& nans, bool& expectEqual, 
  int iComp=0, int nComp=1, int numPtcls=0, int nT=1, int nthStep=0, 
  bool debug=false) {
  std::string s2 = "", sd = "";
  // restart index when string matches
  if(sFirst == fieldName) {
    indP = 0;
    ss >> s2;
    dataLine = true;
    if(s2 != "=")
      expectEqual = true;
    // next character in the same line after fieldNme is '='' 
    if(debug) {
      std::cout << " dataLine: " << dataLine << " of " << fieldName << "\n";
      if(!s2.empty() && s2 != "=")
        std::cout << "WARNING: Unexpected entry: " << s2 << " discarded\n";
    }
  }
  if(debug && dataLine)
    std::cout << ":: "<< indP << " : " << ss.str() << " ::1st " << sFirst 
              << " field " << fieldName << "\n";
  
  if(dataLine) {
    // this is done for every line, not only that of fieldName string
    if(!(sFirst.empty() || sFirst == fieldName)) {
      if((indP < numPtcls || !numPtcls) && 
        (indT == nthStep || nthStep<0)) {
        data[nComp*nT*indP + nComp*indT + iComp] = std::stod(sFirst);
        if(debug)
          printf("%s @ %d indP %d  iComp %d indT %d\n", 
            sFirst.c_str(), nComp*nT*indP + nComp*indT + iComp, indP, iComp, indT);
      }
      ++indT;
      if(debug)
        printf( "indT -> %d \n", indT);
      if(indT == nT) {
        indT = 0;
        ++indP;
        if(debug)
          printf("indT==nT: indP -> %d indT %d\n", indP, indT);
      }
    }  
    if(! ss.str().empty()) {
      while(ss >> sd) {
        if(numPtcls>0 && indP >= numPtcls) {
          if(debug)
            printf("break: numPtcls>0 && indP >= numPtcls \n");
          break;
        }
        // '=' if not with keyword, accept only if first next
        if(expectEqual) {
          expectEqual = false;
          if(sd=="=")
            continue;
        }

        if((indP < numPtcls || !numPtcls) && 
          (indT == nthStep || nthStep<0)) {
          data[nComp*nT*indP + nComp*indT + iComp] = std::stod(sd);
          if(debug)
            printf("%s @ %d indP %d  iComp %d indT %d\n", 
              sd.c_str(), nComp*nT*indP + nComp*indT + iComp, indP, iComp, indT);
        }
        ++indT;
        if(debug)
          printf( "indT -> %d nthStep %d \n", indT, nthStep);
        if(indT == nT) {
          indT = 0;
          ++indP;
          if(debug)
            printf("indT==nT: indP -> %d  indT %d \n", indP, indT);
        }
      } 
    }
    if(semi)
      dataLine = false;
  }
}


void processPtclFile(std::string& fName, std::vector<double>& data,
  PtclStruct& ps, int& numPtcls, int nthStep=0, bool debug=false, bool skipnT=false) {
  std::ifstream ifs(fName);
  if (!ifs.good()) { 
    printf("Error opening PtclInitFile file %s \n", fName.c_str());
    exit(1);
  }

  // can't set in ps, since field names in ps used below not from array
  int nComp = ps.nComp;
  bool foundNP, foundNT, dataInit, foundComp[nComp], dataLine[nComp]; //6=x,y,z,vx,vy,vz
  std::string fieldNames[nComp];
  bool expectEqual = false;
  int indP[nComp], indT[nComp];
  std::set<int> nans;
  for(int i = 0; i < nComp; ++i) {
    indP[i] = 0;
    indT[i] = 0;
    foundComp[i] = dataLine[i] = false;
  }

  fieldNames[0] = ps.xName;
  fieldNames[1] = ps.yName;
  fieldNames[2] = ps.zName;
  fieldNames[3] = ps.vxName;
  fieldNames[4] = ps.vyName;
  fieldNames[5] = ps.vzName;      
  foundNP = foundNT = dataInit = false;
  if(skipnT)
    foundNT = true;

  std::string line, s1, s2, s3;
  while(std::getline(ifs, line)) {
    if(debug)
      std::cout << "Processing  line " << line << '\n';
    // depend on semicolon to mark the end of fields, otherwise
    // data of unused fields added to the previous valid field.
    bool semi = (line.find(';') != std::string::npos);
    std::replace (line.begin(), line.end(), ',' , ' ');
    std::replace (line.begin(), line.end(), ';' , ' ');
    std::stringstream ss(line);
    // first string or number of EACH LINE is got here
    ss >> s1;
    if(debug)
      std::cout << "str s1:" << s1 << "\n";
    
    // Skip blank line
    if(s1.find_first_not_of(' ') == std::string::npos) {
      s1 = "";
      if(!semi)
       continue;
    }
    if(s1 == ps.nPname) {
      ss >> s2 >> s3;
      assert(s2 == "=");
      ps.nP = std::stoi(s3);
      if(numPtcls <= 0)
        numPtcls = ps.nP;
      else if(numPtcls < ps.nP)
        ps.nP = numPtcls;
      else if(numPtcls > ps.nP) {
        numPtcls = ps.nP;
        if(debug)
          std::cout << "Warning: numPtcls " << numPtcls << " reset to " 
                 << ps.nP << ", max. in file\n";
      }
      foundNP = true;
      if(debug)
          std::cout << "nP:" << ps.nP << " Using numPtcls " << numPtcls << "\n";
    }
    if(s1 == ps.nTname) {
      ss >> s2 >> s3;
      assert(s2 == "=");
      ps.nT = std::stoi(s3);
      //if(nthStep <= 0)
      //  nthStep = 0;
      //else 
      if(nthStep >= ps.nT)
      	nthStep = ps.nT-1;
      foundNT = true;
      if(debug)
      	std::cout << " foundNT: setting nT: " << ps.nT << "\n";
    }

    if(!dataInit && foundNP && foundNT) {
    	if(debug)
    	  std::cout << " nComp " << nComp <<  " nP:" << ps.nP 
    	    << " Vsize " << nComp*ps.nP << "\n";
      data.resize(nComp*ps.nP);
      dataInit = true;
    }
    int compBeg = 0, compEnd = nComp;
    // if ; ends data of each parameters, otherwise comment this block
    // to search for each parameter for every data line
    for(int iComp = 0; iComp<nComp; ++iComp) {
      if(dataInit && dataLine[iComp]) {
        compBeg = iComp;
        compEnd = compBeg + 1;
      }
    }
    // NOTE: NaN is replaced with 0 to preserve sequential index of particle
    //TODO change it in storeData()
    if(dataInit) {
      // stored in a single data array of 6+1 components.
      for(int iComp = compBeg; iComp<compEnd; ++iComp) {
        parseFileFieldData(ss, s1, fieldNames[iComp], semi, data, indP[iComp], 
          indT[iComp], dataLine[iComp], nans, expectEqual, iComp, nComp, 
          numPtcls, ps.nT, nthStep, debug);

        if(!foundComp[iComp] && dataLine[iComp]) {
          foundComp[iComp] = true;
          if(debug)
            printf("Found data Component %d\n", iComp);
        }
      }
    }
    s1 = s2 = s3 = "";
  } //while
}


// NOTE: GITR takes last particle as all zeros. Need an extra last 
// dummy ptcl in input source txt file.
int main(int argc, char** argv) {
  if(argc < 2) {
    std::cout << "Usage: " << argv[0] 
      << " <ptcl_src_file>[<outFile.nc>]\n";
    exit(1);
  }
  bool debug = false;

  std::string ptclSource = argv[1];
  std::string outFile = "particleSource_selected.nc";
  if(argc>2)
    outFile = argv[2];

  int numPtcls = 0;
  int nthStep = -1;

  constexpr int dof = 6;
  std::vector<double> data;
  PtclStruct pst("ptcl_source", "nP", "nT", "x", "y", "z", "vx", "vy", "vz");
  processPtclFile(ptclSource, data, pst, numPtcls, nthStep, debug, true);
  
  int np = data.size()/dof;
  double* px = new double[np];
  double* py = new double[np];
  double* pz = new double[np];
  double* pvx = new double[np];
  double* pvy = new double[np];
  double* pvz = new double[np];
  
  int sz = data.size();  
  printf("np %d %d \n", np, sz);
  for(int i=0; i<np ;++i) {
    px[i] = double(data[dof*i]);
    py[i] = double(data[dof*i + 1]);
    pz[i] = double(data[dof*i + 2]);
    pvx[i] = double(data[dof*i + 3]);
    pvy[i] = double(data[dof*i + 4]);
    pvz[i] = double(data[dof*i + 5]);
  }

  printf("x0x1 %g %g \n", px[0], px[1]);

  NcFile ncFile("particleSourceSelected.nc", NcFile::replace);
  NcDim pnp = ncFile.addDim("nP", np); 
  int cst = 1;
  NcDim cons = ncFile.addDim("const", cst);
  NcVar rt = ncFile.addVar("rate", ncDouble, cons);
  NcVar xx = ncFile.addVar("x", ncDouble, pnp);
  NcVar yy = ncFile.addVar("y", ncDouble, pnp);
  NcVar zz = ncFile.addVar("z", ncDouble, pnp);
  NcVar vxx = ncFile.addVar("vx", ncDouble, pnp);
  NcVar vyy = ncFile.addVar("vy", ncDouble, pnp);
  NcVar vzz = ncFile.addVar("vz", ncDouble, pnp);
  xx.putVar(&px[0]);
  yy.putVar(&py[0]);
  zz.putVar(&pz[0]);
  vxx.putVar(&pvx[0]);
  vyy.putVar(&pvy[0]);
  vzz.putVar(&pvz[0]);
  ncFile.close();

  /*
  int ret, ncid, dimid, dimidvar, constid, varidx, varidy, varidz;
  int varidvx, varidvy, varidvz, varidsx, varidsy, varidsz;
  
  nc_create(outFile.c_str(), NC_CLOBBER, &ncid);
  nc_def_dim(ncid, "nP",  np, &dimid);
  int cst = 1;
  nc_def_dim(ncid, "const", cst, &constid); 

  if ((ret = nc_def_var(ncid, "rate", NC_FLOAT, 1, &dimidvar, &constid)))
    ERR(ret);

  if ((ret = nc_def_var(ncid, "x", NC_FLOAT, 1, &dimid, &varidx)))
    ERR(ret);
  if ((ret = nc_def_var(ncid, "y", NC_FLOAT, 1, &dimid, &varidy)))
    ERR(ret);
  if ((ret = nc_def_var(ncid, "z", NC_FLOAT, 1, &dimid, &varidz)))
    ERR(ret);
  if ((ret = nc_def_var(ncid, "vx", NC_FLOAT, 1, &dimid, &varidvx)))
    ERR(ret); 
  if ((ret = nc_def_var(ncid, "vy", NC_FLOAT, 1, &dimid, &varidvy)))
    ERR(ret);
  if ((ret = nc_def_var(ncid, "vz", NC_FLOAT, 1, &dimid, &varidvz)))
    ERR(ret);

  
  if((ret = nc_enddef(ncid)))
    ERR(ret);
  printf("Done defining variables \n");

  if((ret = nc_put_var_float(ncid, varidx, px)))
    ERR(ret);
  if((ret = nc_put_var_float(ncid, varidy, py)))
    ERR(ret);
  if((ret = nc_put_var_float(ncid, varidz, pz)))
    ERR(ret);  
  if((ret = nc_put_var_float(ncid, varidvx, pvx)))
    ERR(ret);
  if((ret = nc_put_var_float(ncid, varidvy, pvy)))
    ERR(ret);
  if((ret = nc_put_var_float(ncid, varidvz, pvz)))
    ERR(ret);  
  printf("Done putting variables\n");
  if((ret = nc_close(ncid)))
    ERR(ret);
  */
  delete [] px, py, pz, pvx, pvy, pvz;

  return 0;
}
