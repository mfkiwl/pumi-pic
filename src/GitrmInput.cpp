#include <iostream>
#include <mpi.h>
#include "GitrmInput.hpp"

GitrmInput::GitrmInput(const std::string& fname, bool debug): cfgFileName(fname) {
  cfg = new libconfig::Config;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  parseConfigFile();

  if(debug)
    testInputConfig();
}

int GitrmInput::parseConfigFile() {
  cfg->setAutoConvert(true);
  try {
    cfg->readFile(cfgFileName);
  } catch(const libconfig::FileIOException& fio) {
    if(!myRank)
      std::cout << "I/O error while reading file." << std::endl;
    return(EXIT_FAILURE);
  } catch(const libconfig::ParseException& ex) {
    if(!myRank)
      std::cout << "Parse error at " << ex.getFile() << ":" 
        << ex.getLine() << " - " << ex.getError() << std::endl;
    return(EXIT_FAILURE);
  }
  if(!myRank)
    std::cout << "Done parsing config file " << cfgFileName << std::endl;
  return 0;
}

void GitrmInput::testInputConfig() {
  //default auto are string's
  auto geomFile = getConfigVar("geometry.fileString");
  std::string ptclSrcTag{"particleSource"};
  auto ptclSrcCfg = getConfigVar(ptclSrcTag+".fileString");
  auto ptclSrcNcFile = getConfigVar(ptclSrcTag+".ncFileString");
  std::string sm{"surfaceModel."};
  auto smFile = getConfigVar(sm+"fileString");
  auto smNEnSput = getConfigVar(sm+"nEsputtRefCoeffString");
  auto smNAngSput = getConfigVar(sm+"nAsputtRefCoeffString");
  auto smEnSputIn = getConfigVar(sm+"nEsputtRefDistInString");
  auto smAngSputIn = getConfigVar(sm+"nAsputtRefDistInString");
  auto smNEnSputRef = getConfigVar(sm+"nEsputtRefDistOutString");
  auto smNEnSputRefOut = getConfigVar(sm+"nEsputtRefDistOutStringRef");
  auto smNAngSputOut = getConfigVar(sm+"nAsputtRefDistOutString");
  auto smEnSputRef = getConfigVar(sm+"E_sputtRefCoeff");
  auto smAngSputRef = getConfigVar(sm+"A_sputtRefCoeff");
  auto smEnSputRefIn = getConfigVar(sm+"E_sputtRefDistIn");
  auto smAngSputRefIn = getConfigVar(sm+"A_sputtRefDistIn");
  auto smEnSputRefOut = getConfigVar(sm+"E_sputtRefDistOut");
  auto smEnSputRefOutRef = getConfigVar(sm+"E_sputtRefDistOutRef");
  auto smAphiSputRefOut = getConfigVar(sm+"Aphi_sputtRefDistOut");
  auto smAthSputRefOut = getConfigVar(sm+"Atheta_sputtRefDistOut");
  auto smSputYld = getConfigVar(sm+"sputtYldString");
  auto smReflYld = getConfigVar(sm+"reflYldString");
  auto smEnYld = getConfigVar(sm+"EDist_Y");
  auto smAphiYld = getConfigVar(sm+"AphiDist_Y");
  auto smAthetaYld = getConfigVar(sm+"AthetaDist_Y");
  auto smEnDistR = getConfigVar(sm+"EDist_R");
  auto smAphiDistR = getConfigVar(sm+"AphiDist_R");
  auto smAthDistR = getConfigVar(sm+"AthetaDist_R");
  std::cout << ptclSrcNcFile << "\n";
  
  std::string surfFlux{"surfaces.flux."};
  auto surfNE = getConfigVar<int>(surfFlux+"nE");
  auto surfE0 = getConfigVar<int>(surfFlux+"E0");
  auto surfE = getConfigVar<int>(surfFlux+"E");
  auto surfNA = getConfigVar<int>(surfFlux+"nA");
  auto surfA0 = getConfigVar<int>(surfFlux+"A0");
  auto surfA = getConfigVar<int>(surfFlux+"A");


}

