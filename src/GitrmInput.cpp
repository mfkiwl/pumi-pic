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

bool readConfigNumsAndGrids3(const std::string& head, std::string& fname, 
  std::string& nr, std::string& nz, std::string& gr, std::string& gz,
  std::string& ny, std::string& gy) {
  readConfigNumsAndGrids2(head, fname, nr, nz, gr, gz);
  ny = readConfigVar(head+"gridNyString");
  gy = readConfigVar(head+"gridYString");
}

bool readConfigNumsAndGrids2(const std::string& head, std::string& fname,
  std::string& nr, std::string& nz, std::string& gr, std::string& gz) {
  fname = readConfigVar(head+"fileString";
  nr = readConfigVar(head+"gridNrString");
  nz = readConfigVar(head+"gridNzString");
  gr = readConfigVar(head+"gridRString");
  gz = readConfigVar(head+"gridZString");
}

void GitrmInput::testInputConfig() {
  //default auto are string's
  auto geomFile = readConfigVar("geometry.fileString");
  std::string ptclSrcTag{"particleSource"};
  auto ptclSrcCfg = readConfigVar(ptclSrcTag+".fileString");
  auto ptclSrcNcFile = readConfigVar(ptclSrcTag+".ncFileString");
  std::string sm{"surfaceModel."};
  auto smFile = readConfigVar(sm+"fileString");
  auto smNEnSput = readConfigVar(sm+"nEsputtRefCoeffString");
  auto smNAngSput = readConfigVar(sm+"nAsputtRefCoeffString");
  auto smEnSputIn = readConfigVar(sm+"nEsputtRefDistInString");
  auto smAngSputIn = readConfigVar(sm+"nAsputtRefDistInString");
  auto smNEnSputRef = readConfigVar(sm+"nEsputtRefDistOutString");
  auto smNEnSputRefOut = readConfigVar(sm+"nEsputtRefDistOutStringRef");
  auto smNAngSputOut = readConfigVar(sm+"nAsputtRefDistOutString");
  auto smEnSputRef = readConfigVar(sm+"E_sputtRefCoeff");
  auto smAngSputRef = readConfigVar(sm+"A_sputtRefCoeff");
  auto smEnSputRefIn = readConfigVar(sm+"E_sputtRefDistIn");
  auto smAngSputRefIn = readConfigVar(sm+"A_sputtRefDistIn");
  auto smEnSputRefOut = readConfigVar(sm+"E_sputtRefDistOut");
  auto smEnSputRefOutRef = readConfigVar(sm+"E_sputtRefDistOutRef");
  auto smAphiSputRefOut = readConfigVar(sm+"Aphi_sputtRefDistOut");
  auto smAthSputRefOut = readConfigVar(sm+"Atheta_sputtRefDistOut");
  auto smSputYld = readConfigVar(sm+"sputtYldString");
  auto smReflYld = readConfigVar(sm+"reflYldString");
  auto smEnYld = readConfigVar(sm+"EDist_Y");
  auto smAphiYld = readConfigVar(sm+"AphiDist_Y");
  auto smAthetaYld = readConfigVar(sm+"AthetaDist_Y");
  auto smEnDistR = readConfigVar(sm+"EDist_R");
  auto smAphiDistR = readConfigVar(sm+"AphiDist_R");
  auto smAthDistR = readConfigVar(sm+"AthetaDist_R");
  std::cout << ptclSrcNcFile << "\n";
  
  std::string surfFlux{"surfaces.flux."};
  auto surfNE = readConfigVar<int>(surfFlux+"nE");
  auto surfE0 = readConfigVar<double>(surfFlux+"E0");
  auto surfE = readConfigVar<double>(surfFlux+"E");
  auto surfNA = readConfigVar<int>(surfFlux+"nA");
  auto surfA0 = readConfigVar<double>(surfFlux+"A0");
  auto surfA = readConfigVar<double>(surfFlux+"A");
 
  std::string prof{"backgroundPlasmaProfiles."};
  auto profZ = readConfigVar<double>(prof+"Z");
  auto profAmu = readConfigVar<double>(prof+"amu");
  auto profPot = readConfigVar<double>(prof+"biasPotential");

  std::string profB{prof+"Bfield."};
  auto profBr = readConfigVar<double>(profB+"r");
  auto profBy = readConfigVar<double>(profB+"y");
  auto profBz = readConfigVar<double>(profB+"z");
  std::string profBfile{};
  std::string profBNrStr{};
  std::string profBNyStr{};
  std::string profBNzStr{};
  std::string profBGridRStr{};
  std::string profBGridYStr{};
  std::string profBGridZStr{};
  readConfigNumsAndGrids3(profB, profBfile, profBNrStr, profBNzStr,
    profBGridRStr, profBGridZStr, profBNyStr, profBGridYStr);
  auto profBRStr = readConfigVar(profB+"rString");
  auto profBZStr = readConfigVar(profB+"zString");
  auto profBYStr = readConfigVar(profB+"yString");
  
  std::string profE{prof+"Efield."};
  auto profErStr = readConfigVar<double>(profE+"Er"); 
  auto profEzStr = readConfigVar<double>(profE+"Ez");
  auto profEtStr = readConfigVar<double>(profE+"Et");
  std::string profEfile{};
  std::string profENrStr{};
  std::string profENyStr{};
  std::string profENzStr{};
  std::string profEGridRStr{};
  readConfigNumsAndGrids2(profE, profEfile, profENrStr, profENzStr,
    profEGridRStr, profEGridZStr);
  auto profERadStr  = readConfigVar(profE+"radialComponentString");
  auto profEAxialStr = readConfigVar(profE+"axialComponentString");
  auto profETorStr = readConfigVar(profE+"toroidalComponentString");

  std::string profDtsE{prof+"dtsEfield."}; 
  auto profDtsErStr = readConfigVar<double>(profDtsE+"dtsEr");
  auto profDtsEzStr = readConfigVar<double>(profDtsE+"dtsEz");
  auto profDtsEtStr = readConfigVar<double>(profDtsE+"dtsEt");
  auto profDtsSheathStr = readConfigVar(profDtsE+"sheathDTS");
  std::string profDtsfile{};
  std::string profDtsNrStr{};
  std::string profDtsNzStr{};
  std::string profDtsGridRStr{};
  std::string profDtsGridZStr{};
  readConfigNumsAndGrids2(profDtsE, profDtsfile, profDtsNrStr,
    profDtsNzStr, profDtsGridRStr, profDtsGridZStr);

  std::string profTemp{prof+"Temperature."};
  auto profTempTiStr = readConfigVar<double>(profTemp+"ti");
  auto profTempTeStr = readConfigVar<double>(profTemp+"te");
  auto profTempIonStr = readConfigVar(profTemp+"IonTempString");
  auto profTempElStr = readConfigVar(profTemp+"ElectronTempString");
  std::string profTempFile{};
  std::string profTempNrStr{};
  std::string profTempNzStr{};
  std::string profTempGridRStr{};
  std::string profTempGridZStr{};
  readConfigNumsAndGrids2(profTemp, profTempFile, profTempNrStr,
    profTempNzStr, profTempGridRStr, profTempGridZStr);

  std::string profDen{prof+"Density."};
  auto profDenTiStr = readConfigVar<double>(profDen+"ni");
  auto profDenTeStr = readConfigVar<double>(profDen+"ne");
  auto profDenIonStr = readConfigVar(profDen+"IonDensityString");
  auto profDenElStr = readConfigVar(profDen+"ElectronDensityString");
  std::string profDenFile{};
  std::string profDenNrStr{};
  std::string profDenNzStr{};
  std::string profDenGridRStr{};
  std::string profDenGridZStr{};
  readConfigNumsAndGrids2(profDen,profDenFile, profDenNrStr,
     profDenNzStr, profDenGridRStr, profDenGridZStr);

  std::string profDiff{prof+"Diffusion."};
  auto profDiffPerpStr = readConfigVar<double>(profDiff+"Dperp");
  auto profDiffVarStr = readConfigVar(profDiff+"variableString");
  std::string profDiffFile{};
  std::string profDiffNrStr{};
  std::string profDiffNzStr{};
  std::string profDiffGridRStr{};
  std::string profDiffGridZStr{};
  readConfigNumsAndGrids2(profDiff, profDiffFile, profDiffNrStr,
     profDiffNzStr, profDiffGridRStr, profDiffGridZStr);

  std::string profFV{prof+"FlowVelocity."};
  auto profFlowVIntNum = readConfigVar<int>(profFV+"interpolatorNumber");
  auto profFlowVr = readConfigVar<double>(profFV+"flowVr");
  auto profFlowVy = readConfigVar<double>(profFV+"flowVy");
  auto profFlowVz = readConfigVar<double>(profFV+"flowVz");
  std::string profFlowVFile{};
  std::string profFlowVNrStr{};
  std::string profFlowVNzStr{};
  std::string profFlowVGridRStr{};
  std::string profFlowVGridZStr{};
  readConfigNumsAndGrids2(profFV, profFlowVFile, profFlowVNrStr,
     profFlowVNzStr, profFlowVGridRStr, profFlowVGridZStr);
  auto profFlowVrStr = readConfigVar(profFV+"flowVrString");

  std::string profCL{prof+"ConnectionLength."};
  auto profCLIntNum = readConfigVar<int>(profCL+"interpolatorNumber");
  auto profCLenLc = readConfigVar<double>(profCL+"Lc");
  auto profCLenS = readConfigVar<double>(profCL+"s");
  std::string profCLenFile{};
  std::string profCLenNrStr{};
  std::string profCLenNzStr{};
  std::string profCLenGridRStr{};
  std::string profCLenGridZStr{};
  std::string profCLenNyStr{};
  std::string profCLenGridYStr{};
  readConfigNumsAndGrids3(profCL, profCLenFile, profCLenNrStr, profCLenNzStr,
    profCLenGridRStr, profCLenGridZStr, profCLenNyStr, profCLenGridYStr);
  auto profCLenLcStr = readConfigVar(profCL+"LcString");
  auto profCLenSStr = readConfigVar(profCL+"SString");
  auto profCLenFVStr = readConfigVar(profCL+"flowVtString");

  std::string profTGr{prof+"gradT."};
  auto profTGrTeR = readConfigVar<double>(profTGr+"gradTeR");
  auto profTGrTeY = readConfigVar<double>(profTGr+"gradTeY");
  auto profTGrTeZ = readConfigVar<double>(profTGr+"gradTeZ");
  auto profTGrTiR = readConfigVar<double>(profTGr+"gradTiR");
  auto profTGrTiY = readConfigVar<double>(profTGr+"gradTiY");
  auto profTGrTiZ = readConfigVar<double>(profTGr+"gradTiZ");
  std::string profTGrFile{};
  std::string profTGrNrStr{};
  std::string profTGrNzStr{};
  std::string profTGrGridRStr{};
  std::string profTGrGridZStr{};
  readConfigNumsAndGrids2(profTGr, profTGrFile, profTGrNrStr,
    profTGrNzStr, profTGrGridRStr, profTGrGridZStr);
  auto profTGrTeRStr = readConfigVar(profTGr+"gradTeRString");
  auto profTGrTeYStr = readConfigVar(profTGr+"gradTeYString");
  auto profTGrTeZStr = readConfigVar(profTGr+"gradTeZString");
  auto profTGrTiRStr = readConfigVar(profTGr+"gradTiRString");
  auto profTGrTiYStr = readConfigVar(profTGr+"gradTiYString");
  auto profTGrTiZStr = readConfigVar(profTGr+"gradTiZString");
  
  std::string profLc{prof+"Lc."};
  auto profLcVal = readConfigVar<double>(profLc+"value");
  std::string profLcFile{};
  std::string profLcNrStr{};
  std::string profLcNzStr{};
  std::string profLcGridRStr{};
  std::string profLsGridZStr{};
  readConfigNumsAndGrids2(profLc, profLcFile, profLcNrStr,
     profLcNzStr, profLcGridRStr, profLsGridZStr);
  auto profLcVarStr = readConfigVar<double>(profLc+"variableString");
 
  std::string profS{prof+"s."};
  auto profSVal = readConfigVar<double>(profS+"value");
  std::string profSFile{};
  std::string profSNrStr{};
  std::string profSNzStr{};
  std::string profSGridRStr{};
  std::string profSGridZStr{};
  readConfigNumsAndGrids2(profS, profSFile, profSNrStr,
     profSNzStr, profSGridRStr, profSGridZStr);
  auto profLcVarStr = readConfigVar<double>(profLc+"variableString");
  
  std::string conLen{"connectionLength."};
  auto conLenNTrSteps = readConfigVar<int>(conLen+"nTraceSteps");
  auto conLenDr = readConfigVar<double>(conLen+"dr");
  auto conLenNetX0 = readConfigVar<double>(conLen+"netx0");
  auto conLenNetX1 = readConfigVar<double>(conLen+"netx1");
  auto conLenNX = readConfigVar<int>(conLen+"nX");
  auto conLenNetY0 = readConfigVar<double>(conLen+"nety0");
  auto conLenNetY1 = readConfigVar<double>(conLen+"nety1");
  auto conLenNY = readConfigVar<int>(conLen+"nY");
  auto conLenNetZ0 = readConfigVar<double>(conLen+"netz0");
  auto conLenNetZ1 = readConfigVar<double>(conLen+"netz1");
  auto conLenNZ = readConfigVar<int>(conLen+"nZ");
  std::string conLenFile{};
  std::string conLenNrStr{};
  std::string conLenNzStr{};
  std::string conLenGridRStr{};
  std::string conLenGridZStr{};
  std::string conLenNyStr{};
  std::string conLenGridYStr{};
  readConfigNumsAndGrids3(conLen, conLenFile, conLenNrStr, conLenNzStr,
    conLenGridRStr, conLenGridZStr, conLenNyStr, conLenGridYStr);
  auto conLenLcStr = readConfigVar(conLen+"LcString");
  auto conLenSStr = readConfigVar(conLen+"SString");
  auto conLenFVStr = readConfigVar(conLen+"noIntersectionString");

  std::string ptclSrc{"impurityParticleSource.");
  auto ptclSrcNP = readConfigVar<int>(ptclSrc+"nP");
  auto ptclSrcStrength = readConfigVar<double>(ptclSrc+"sourceStrength");
  auto ptclSrcZ = readConfigVar<double>(ptclSrc+"Z");
  auto ptclSrcMatZ = readConfigVar<int>(ptclSrc+"source_material_Z");
  auto ptclSrcMatSurfBindEn = readConfigVar<double>(ptclSrc+
      "source_material_SurfaceBindingEnergy");
  auto ptclSrcMatAlpha = readConfigVar<int>(ptclSrc+"source_materialAlpha");
  std::string ptclSrcInit{ptclSrc+"initialConditions."};
  auto ptclSrcInitXstart = readConfigVar<double>(ptclSrcInit+"x_start");
  auto ptclSrcInitYstart = readConfigVar<double>(ptclSrcInit+"y_start");
  auto ptclSrcInitZstart = readConfigVar<double>(ptclSrcInit+"z_start");
  auto ptclSrcInitEvXstart = readConfigVar<double>(ptclSrcInit+"energy_eV_x_start");
  auto ptclSrcInitEvYstart = readConfigVar<double>(ptclSrcInit+"energy_eV_y_start");
  auto ptclSrcInitEvZstart = readConfigVar<double>(ptclSrcInit+"energy_eV_z_start");
  auto ptclSrcInitAmu = readConfigVar<double>(ptclSrcInit+"impurity_amu");
  auto ptclSrcInitZ = readConfigVar<double>(ptclSrcInit+"impurity_Z");
  auto ptclSrcInitQ = readConfigVar<double>(ptclSrcInit+"charge");
  auto ptclSrcInitEV = readConfigVar<double>(ptclSrcInit+"energy_eV");
  auto ptclSrcInitPhi = readConfigVar<double>(ptclSrcInit+"phi");
  auto ptclSrcInitTheta = readConfigVar<double>(ptclSrcInit+"theta");
  
  std::string ptclSrcIoni{ptclSrc+"ionization."};
  auto ptclSrcIoniFile = readConfigVar(ptclSrcIoni+"fileString");
  auto ptclSrcIoniTGridStr = readConfigVar(ptclSrcIoni+"TempGridString");
  auto ptclSrcIoniDenGridStr = readConfigVar(ptclSrcIoni+"DensGridString");
  auto ptclSrcIoniNQStr = readConfigVar(ptclSrcIoni+"nChargeStateString");
  auto ptclSrcIoniTGridVarStr = readConfigVar(ptclSrcIoni+"TempGridVarName");
  auto ptclSrcIoniDenGridVarStr = readConfigVar(ptclSrcIoni+"DensGridVarName");
  auto ptclSrcIoniCoeffVarStr = readConfigVar(ptclSrcIoni+"CoeffVarName");

  std::string ptclSrcRec{ptclSrc+"recombination."};
  auto ptclSrcRecFile = readConfigVar(ptclSrcRec+"fileString");
  auto ptclSrcRecTGridStr = readConfigVar(ptclSrcRec+"TempGridString");
  auto ptclSrcRecDenGridStr = readConfigVar(ptclSrcRec+"DensGridString");
  auto ptclSrcRecNQStr = readConfigVar(ptclSrcRec+"nChargeStateString");
  auto ptclSrcRecTGridVarStr = readConfigVar(ptclSrcRec+"TempGridVarName");
  auto ptclSrcRecDenGridVarStr = readConfigVar(ptclSrcRec+"DensGridVarName");
  auto ptclSrcRecCoeffVarStr = readConfigVar(ptclSrcRec+"CoeffVarName");

  std::string tStep{"timeStep."};
  auto tStepDt = readConfigVar<double>(tStep+"dt");
  double tStepNPtsPerGyroOrbit = readConfigVar<double>(tStep+"nPtsPerGyroOrbit");
  auto tStepIoniNDt = readConfigVar<int>(tStep+"ionization_nDtPerApply");
  auto tStepCollNDt = readConfigVar<int>(tStep+"collision_nDtPerApply");
  auto tStepNT = readConfigVar<int>(tStep+"nT");

  std::string fEval{"forceEvaluation."};
  auto fEvalX0 = readConfigVar<double>(fEval+"X0");
  auto fEvalX1 = readConfigVar<double>(fEval+"X1");
  auto fEvalY0 = readConfigVar<double>(fEval+"Y0");
  auto fEvalY1 = readConfigVar<double>(fEval+"Y1");
  auto fEvalZ0 = readConfigVar<double>(fEval+"Z0");
  auto fEvalZ1 = readConfigVar<double>(fEval+"Z1");
  auto fEvalNR = readConfigVar<int>(fEval+"nR");
  auto fEvalNY = readConfigVar<int>(fEval+"nY");
  auto fEvalNZ = readConfigVar<int>(fEval+"nZ");
  auto fEvalPtclEn = readConfigVar<double>(fEval+"particleEnergy");

  std::string diag{"diagnostics."};
  auto diagSubSampleFac = readConfigVar<int>(diag+"trackSubSampleFactor");

  std::string flags{"flags."};
  auto optIoni = readConfigVar<int>(flags+"USEIONIZATION");
  auto optRecomb = readConfigVar<int>(flags+"USERECOMBINATION");
  auto optPerpDiff = readConfigVar<int>(flags+"USEPERPDIFFUSION");
  auto optCoulomb = readConfigVar<int>(flags+"USECOULOMBCOLLISIONS");
  auto optFriction = readConfigVar<int>(flags+"USEFRICTION");
  auto optAngScatt = readConfigVar<int>(flags+"USEANGLESCATTERING");
  auto optHeat = readConfigVar<int>(flags+"USEHEATING");
  auto optThermal = readConfigVar<int>(flags+"USETHERMALFORCE");
  auto optSurf = readConfigVar<int>(flags+"USESURFACEMODEL");
  auto optSheathFld = readConfigVar<int>(flags+"USESHEATHEFIELD");
  auto optBias = readConfigVar<int>(flags+"BIASED_SURFACE");
  auto optPreSheathFld = readConfigVar<int>(flags+"USEPRESHEATHEFIELD");
  auto optBInt = readConfigVar<int>(flags+"BFIELD_INTERP");
  auto optLCInt = readConfigVar<int>(flags+"LC_INTERP");
  auto optGenLC = readConfigVar<int>(flags+"GENERATE_LC");
  auto optEInt = readConfigVar<int>(flags+"EFIELD_INTERP");
  auto optPreSheathInt = readConfigVar<int>(flags+"PRESHEATH_INTERP");
  auto optDenInt = readConfigVar<int>(flags+"DENSITY_INTERP");
  auto optTempInt = readConfigVar<int>(flags+"TEMP_INTERP");
  auto optFlowVInt = readConfigVar<int>(flags+"FLOWV_INTERP");
  auto optGradTInt = readConfigVar<int>(flags+"GRADT_INTERP");
  auto optOdeInt = readConfigVar<int>(flags+"ODEINT");
  auto optSeed = readConfigVar<int>(flags+"FIXEDSEEDS");
  auto optPtclSeed = readConfigVar<int>(flags+"PARTICLESEEDS");
  auto optGeomTrace = readConfigVar<int>(flags+"GEOM_TRACE");
  auto optHist = readConfigVar<int>(flags+"PARTICLE_TRACKS");
  auto optPtclSrcSpace = readConfigVar<int>(flags+"PARTICLE_SOURCE_SPACE");
  auto optPtclSrcEn = readConfigVar<int>(flags+"PARTICLE_SOURCE_ENERGY");
  auto optPtclSrcAng = readConfigVar<int>(flags+"PARTICLE_SOURCE_ANGLE");
  auto optPtclSrcFile = readConfigVar<int>(flags+"PARTICLE_SOURCE_FILE");
  auto optSpect = readConfigVar<int>(flags+"SPECTROSCOPY");
  auto optCylSymm = readConfigVar<int>(flags+"USECYLSYMM");
  auto optFldAligned = readConfigVar<int>(flags+"USEFIELDALIGNEDVALUES");
  auto optForceEval = readConfigVar<int>(flags+"FORCE_EVAL");
}



