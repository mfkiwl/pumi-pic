#ifndef GITRM_MESH_HPP
#define GITRM_MESH_HPP

#include <vector>
#include <cfloat>
#include <set>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <netcdf>

#include "Omega_h_int_scan.hpp"
#include "Omega_h_mesh.hpp"
#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "GitrmInputOutput.hpp"

namespace o = Omega_h;
namespace p = pumipic;

class GitrmBoundary;

//presheath efield is always used ??
// sheath efield is calcualted efield, it is always used. Skip calling
// gitrm_calculateE for neutrals.

// D3D 0.8 to 2.45 m radial

namespace gitrm {
  const double surfaceAndMaterialModelZ = 74;
}

//TODO put in config class, and get from input class

const bool CREATE_GITR_MESH = false;
const int USE_READIN_CSR_BDRYFACES = 1;
const int STORE_BDRYDATA_PIC = 1;
const int USE_ALL_BDRY_FACES_BRUTE_FORCE = 0; //TODO check compatibility with calculation of bdries
const bool MUST_FIND_ALL_PTCLS_INIT = false;
const int PTCLS_SPLIT_READ = 0;

const o::LO D2BDRY_GRIDS_PER_TET =1;// 10; // if csr bdry not re-used
const int D2BDRY_MIN_SELECT = 1;//10; //that many instead of the closest one
const int D2BDRY_MEM_FACTOR = 1;
const bool WRITE_TEXT_D2BDRY_FACES = false;
const bool WRITE_BDRY_FACE_COORDS_NC = false;
const bool WRITE_MESH_FACE_COORDS_NC = false;

const int USE_2DREADIN_IONI_REC_RATES = 1;
const int USE3D_BFIELD = 0;
const int USE2D_INPUTFIELDS = 1;
const int USE_CYL_SYMMETRY = 1;
const int USE_PRESHEATH_EFIELD = 1;

// in GITR only constant EField is used.
const o::Real CONSTANT_EFIELD0 = 0;
const o::Real CONSTANT_EFIELD1 = 0;
const o::Real CONSTANT_EFIELD2 = 0;

const o::Real CONSTANT_BFIELD0 = 0;
const o::Real CONSTANT_BFIELD1 = 0;
const o::Real CONSTANT_BFIELD2 = -0.08;

const o::Real CONSTANT_FLOW_VELOCITY0 = 0;
const o::Real CONSTANT_FLOW_VELOCITY1 = 0;
const o::Real CONSTANT_FLOW_VELOCITY2 = -20000;
// 3 vtx, 1 bdry faceId & 1 bdry elId as Reals.
enum { BDRY_FACE_STORAGE_SIZE_PER_FACE = 1, BDRY_FACE_STORAGE_IDS=0 };
const int BDRY_STORAGE_SIZE_PER_FACE = 1;
// Elements face type
enum {INTERIOR=1, EXPOSED=2};
//if not only sheath bdry faces included in d2bdry calculation
//this is used to skip any model ids. Not valid if only sheath bdry is included
const int SKIP_MODEL_IDS_FROM_DIST2BDRY = 0; //set to 0
const int CALC_EFIELD_USING_D2BDRY = 1; //true unless hpic provides it

#define MESHDATA(mesh) \
  const auto nel = mesh.nelems(); \
  const auto coords = mesh.coords(); \
  const auto mesh2verts = mesh.ask_elem_verts(); \
  const auto dual_elems = mesh.ask_dual().ab2b; \
  const auto dual_faces = mesh.ask_dual().a2ab; \
  const auto face_verts = mesh.ask_verts_of(2); \
  const auto down_r2f = mesh.ask_down(3, 2).ab2b; \
  const auto down_r2fs = mesh.ask_down(3, 2).ab2b; \
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);

class GitrmMesh {
public:
  GitrmMesh(p::Mesh* pp, o::Mesh* full, o::Mesh& mesh, char* owners);
  ~GitrmMesh();

  GitrmMesh(GitrmMesh const&) = delete;
  GitrmMesh& operator =(GitrmMesh const&) = delete;

  void createSurfaceGitrMesh();
  void printBdryFaceIds(bool printIds=true, o::LO minNums=0);
  void printBdryFacesCSR(bool printIds=true, o::LO minNums=0);
  void test_preProcessDistToBdry();

  o::Mesh* fullMesh;
  o::Mesh& mesh;
  p::Mesh* picparts;
  GitrmBoundary* gbdry;

  GitrmBoundary* getBdryPtr() const { return gbdry; }
  p::Mesh* getPicparts() const { return picparts; }
  o::Mesh& getMesh() const { return mesh; }
  o::Mesh* getFullMesh() const { return fullMesh; }
  o::LOs getOwnersAll() const { return ownersAll; }

  /** Distance to bdry search
  */
  void storeOwners(char* owners);

  /** Initialize simulation geometry (pisce, iter etc) and load mesh fields.
   *  This is to be called from where the fields and geometry info available.
   */
  void initGeometryAndFields(const std::string& bFile, const std::string& profFile,
    const std::string& thermGradientFile, const std::string& geomName, int deb=0);

  void initPiscesGeometry();
  void initIterGeometry();

  void preProcessBdryFacesBfs();
  void preprocessStoreBdryFacesBfs(o::Write<o::LO>& numBdryFaceIdsInElems,
    o::Write<o::LO>& bdryFacesCsrW, int csrSize);

  /** if dist2bdry pointers are local element ids of picpart then convert
   * it to original global elements id based
   */
  void convertDist2BdryDataToGlobal(const o::LOs bfPtrs, const o::LOs bFaces,
    o::LOs bdryFacePtrs, o::LOs bdryFaces);
  void writeDist2BdryFacesData(const std::string outFileName="d2bdryFaces.nc",
    int nD2BdryTetSubDiv=0);
  o::LOs getBdryFacesCsrBFS() const {return *bdryFacesCsrBFS;}
  o::LOs getBdryFacePtrsBFS() const {return *bdryFacePtrsBFS;}

  /** Store closest boundary faces in each element in the domain.
   * onlyMaterialSheath skip others consistent with GITR sheath hash creation.
   */
  o::LOs selectBdryFacesOfModelIds(o::LO nFaces, bool onlyMaterialSheath=true);
  void preprocessSelectBdryFacesFromAll(bool onlyMaterialSheath=true);
  void preprocessSelectBdryFacesOnPicpart( bool onlyMaterialSheath=true);

  o::LOs getBdryCsrCalculatedPtrs() { return bdryFacePtrsSelected;}
  o::LOs getBdryFidsCalculated() { return bdryFacesSelectedCsr;}

  void writeBdryFacesDataText(int, std::string fileName="bdryFacesData.txt");
  void writeBdryFaceCoordsNcFile(int mode, std::string fileName="meshFaces.nc");

  int readDist2BdryFacesData(const std::string &);

  o::LOs getBdryCsrReadInPtrs() { return bdryCsrReadInDataPtrs;}
  o::LOs getBdryCsrReadInFids() { return bdryCsrReadInData;}

  void markBdryFacesOnGeomModelIds(const o::LOs& gFaces,
     o::Write<o::LO>& mark_d, o::LO newVal, bool init);

  void setFaceId2BdryFaceIdMap();
  o::LOs getBdryFaceOrderedIds() const {return bdryFaceOrderedIds; }
  int nbdryFaces = 0;

  void setFaceId2SurfaceAndMaterialIdMap();
  int nSurfMaterialFaces = 0;
  o::LOs getSurfaceAndMaterialOrderedIds() const { return surfaceAndMaterialOrderedIds; }

  int nDetectSurfaces = 0;
  o::LOs getDetectorMeshFaceOrderedIds() const {return detectorMeshFaceOrderedIds;}

  void setFaceId2BdryFaceMaterialsZmap();
  o::LOs getBdryFaceMaterialZs() const {return bdryFaceMaterialZs;}

  void initBField(const std::string &f="bFile");
  void load3DFieldOnVtxFromFile(const std::string tagName,
   const std::string &file, Field3StructInput& fs, o::Write<o::Real>& readInData_d);

  bool addTagsAndLoadProfileData(const std::string &, const std::string &,
   const std::string &f="gradfile");
  bool calculateBdryFaceFields(bool init, bool debug=false);
  bool initBdry = false;

  void loadScalarFieldOnBdryFacesFromFile(const std::string, const std::string &,
    Field3StructInput &, int debug=0);

  void loadScalarFieldOnBdryFacesFromFile_(const std::string tagName,
  const std::string &file, Field3StructInput& fs, int debug);

  void load1DFieldOnVtxFromFile(const std::string&, const std::string&,
     Field3StructInput&, o::Write<o::Real>&, o::Write<o::Real>&, int debug=0);

  int markDetectorSurfaces(bool render=false);
  void writeResultAsMeshTag(o::Write<o::LO>& data_d);
  void test_interpolateFields(bool debug=false);
  void printDensityTempProfile(double rmax=0.2, int gridsR=20,
    double zmax=0.5, int gridsZ=10);
  void compareInterpolate2d3d(const o::Reals& data3d, const o::Reals& data2d,
    double x0, double z0, double dx, double dz, int nx, int nz, bool debug=false);


  o::Reals getEfield2d() const {return *Efield_2d;}
  o::Reals getBfield2d() const {return bField_2d;}
  o::Reals getDensIon() const {return *densIon_d;}
  o::Reals getDensEl() const {return *densEl_d;}
  o::Reals getTemIon() const {return *temIon_d;}
  o::Reals getTemEl() const {return *temEl_d;}
  
  o::Reals getIonDens2dGrid(int i) const {
    return (i==1) ? ionDens2dGridX : ionDens2dGridZ;
  }
  o::Reals getElDens2dGrid(int i) const {
    return (i==1) ? elDens2dGridX : elDens2dGridZ;
  }
  o::Reals getIonTemp2dGrid(int i) const {
    return (i==1) ? ionTemp2dGridX : ionTemp2dGridZ;
  }
  o::Reals getElTemp2dGrid(int i) const {
    return (i==1) ? elTemp2dGridX : elTemp2dGridZ;
  }
  o::Reals getIonTempGrad2dGrid(int i) const {
    return (i==1) ? ionTempGrad2dGridX : ionTempGrad2dGridZ;
  }
  o::Reals getBfield2dGrid(int i) const {
    return (i==1) ? bField2dGridX : bField2dGridZ;
  }
  o::Reals getGradTi() const {return *gradTi_d;}
  o::Reals getGradTe() const {return *gradTe_d;}

  o::Reals getDensIonVtx() const{ return *densIonVtx_d;}
  o::Reals getTempIonVtx() const{ return *tempIonVtx_d;}
  o::Reals getDensElVtx() const{ return *densElVtx_d;}
  o::Reals getTempElVtx() const{ return *tempElVtx_d;}
  o::Reals getGradTiVtx() const{ return *gradTi_vtx_d;}
  o::Reals getGradTeVtx() const{ return *gradTe_vtx_d;}

  o::Reals getFlowVel2d() const { return flowVel_d; }
  o::Reals getFlowVel2dGrid(int i) const {
    return (i==1) ? flowVel2dGridX : flowVel2dGridZ;
  }
  o::Real getFlowVelX0() const { return flowVX0; }
  o::Real getFlowVelZ0() const { return flowVZ0; }
  o::LO getFlowVelNx() const { return flowVNx; }
  o::LO getFlowVelNz() const { return flowVNz; }
  o::Real getFlowVelDx() const { return flowVDx; }
  o::Real getFlowVelDz() const { return flowVDz; }

  /** geometric model ids of detector segments */
  o::LOs getDetectorModelIds() const {return detectorModelIds; }
  int getNumDetectorModelIds() const { return detectorModelIds.size(); }
  /** geometric model ids of */
  o::LOs getBdryMaterialModelIds() const { return bdryMaterialModelIds;}
  o::LOs getBdryMaterialModelIdsZ() const { return bdryMaterialModelIdsZ; }
  o::LOs getSurfaceAndMaterialModelIds() const { return surfaceAndMaterialModelIds;}
  int getNumSurfaceAndMaterialModelIds() const { return surfaceAndMaterialModelIds.size(); }
  o::LOs getSurfMatGModelSeqNums() const { return surfMatGModelSeqNums;}

  // Used in boundary init and if 2D field is used for particles
  //TODO move to private, delete unsued
  o::Real bGridX0 = 0;
  o::Real bGridZ0 = 0;
  o::Real bGridDx = 0;
  o::Real bGridDz = 0;
  o::LO bGridNx = 0;
  o::LO bGridNz = 0;
  o::Real eGridX0 = 0;
  o::Real eGridZ0 = 0;
  o::Real eGridDx = 0;
  o::Real eGridDz = 0;
  o::LO eGridNx = 0;
  o::LO eGridNz = 0;


  o::Real densIonX0 = 0;
  o::Real densIonZ0 = 0;
  o::LO densIonNx = 0;
  o::LO densIonNz = 0;
  o::Real densIonDx = 0;
  o::Real densIonDz = 0;
  o::Real tempIonX0 = 0;
  o::Real tempIonZ0 = 0;
  o::LO tempIonNx = 0;
  o::LO tempIonNz = 0;
  o::Real tempIonDx = 0;
  o::Real tempIonDz = 0;

  o::Real densElX0 = 0;
  o::Real densElZ0 = 0;
  o::LO densElNx = 0;
  o::LO densElNz = 0;
  o::Real densElDx = 0;
  o::Real densElDz = 0;
  o::Real tempElX0 = 0;
  o::Real tempElZ0 = 0;
  o::LO tempElNx = 0;
  o::LO tempElNz = 0;
  o::Real tempElDx = 0;
  o::Real tempElDz = 0;

  o::Real gradTiX0 = 0;
  o::Real gradTiZ0 = 0;
  o::Real gradTiNx = 0;
  o::Real gradTiNz = 0;
  o::Real gradTiDx = 0;
  o::Real gradTiDz = 0;

  o::Real gradTeX0 = 0;
  o::Real gradTeZ0 = 0;
  o::Real gradTeNx = 0;
  o::Real gradTeNz = 0;
  o::Real gradTeDx = 0;
  o::Real gradTeDz = 0;

  int numDetectorSurfaceFaces = 0;
  std::string getGeometryName() const { return geomName; }
  bool isUsingConstBField() const { return useConstBField; }
  bool isUsingPreSheathEField() const { return usePreSheathEField; }
  bool isBiasedSurface() const { return biasedSurface; }
  double getBiasPotential() const { return biasPotential; }
  bool isUsingConstFlowVel() const { return useConstFlowVel; }
  bool isUsingAllBdryFaces() const { return useAllBdryFaces; }

  void loadPreSheathEField(const std::string& psFile);
  o::Reals getPreSheathEField() const { return preSheathEField; }
  o::Reals getPreSheathEFieldGridX() const { return preSheathEFieldGridX; }
  o::Reals getPreSheathEFieldGridZ() const { return preSheathEFieldGridZ; }

  double getImpurityAmu() const { return impurityAmu; }
  double getBackgroundAmu() const { return backgroundAmu; }
  int getBackgroundZ() const { return backgroundZ;}
  int getImpurityZ() const { return impurityZ; }


  o::Real getPerpDiffusionCoeft() const { return perpDiffCoeft; }

private:

  o::LOs ownersAll;
  o::LO numNearBdryElems = 0;
  o::LO numAddedBdryFaces = 0;
  std::string geomName{};
  bool useConstBField = false;
  bool usePreSheathEField = false;
  bool biasedSurface = false;
  double biasPotential = 0;
  bool useConstFlowVel = false;
  bool useAllBdryFaces = false;

  std::shared_ptr<o::LOs> bdryFacesCsrBFS;
  std::shared_ptr<o::LOs> bdryFacePtrsBFS;
  o::LOs bdryFacePtrsSelected;
  o::LOs bdryFacesSelectedCsr;
  o::LOs bdryCsrReadInDataPtrs;
  o::LOs bdryCsrReadInData;

  o::LOs bdryFaceOrderedIds;
  o::LOs surfMatGModelSeqNums;
  o::LOs surfaceAndMaterialOrderedIds;
  o::LOs detectorMeshFaceOrderedIds;
  o::LOs bdryFaceMaterialZs;
  std::string profileNcFile = "profile.nc";
  //D3D_major rad =1.6955m; https://github.com/SCOREC/Fusion_Public/blob/master/
  // samples/D-g096333.03337/g096333.03337#L1033
  //TODO shared_ptr introduced to solve error replacing o::Reals by filled Reals
  std::shared_ptr<o::Reals> Efield_2d;
  o::Reals bField_2d;
  std::shared_ptr<o::Reals> densIon_d;
  std::shared_ptr<o::Reals> densEl_d;
  std::shared_ptr<o::Reals> temIon_d;
  std::shared_ptr<o::Reals> temEl_d;
  std::shared_ptr<o::Reals> gradTi_d;
  std::shared_ptr<o::Reals> gradTe_d;
  o::Reals ionDens2dGridX;
  o::Reals ionDens2dGridZ;
  o::Reals elDens2dGridX;
  o::Reals elDens2dGridZ;
  o::Reals ionTemp2dGridX;
  o::Reals ionTemp2dGridZ;
  o::Reals elTemp2dGridX;
  o::Reals elTemp2dGridZ;
  o::Reals ionTempGrad2dGridX;
  o::Reals ionTempGrad2dGridZ;
  o::Reals elTempGrad2dGridX;
  o::Reals elTempGrad2dGridZ;

  // to replace tag
  std::shared_ptr<o::Reals> densIonVtx_d;
  std::shared_ptr<o::Reals> tempIonVtx_d;
  std::shared_ptr<o::Reals> densElVtx_d;
  std::shared_ptr<o::Reals> tempElVtx_d;
  std::shared_ptr<o::Reals> gradTi_vtx_d;
  std::shared_ptr<o::Reals> gradTe_vtx_d;

  //floc velocity
  o::Reals flowVel_d;
  o::Real flowVX0 = 0;
  o::Real flowVZ0 = 0;
  o::LO flowVNx = 0;
  o::LO flowVNz = 0;
  o::Real flowVDx = 0;
  o::Real flowVDz = 0;
  o::Reals flowVel2dGridX;
  o::Reals flowVel2dGridZ;
  o::Reals bField2dGridX;
  o::Reals bField2dGridZ;

  o::Reals preSheathEField;
  o::Reals preSheathEFieldGridX;
  o::Reals preSheathEFieldGridZ;

  //get model Ids by opening mesh/model in Simmodeler
  o::LOs detectorModelIds;
  o::LOs bdryMaterialModelIds;
  o::LOs bdryMaterialModelIdsZ;
  o::LOs surfaceAndMaterialModelIds;

  o::Real backgroundAmu = 0;
  o::LO backgroundZ = 0;
  o::Real impurityAmu = 0;
  o::LO impurityZ = 0;
  o::Real perpDiffCoeft = 0;

  int rank = -1;
  bool exists = false;
};


namespace utils {

inline o::Write<o::LO> makeCsrPtrs(o::Write<o::LO>& nums_d, int tot, int& sum) {
  auto check = (tot == nums_d.size());
  OMEGA_H_CHECK(check);
  sum = o::get_sum(o::LOs(nums_d));
  return o::deep_copy(o::offset_scan(o::LOs(nums_d)));
}

template<typename T = o::LO>
o::Read<T> createBdryFaceClassArray(o::Mesh& mesh, const o::LOs& gFaceIds,
   const o::Read<T>& gFaceValues, T initVal, const std::string& name="",
   bool allBdry = false, bool replace = false) {
  auto nf = mesh.nfaces();
  o::Write<T> bdryArray(nf, initVal, name);
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");
  auto size = gFaceIds.size();
  o::parallel_for(nf, OMEGA_H_LAMBDA(const T& fid) {
    if(side_is_exposed[fid]) {
      for(auto id=0; id < size; ++id) {
        if(gFaceIds[id] == faceClassIds[fid]) {
          auto v = (replace) ? gFaceValues[id] : 1;
          bdryArray[fid] = v;
        }
      }
      bdryArray[fid] = (allBdry) ? 1 : bdryArray[fid];
    }
  },"createBdryFaceClassArray");
  return o::Read<T>(bdryArray);
}

//create ordered ids starting 0 for given geometric bdry ids.
//copy associated values if fill is true.
template<typename T = o::LO>
o::Read<T> createOrderedBdryIds(o::Mesh& mesh, const o::LOs& gFaceIds, int& nSurfaces,
    const o::LOs& gFaceValues, bool allBdry, bool replace) {
  auto nf = mesh.nfaces();
  o::LO initVal = 0;
  auto isSurface_d = utils::createBdryFaceClassArray(mesh, gFaceIds, gFaceValues,
    initVal, "isSelectBdryFace", allBdry, replace);
  nSurfaces = o::get_sum(o::Read<T>(isSurface_d));
  auto scanIds_r = o::offset_scan(o::Read<T>(isSurface_d));
  auto scanIds_w = o::deep_copy(scanIds_r);
  //set non-surface, ie, repeating numbers, to -1
  o::parallel_for(nf, OMEGA_H_LAMBDA(const T& fid) {
    if(!isSurface_d[fid])
      scanIds_w[fid]= -1;
  },"createOrderedBdryIds");
  //last index of scanned is cumulative sum
  OMEGA_H_CHECK(nSurfaces == (o::HostRead<T>(scanIds_r))[nf-1]);
  return o::Read<T>(scanIds_w);
}

//map from global(over full-mesh) to local. Output has size of global.
//For large mesh, arrays of size of full mesh entities may be problem
//Pass value of invalid entries of the return array if != -1
template<typename T>
o::LOs makeLocalIdMap(const o::Read<T>& data, T val, int& nums,
   std::string name="", T invalid=-1) {
  OMEGA_H_CHECK(val != invalid);
  auto n = data.size();
  auto flags = o::LOs(data); //TODO fix LO vs T
  o::Write<o::LO> localIdFlags(n, 0, "localIds");
  auto setSelect = OMEGA_H_LAMBDA(const T& e) {
    if(data[e] == val)
      localIdFlags[e] = 1;
  };
  //to pass -1, change the default
  o::parallel_for(n, setSelect, "setLocalFlags");
  flags = o::LOs(localIdFlags);

  if(name.empty())
    name = "localCoreIds";
  auto scan_r = o::offset_scan(flags, name);
  o::Write<o::LO> localIds(o::deep_copy(scan_r));
  nums = o::HostWrite<o::LO>(localIds)[n]; // T?
  auto fixLocal = OMEGA_H_LAMBDA(const T& e) {
    if(data[e] != val)
      localIds[e] = invalid;
  };
  o::parallel_for(n, fixLocal, "fixLocalIds");
  return localIds;
}

//temporary
inline o::Reals getConstEField() {
  o::HostWrite<o::Real> ef(3);
  ef[0] = CONSTANT_EFIELD0;
  ef[1] = CONSTANT_EFIELD1;
  ef[2] = CONSTANT_EFIELD2;
  return o::Reals(o::Write<o::Real>(ef));
}

inline o::Reals getConstBField() {
  o::HostWrite<o::Real> bf(3);
  bf[0] = CONSTANT_BFIELD0;
  bf[1] = CONSTANT_BFIELD1;
  bf[2] = CONSTANT_BFIELD2;
  return o::Reals(o::Write<o::Real>(bf));
}

}//ns


namespace gitrm {
/** @brief Function to mark (to newVal) bdry faces on the classified model ids.
 * Also if init is true, the passed in array is initialized to 1 if the
 * corresponding element is on the classified model ids. If newVal=1 then
 * the default is expected to be 0 and init should be false, ie, only
 * matching entries should be 1. Otherwise the initialized values won't be
 * different from that of matching class ids.
*/
inline void markBdryFacesOnGeomModelIds(o::Mesh& mesh, const o::LOs& gFaces,
   o::Write<o::LO>& mark_d, o::LO newVal, bool init) {
  if(init && newVal)
    std::cout << "****WARNING markBdryFacesOnGeomModelIds init = newVal\n";
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);
  auto faceClassIds = mesh.get_array<o::ClassId>(2, "class_id");
  auto nIds = gFaces.size();
  auto lambda = OMEGA_H_LAMBDA(const o::LO& fid) {
    auto val = mark_d[fid];
    if(init && side_is_exposed[fid])
      val = 1;
    for(o::LO id=0; id < nIds; ++id)
      if(gFaces[id] == faceClassIds[fid]) {
        val = newVal;
      }
    mark_d[fid] = val;
  };
  o::parallel_for(mesh.nfaces(), lambda, "MarkFaces");
}
}

#endif// define
