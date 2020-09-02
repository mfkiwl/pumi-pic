#ifndef GITRM_PARTICLES_HPP
#define GITRM_PARTICLES_HPP

#include <fstream>
#include <cstdlib>

#include <cuda.h>
#include <curand_kernel.h>

#include <mpi.h>
#include <netcdf>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <particle_structs.hpp>
#include <pumipic_mesh.hpp>

#include "GitrmMesh.hpp"
#include "GitrmBoundary.hpp"

#define DIST2BDRY_TEST 0

using pumipic::fp_t;
using pumipic::Vector3d;
using pumipic::lid_t;
using pumipic::SellCSigma;
using pumipic::MemberTypes;

namespace o = Omega_h;
namespace p = pumipic;

typedef MemberTypes < Vector3d, Vector3d, int, long int, Vector3d, Vector3d,
   int, fp_t, fp_t, int, fp_t, fp_t, int, fp_t> Particle;

// 'Particle' definition retrieval indices.
enum {PTCL_POS, PTCL_NEXT_POS, PTCL_ID,PTCL_ID_GLOBAL, PTCL_VEL, PTCL_EFIELD, PTCL_CHARGE,
 PTCL_WEIGHT, PTCL_FIRST_IONIZEZ, PTCL_PREV_IONIZE, PTCL_FIRST_IONIZET,
 PTCL_PREV_RECOMBINE, PTCL_HIT_NUM, PTCL_VMAG_NEW};

enum { //move it to mesh class
  INVALID = 0,
  PISCES_ID,
  ITER_ID
};

typedef p::ParticleStructure<Particle> PS;

class GitrmParticles {
public:
  GitrmParticles(GitrmMesh* gm, p::Mesh& picparts, long int totalPtcls, int nIter, double dT,
    bool useCudaRnd=false, unsigned long int seed=0, unsigned long int seq=0,
    bool gitrRnd=false);
  ~GitrmParticles();
  GitrmParticles(GitrmParticles const&) = delete;
  void operator=(GitrmParticles const&) = delete;

  PS* ptcls;
  p::Mesh& picparts;
  o::Mesh& mesh;
  GitrmMesh* gm;

  o::Real timeStep = 0;
  long int totalPtcls = 0;
  int numIterations = 0;
  int ptclSplitRead = 0;
  /** Random number setting
   */
  Kokkos::Random_XorShift64_Pool<> rand_pool;
  void initRandGenerator(unsigned long int seed);
  /** Default is Kokkos Rnd if useCudaRnd is false
   */
  bool useCudaRnd;
  curandState *cudaRndStates;
  o::Write<curandState*> cudaRndStates_d;

  void assignParticles(const o::LOs& elemIdOfPtclsAll, o::LOs& numPtclsInElems,
   o::LOs& elemIdOfPtcls, o::LOs& ptclDataInds);

  void defineParticles(const o::LOs& ptclsInElem, int elId=-1);
  void initPtclsOnGeometricBoundarys(const GitrmMesh& gm);

  void findInitialBdryElemIdInADir(o::Real theta, o::Real phi, o::Real r,
    o::LO &initEl, o::Write<o::LO> &elemAndFace,
    o::LO maxLoops=100, o::Real outer=2);

  void setPtclInitRndDistribution(o::Write<o::LO> &);

  void initPtclsInADirection(o::Real theta, o::Real phi, o::Real r,
    o::LO maxLoops = 100, o::Real outer=2);

  void initPtclsOnGeometricIdSurfaces();

  void setInitialTargetCoords(o::Real dTime);

  void initPtclsFromFile(const std::string& fName, o::LO maxLoops=100,
    bool printSource=false);

  void initPtclChargeIoniRecombData();
  void initPtclSurfaceModelData();

  void setPidsOfPtclsLoadedFromFile(const o::LOs& ptclIdPtrsOfElem,
    const o::LOs& ptclIdsInElem, const o::LOs& elemIdOfPtcls, const o::LOs& ptclDataInds);

  void setPtclInitData(const o::Reals& data);

  void convertInitPtclElemIdsToCSR(const o::LOs& numPtclsInElems,
    const o::LOs& ptclIdPtrsOfElem, const o::LOs& elemIds, o::LOs& ptclIdsOfElem);

  int readGITRPtclStepDataNcFile(const std::string& ncFileName, int& maxNPtcls,
    bool debug=false);
  void checkCompatibilityWithGITRflags(int timestep);

  bool searchPtclInAllElems(const o::Reals& data, const o::LO pind, o::LO& parentElem, o::LOs& validPtcls);

  o::LO searchAllPtclsInAllElems(const o::Reals& data, o::Write<o::LO>& elemIdOfPtcls,
    o::Write<o::LO>& numPtclsInElems, o::LOs& validPtcls);

  o::LO searchPtclsByAdjSearchFromParent(const o::Reals& data, const o::LO parentElem,
    o::Write<o::LO>& numPtclsInElemsAll, o::Write<o::LO>& elemIdOfPtclsAll, o::LOs& validPtcls);
  void findElemIdsOfPtclCoordsByAdjSearch(const o::Reals& data, o::LOs& elemIdOfPtcls,
   o::LOs& ptclDataInds, o::LOs& numPtclsInElems, o::LOs& validPtcls);

  void storePtclDataInGridsRZ(o::LO iter, o::Write<o::LO>& data_d,
   int gridsR=1, int gridsZ=10, double radMax=0.2, double zMax=0.8,
   double radMin=0, double zMin=0);

  // particle dist to bdry
  o::Reals closestPoints;
  o::LOs closestBdryFaceIds;

  void initPtclWallCollisionData();
  void resetPtclWallCollisionData();
  void convertPtclWallCollisionData();
  // wall collision; index over ptcl not pids
  o::Reals wallCollisionPts;
  o::LOs wallCollisionFaceIds;
  o::Write<o::Real> wallCollisionPts_w;
  o::Write<o::LO> wallCollisionFaceIds_w;
  void updatePtclHistoryData(int iter, int nT, const o::LOs& elem_ids);
  void initPtclHistoryData(int histInterval);

  o::Write<o::LO> ptclIdsOfHistoryData;
  o::Write<o::Real> ptclHistoryData;
  o::Write<o::LO> lastFilledTimeSteps;
  o::Write<o::Real> ptclEndPoints;
  o::Reals ptcl_created_data; //PARTICLE CREATED DATA
  o::Write<o::Real> ptcl_created_data_w;

  void initPtclEndPoints();
  void writeOutPtclEndPoints(const std::string& file="positions_gitrm.m") const;

  o::Write<o::Real> storedBdryFaces;

  int numInitPtcls = 0;  // ptcls->nPtcls()
  int numPtclsRead = 0;
  int dofHistory = 1;
  int histStep = 0;
  int histInterval = 0;
  int nFilledPtclsInHistory = 0;
  int nThistory = 0;
  void writePtclStepHistoryFile(std::string ncfile, bool debug=false) const;
  void writeDetectedParticles(std::string fname="results.txt",
    std::string header="particles collected") const ;
  void updateParticleDetection(const o::LOs& elem_ids, o::LO iter, bool last=false,
    bool debug=true);
  void initPtclDetectionData(int numGrid=14);
  o::Write<o::LO> collectedPtcls;

  int getCommRank() const { return rank;}
  void setMyCommRank();

  // test GITR step data
  int useGitrRndNums = 0;
  o::Reals testGitrPtclStepData;
  int testGitrDataIoniRandInd = -1;
  int testGitrDataRecRandInd = -1;
  int testGitrStepDataDof = -1;
  int testGitrStepDataNumTsteps = -1;
  int testGitrStepDataNumPtcls = -1;
  int testGitrCollisionRndn1Ind = -1;
  int testGitrCollisionRndn2Ind = -1;
  int testGitrCollisionRndxsiInd = -1;
  int testGitrCrossFieldDiffRndInd = -1;
  int testGitrReflectionRndInd = -1;
  int testGitrOptIoniRec = 0;
  int testGitrOptDiffusion = 0;
  int testGitrOptCollision = 0;
  int testGitrOptSurfaceModel = 0;

  bool ranIonization = false;
  bool ranRecombination = false;
  bool ranCoulombCollision = false;
  bool ranDiffusion = false;
  bool ranSurfaceReflection = false;
  bool isFullCopyMesh() { return isFullMesh;}
  void printNumPtclsInElems(const std::string& f="numsOfPtclsInElems.txt");

private:
  o::LOs elemOwners;
  int rank = -1;
  int commSize =-1;
  bool isFullMesh = false;
  bool mustFindAllPtcls = true;
  int geometryId = 0;
  o::LOs numOfPtclsInElems;

};

//timestep +1
extern int iTimePlusOne;

namespace gitrm {
const o::Real maxCharge = 73; // 74-1
const o::Real ELECTRON_CHARGE = 1.60217662e-19;
const o::Real PROTON_MASS = 1.6737236e-27;
const o::Real PTCL_AMU=184.0; //W,tungsten
const o::LO PARTICLE_Z = 74;
const o::LO ptclHistAllocationStep = 10000;
o::Reals getConstEField();
o::Reals getConstBField();

template<typename T>
void reallocate_data(o::Write<T>& data, o::LO size, T init=0) {
  auto n = data.size();
  auto dataIn = o::Read<T>(data);
  data = o::Write<T>(n+size, init);
  o::parallel_for(n, OMEGA_H_LAMBDA(const int& i) {
    data[i] = dataIn[i];
  },"kernel_reallocate_data");
}

inline int getCommRank() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

inline void printCudaMemInfo() {
  size_t mf, ma;
  printf("executing CUDA meminfo \n");
  cudaMemGetInfo(&mf, &ma); //TODO
  double div = 1024*1024*1024;
  std::cout << "free: " << mf/div << " GB. total: " << ma/div << std::endl;
}

} //ns

/** @brief Calculate distance of particles to domain boundary.
 * Not yet clear if a pre-determined depth can be used
*/
inline void gitrm_findDistanceToBdry(GitrmParticles& gp,
   GitrmMesh& gm, int debug=0) {
  if(debug)
    printf("gitrm_findDistanceToBdry \n");
  int tstep = iTimePlusOne;
  int rank = gp.getCommRank();
  auto* ptcls = gp.ptcls;

  //NOTE: this is only for picpart or if picpart itself is full mesh
  o::Mesh& mesh = gm.mesh;
  const auto f2rPtr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2rElem = mesh.ask_up(o::FACE, o::REGION).ab2b;
  const auto face_verts = mesh.ask_verts_of(2);
  const auto coords = mesh.coords();
  auto* fullMesh = gm.getFullMesh();
  const auto full_face_verts = fullMesh->ask_verts_of(2);
  const auto full_coords = fullMesh->coords();

  const int useReadInCsr = USE_READIN_CSR_BDRYFACES;
  //TODO only works for full mesh. Verify that in these get calls.
  //Don't use picpart mesh coords and face verts with this
  const auto bdryCsrReadInDataPtrs = gm.getBdryCsrReadInPtrs();
  const auto bdryCsrReadInData = gm.getBdryCsrReadInFids();
  const auto bdryFaceOrderedIds = gm.getBdryFaceOrderedIds();

  auto* b = gm.getBdryPtr();
  //if the next flag is false, then assert in the get calls that it is fullMesh
  const int useStoredBdryDataOnPic = !(b->isUsingOrigBdryData());
  const auto bdryFaceVertCoordsPic = b->getBdryFaceVertCoordsPic();
  const auto bdryFaceVertsPic = b->getBdryFaceVertsPic();
  const auto bdryFaceIdsPic = b->getBdryFaceIdsPic();
  const auto bdryFaceIdPtrsPic = b->getBdryFaceIdPtrsPic();
  const auto origGlobalIds = b->getOrigGlobalIds();

  //TODO unify the get method
  //const auto  bdryFaceIds = gm.getBdryFidsCalculated();
  //const auto  bdryFaceIdPtrs = gm.getBdryCsrCalculatedPtrs();

  const auto psCapacity = ptcls->capacity();
  o::Write<o::Real> closestPoints(psCapacity*3, 0, "closest_points");
  o::Write<o::LO> closestBdryFaceIds(psCapacity, -1, "closest_fids");
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto vel_ps = ptcls->get<PTCL_VEL>();

  if(debug > 1)
    MPI_Barrier(MPI_COMM_WORLD);

  if(debug > 1 && tstep==0)
    std::cout << rank << " : useReadInCsr " << useReadInCsr
              << " useStoredBdryDataOnPic " << useStoredBdryDataOnPic << "\n";
  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if (mask > 0) {
      auto ptcl = pid_ps(pid);
      o::LO beg = 0;
      o::LO nFaces = 0;
      o::LO end = 0;

      //TODO merge
      if(useReadInCsr && useStoredBdryDataOnPic) {
        beg = bdryFaceIdPtrsPic[elem];
        end = bdryFaceIdPtrsPic[elem+1];
        nFaces = end - beg;

      } else if(useReadInCsr) {
        beg = bdryCsrReadInDataPtrs[elem];
        end = bdryCsrReadInDataPtrs[elem+1];
        nFaces = end - beg;
      } //else { //TODO calculated bdry data disabled
       // beg = bdryFaceIdPtrsPic[elem];// bdryFacePtrs[elem];
       // nFaces = bdryFaceIdPtrsPic[elem+1] - beg; //bdryFacePtrs[elem+1] - beg;
     // }

      if(!nFaces)
        printf("WARNING %d: ptcl %d tstep %d elem %d gid %ld nFaces %d beg %d end %d\n",
         rank, ptcl, tstep, elem, origGlobalIds[elem], nFaces, beg, end);
      OMEGA_H_CHECK(nFaces);

      double dist = 0;
      double min = 1.0e+30;
      o::LO bfid = -1;
      o::LO fid = -1;
      auto ref = p::makeVector3(pid, pos_ps);
      auto point = o::zero_vector<3>();
      auto pt = o::zero_vector<3>();
      o::Matrix<3,3> face;
      for(o::LO ii = 0; ii < nFaces; ++ii) {
        o::LO ind = beg + ii;
        if(useStoredBdryDataOnPic)
          bfid = bdryFaceIdsPic[ind];
        else if(useReadInCsr)
          bfid = bdryCsrReadInData[ind];

        //else
        //  bfid = bdryFaceIdsPic[ind];// TODO bdryFaces[ind];
        //using from this PICpart
        if(useStoredBdryDataOnPic) {
          //bfid is the index of the storedBdryFaces in this case
          face = p::get_face_coords_of_tet(bdryFaceVertsPic,
            bdryFaceVertCoordsPic, bfid);
        } else { //only picpart
          face = p::get_face_coords_of_tet(face_verts, coords, bfid);
        }

        int region = -1;
        pt = p::closest_point_on_triangle(face, ref, &region);
        dist = o::norm(pt - ref);

        if(debug > 2) {
          o::LO gid = origGlobalIds[elem]; //GO
          auto gbeg = bdryCsrReadInDataPtrs[gid];
          auto gbfid = bdryCsrReadInData[gbeg+ii];
          auto gf = p::get_face_coords_of_tet(full_face_verts, full_coords, gbfid);
          auto gpt = p::closest_point_on_triangle(gf, ref, &region);
          auto gdist = o::norm(gpt - ref);
          auto eq = p::almost_equal(gdist, dist);
          if(!eq)
            printf("%d: ptcl %d elem %d gelem %d nf %d i %d beg %d gebg %d"
             " bfid %d gbfid %d d2bdry %g gd2bdry %g\n",
             rank, ptcl, elem, gid, nFaces, ii, beg, gbeg, bfid, gbfid, dist, gdist);

          auto f = face;

          if(debug > 3)
            printf("%d: p %d e%d gel %d ii %d f %g %g %g %g %g %g %g %g %g "
             "gf %g %g %g %g %g %g %g %g %g\n", rank, ptcl, elem,
              gid, ii, f[0][0],f[0][1],f[0][2],
              f[1][0],f[1][1],f[1][2], f[2][0],f[2][1],f[2][2], gf[0][0],gf[0][1],gf[0][2],
              gf[1][0],gf[1][1],gf[1][2], gf[2][0],gf[2][1],gf[2][2]);

          for(int i=0; i<3; ++i)
            for(int j=0; j<3; ++j) {
              auto fv = f[i][j];
              auto gfv = gf[i][j];
              auto eqf = p::almost_equal(fv, gfv);
              if(!eqf)
                printf("%d: closest-bdry ptcl %d elem %d f %g gf %g\n", rank, ptcl, elem,fv,gfv);
              OMEGA_H_CHECK(eqf);
            }
          OMEGA_H_CHECK(eq);
        }

        if(dist < min) {
          min = dist;
          fid = bfid;
          for(int i=0; i<3; ++i)
            point[i] = pt[i];
        }
        OMEGA_H_CHECK(fid >= 0);
      } //for nFaces

      closestBdryFaceIds[pid] = fid;
      for(o::LO j=0; j<3; ++j)
        closestPoints[pid*3+j] = point[j];

      if(debug >2) {
        o::Matrix<3,3> f;
        if(useStoredBdryDataOnPic)
          f = p::get_face_coords_of_tet(bdryFaceVertsPic,
            bdryFaceVertCoordsPic, fid);
        else  //only picpart
          f = p::get_face_coords_of_tet(face_verts, coords, fid);
        printf("%d: dist2bdry: ptcl %d tstep %d el %d MINdist %.15e nf %d bfid %d "
          "pos %.15e %.15e %.15e face %g %g %g : %g %g %g : %g %g %g\n",
          rank, ptcl, tstep, elem, min, nFaces, fid, ref[0], ref[1], ref[2],
          f[0][0], f[0][1], f[0][2], f[1][0],f[1][1], f[1][2],f[2][0],f[2][1],f[2][2]);
      } //debug

      auto vel = p::makeVector3(pid, vel_ps);
      if(debug > 1 || isnan(ref[0]) || isnan(ref[1]) || isnan(ref[2])
        || isnan(vel[0]) || isnan(vel[1]) || isnan(vel[2])) {
        printf("%d: ptcl %d tstep %d pos %g %g %g vel %g %g %g\n", rank, ptcl, tstep,
          ref[0], ref[1], ref[2], vel[0], vel[1], vel[2]);
      }
    } //mask
  };

  p::parallel_for(ptcls, lambda, "dist2bdry_kernel");
  gp.closestPoints = o::Reals(closestPoints);
  gp.closestBdryFaceIds = o::LOs(closestBdryFaceIds);
}

#endif//define

