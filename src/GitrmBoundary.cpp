#include <iostream>

#include "GitrmBoundary.hpp"


GitrmBoundary::GitrmBoundary(GitrmMesh* gm, FaceFieldFillTarget t):
   gm(gm), target(t) {
  picparts = gm->getPicparts();
  mesh = picparts->mesh();
  fullMesh = gm->getFullMesh();
  rank = picparts->comm()->rank();
  std::cout << "Initializing GitrmBoundary\n";
 
  initBoundaries();
}

void GitrmBoundary::initBoundaries() {
  //TODO retrieve settings from GitrmInput class to replace global consts
  isbiasedSurface = (BIASED_SURFACE) ? true : false;
  biasPotential = BIAS_POTENTIAL;
  // calcEfieldUsingD2Bdry = CALC_EFIELD_USING_D2BDRY;

  //set flag for enabling tag fields
  if(target == PICPART) {
    useBdryTagFields= true;
    useStoredBdryData = false;
  } else if(target == BDRY) {
    useBdryTagFields = false;
    useStoredBdryData = true;
  } else {
    Omega_h_fail("Error : Dist-to-bdry target not known\n");
  }
}

//to be called from gm, after calculating el, ion fields
void GitrmBoundary::storeBdryData() {
  if(useStoredBdryData) {
    if(!stored)
      stored = storeBdrySurfaceMeshForPicpart();
    storeBdryFaceFieldsForPicpart(stored);
  }
}

//to be called from gm, if tags added on mesh, after calculating el, ion fields
void GitrmBoundary::calculateBdryData(GitrmMeshFaceFields& ff) {
  if(useStoredBdryData)
    Omega_h_fail("Error: In dist-to-bdry data storage\n");
  auto stored = calculateBdryFaceFields(ff); 
}

//TODO use compile time option to use tag. Delete tags if tag not used
//TODO get the tag names from GitrmFields class
o::Reals GitrmBoundary::getBxSurfNormAngle() const {
  if(useStoredBdryData)
    return angleBdryBfield;
  OMEGA_H_CHECK(useBdryTagFields);
  return mesh->get_array<o::Real>(o::FACE, "angleBdryBfield");
}

o::Reals GitrmBoundary::getPotential() const {
  if(useStoredBdryData)
    return potential;
  OMEGA_H_CHECK(useBdryTagFields);
  return mesh->get_array<o::Real>(o::FACE, "potential");
}

o::Reals GitrmBoundary::getDebyeLength() const {
  if(useStoredBdryData)
    return debyeLength;
  OMEGA_H_CHECK(useBdryTagFields);
  return mesh->get_array<o::Real>(o::FACE, "DebyeLength");
}

o::Reals GitrmBoundary::getLarmorRadius() const {
  if(useStoredBdryData)
    return larmorRadius;
  OMEGA_H_CHECK(useBdryTagFields);
  return mesh->get_array<o::Real>(o::FACE, "LarmorRadius");
}

o::Reals GitrmBoundary::getChildLangmuirDist() const {
  if(useStoredBdryData)
    return childLangmuirDist;
  OMEGA_H_CHECK(useBdryTagFields);
  return mesh->get_array<o::Real>(o::FACE, "ChildLangmuirDist");
}

o::Reals GitrmBoundary::getElDensity() const {
  if(useStoredBdryData)
    return elDensity;
  OMEGA_H_CHECK(useBdryTagFields);
  return mesh->get_array<o::Real>(o::FACE, "ElDensity");
}

o::Reals GitrmBoundary::getIonDensity() const {
  if(useStoredBdryData)
    return ionDensity;
  OMEGA_H_CHECK(useBdryTagFields);
  return mesh->get_array<o::Real>(o::FACE, "IonDensity");
}

o::Reals GitrmBoundary::getElTemp() const {
  if(useStoredBdryData)
    return elTemp;
  OMEGA_H_CHECK(useBdryTagFields);
  return mesh->get_array<o::Real>(o::FACE, "ElTemp");
}

o::Reals GitrmBoundary::getIonTemp() const {
  if(useStoredBdryData)
    return ionTemp;
  OMEGA_H_CHECK(useBdryTagFields);
  return mesh->get_array<o::Real>(o::FACE, "IonTemp");
}


void GitrmBoundary::storeBdryFaceFieldsForPicpart(bool pre, int debug) {
  if(!pre)
    Omega_h_fail("Required Bdry Mesh should be stored prior to %s\n", __FUNCTION__);
  //used similar to that from GitrmMesh
  GitrmMeshFaceFields ff;
  calculateBdryFaceFields(ff);
  angleBdryBfield = ff.angle;
  potential = ff.potential;
  debyeLength = ff.debyeLength;
  larmorRadius = ff.larmorRadius;
  childLangmuirDist = ff.childLangmuirDist;
  bdryFlux = ff.bdryFlux;
  impact = ff.impact;
}


//TODO replace hard-coded constants to use the common values
bool GitrmBoundary::calculateBdryFaceFields(GitrmMeshFaceFields& ff, int debug) {
  std::cout << __FUNCTION__ << " bdry\n";
  o::LO nf = (useBdryTagFields) ? mesh->nfaces() : o::LOs(*bdryFaceVertsPic).size()/3;
  o::LOs face_verts = (useBdryTagFields) ? mesh->ask_verts_of(2): o::LOs(*bdryFaceVertsPic);
  o::Read<o::I8> exposed = (useBdryTagFields) ? mark_exposed_sides(mesh) :
                           o::Read<o::I8>(o::Write<o::I8>(nf, 1, "dummyExposed"));
  o::Reals coords = (useBdryTagFields) ? mesh->coords() : o::Reals(*bdryFaceVertCoordsPic);
  //tag added only if the fields are added on the picpart mesh,
  //not when separate data is stored in bdry class.

  //if addTag=true, make sure density,temp tags are added prior to this
  //The addTag flag sanity check already done in the selectMeshData(..)
  auto density = getIonDensity();
  auto ne = getElDensity();
  auto te = getElTemp();
  auto ti = getIonTemp();

  const o::LO background_Z = BACKGROUND_Z;
  const o::Real background_amu = BACKGROUND_AMU;
  auto useConstantBField = USE_CONSTANT_BFIELD;
  OMEGA_H_CHECK(useConstantBField);
  o::Reals bField_const(3);
  if(useConstantBField) {
    bField_const = o::Reals(o::Write<o::Real>(o::HostWrite<o::Real>(
      {CONSTANT_BFIELD0, CONSTANT_BFIELD1, CONSTANT_BFIELD2} )));
  }
  o::Real potential = BIAS_POTENTIAL;
  auto biased = BIASED_SURFACE;
  auto bGrids = o::Reals(o::Write<o::Real>(o::HostWrite<o::Real>(
    {gm->bGridX0, gm->bGridZ0, gm->bGridDx, gm->bGridDz})));
  auto bGridsN = o::LOs(o::Write<o::LO>(o::HostWrite<o::LO>({gm->bGridNx, gm->bGridNz})));
  const auto bfield_2dm = gm->getBfield2d();

  o::Write<o::Real> angle_d(nf, 0, "angle");
  o::Write<o::Real> debyeLength_d(nf, 0, "debyeLength");
  o::Write<o::Real> larmorRadius_d(nf, 0, "larmorRadius");
  o::Write<o::Real> childLangmuirDist_d(nf, 0, "childLangmuirDist");
  o::Write<o::Real> flux_d(nf, 0, "bdryFlux");
  o::Write<o::Real> impacts_d(nf, 0, "impacts");
  o::Write<o::Real> potential_d(nf, 0, "potential");
  int rank = this->rank;

  auto calculate = OMEGA_H_LAMBDA(const o::LO& fid) {
    if(exposed[fid]) {
      auto B = o::zero_vector<3>();
      auto fcent = p::face_centroid_of_tet(fid, coords, face_verts);
      if(!rank && debug)
        printf(" fid:%d::  %.5f %.5f %.5f \n", fid, fcent[0], fcent[1], fcent[2]);
      if(useConstantBField) {
        for(auto i=0; i<3; ++i)
          B[i] = bField_const[i];
      } else {
        p::interp2dVector(bfield_2dm, bGrids[0], bGrids[1], bGrids[2], bGrids[3],
           bGridsN[0], bGridsN[1], fcent, B, false);
      }
      //normal on boundary points outwards
      auto surfNormOut = p::bdry_face_normal_of_tet(fid,coords,face_verts);
      auto surfNormIn = -surfNormOut;
      auto magB = o::norm(B);
      auto magSurfNorm = o::norm(surfNormIn);
      auto angleBS = o::inner_product(B, surfNormIn);
      auto theta = acos(angleBS/(magB*magSurfNorm));
      if (theta > o::PI * 0.5) {
        theta = abs(theta - o::PI);
      }
      angle_d[fid] = theta*180.0/o::PI;

      if(!rank && debug) {
        printf("fid:%d surfNormOut:%g %g %g angleBS=%g theta=%g angle=%g\n", fid,
          surfNormOut[0], surfNormOut[1], surfNormOut[2],angleBS,theta,angle_d[fid]);
      }
      const auto tion = ti[fid];
      const auto tel = te[fid];
      const auto nel = ne[fid];
      const auto dens = density[fid];  //3.0E+19 ?
      o::Real dlen = 0;
      if(o::are_close(nel, 0.0)){
        dlen = 1.0e12;
      }
      else { //TODO use common constants
        dlen = sqrt(8.854187e-12*tel/(nel*pow(background_Z,2)*1.60217662e-19));
      }
      debyeLength_d[fid] = dlen;
      larmorRadius_d[fid] = 1.44e-4*sqrt(background_amu*tion/2)/(background_Z*magB);
      flux_d[fid] = 0.25* dens *sqrt(8.0*tion*1.60217662e-19/(o::PI*background_amu));
      impacts_d[fid] = 0.0;
      o::Real pot = potential;
      if(biased) {
        o::Real cld = 0;
        if(tel > 0.0) {
          cld = dlen * pow(abs(pot)/tel, 0.75);
        }
        else {
          cld = 1e12;
        }
        childLangmuirDist_d[fid] = cld;
      } else {
        pot = 3.0*tel;
      }
      potential_d[fid] = pot;

      if(!rank && debug)// || o::are_close(angle_d[fid],0))
        printf("angle[%d] %.5f\n", fid, angle_d[fid]);
    }
  };
  o::parallel_for(nf, calculate, "calculate_face_data");

  ff.angle = o::Reals(angle_d);
  ff.debyeLength = o::Reals(debyeLength_d);
  ff.larmorRadius = o::Reals(larmorRadius_d);
  ff.childLangmuirDist = o::Reals(childLangmuirDist_d);
  ff.bdryFlux = o::Reals(flux_d);
  ff.impact = o::Reals(impacts_d);
  ff.potential = o::Reals(potential_d);
  return true;
}

o::Reals GitrmBoundary::calculateScalarFieldOnBdryFacesFromFile(const std::string& fieldName,
    const std::string &file, Field3StructInput& fs, int debug) {
  debug = 1;
  auto rank = this->rank;
  if(!rank && debug)
    std::cout << __FUNCTION__ <<"\n";

  if(!stored)
    stored = storeBdrySurfaceMeshForPicpart();
  readInputDataNcFileFS3(file, fs);

  o::LO nf = (useBdryTagFields) ? mesh->nfaces() : o::LOs(*bdryFaceVertsPic).size()/3;
  o::LOs face_verts = (useBdryTagFields) ? mesh->ask_verts_of(2): o::LOs(*bdryFaceVertsPic);
  o::Read<o::I8> exposed = (useBdryTagFields) ? mark_exposed_sides(mesh) :
                           o::Read<o::I8>(o::Write<o::I8>(nf, 1, "dummyExposed"));
  o::Reals coords = (useBdryTagFields) ? mesh->coords() : o::Reals(*bdryFaceVertCoordsPic);
  if(!rank && debug > 1)
    std::cout << "selecting Data done nf " << nf << " nv " << face_verts.size() <<
      " ncoords " << coords.size()  << "\n";
 
  auto addTag = useBdryTagFields;

  int nR = fs.getNumGrids(0);
  int nZ = fs.getNumGrids(1);
  o::Real rMin = fs.getGridMin(0);
  o::Real zMin = fs.getGridMin(1);
  o::Real dr = fs.getGridDelta(0);
  o::Real dz = fs.getGridDelta(1);

  if(!rank && debug > 1)
    printf("nR %d nZ %d dr %g, dz %g, rMin %g, zMin %g \n",
        nR, nZ, dr, dz, rMin, zMin);

  //Interpolate at vertices and Set tag
  o::Write<o::Real> tag_d(nf, 0, "fieldName");
  const auto readInData_d = o::Reals(fs.data);
  auto len = readInData_d.size(); 
  if(!rank && debug > 1) {
    std::cout << fieldName << " file " << file << "\n";
  }
  auto lambda = OMEGA_H_LAMBDA(const o::LO& fid) {
    if(!exposed[fid]) {
      tag_d[fid] = 0;
      return;
    }
    auto pos = p::face_centroid_of_tet(fid, coords, face_verts);
    bool cylSymm = true;
    o::Real val = p::interpolate2dField(readInData_d, rMin, zMin, dr, dz,
      nR, nZ, pos, cylSymm, 1, 0);
    tag_d[fid] = val;
  };
  o::parallel_for(nf, lambda, "Calculate_face_fields");

  //TODO get fieldName (tagName) from Gitrmfields and store corresponding field data
  if(!addTag && fieldName=="IonDensity")
    ionDensity = o::Reals(tag_d);
  else if(!addTag && fieldName=="ElDensity")
    elDensity = o::Reals(tag_d);
  else if(!addTag && fieldName=="IonTemp")
    ionTemp = o::Reals(tag_d);
  else if(!addTag && fieldName=="ElTemp")
    elTemp = o::Reals(tag_d);
  if(!rank && debug>1)
    std::cout << "done " << __FUNCTION__ << "\n";
  return o::Reals(tag_d);
}


//TODO delete fullMesh if not needed later
//from the global d2bdry CSR data, based on full mesh elements
bool GitrmBoundary::storeBdrySurfaceMeshForPicpart(int debug) {
  debug = 1;
  if(!rank && debug)
    std::cout << __FUNCTION__ << " Bdry\n";
  const auto nfAll = fullMesh->nfaces();
  const auto nvAll = fullMesh->nverts();
  const auto allFaceVerts = fullMesh->ask_verts_of(2);
  const auto allCoords = fullMesh->coords();
  const auto dim = mesh->dim();
  const auto nelems = mesh->nelems();
  auto globalIds = picparts->globalIds(dim);

  const auto picOwners = picparts->entOwners(dim);
  const auto picSafeTags = picparts->safeTag();
  if(!rank && debug > 1)
    std::cout << rank << ": nelems " << nelems << " nfaces " << mesh->nfaces() 
      << " allFaceVertsSize " << allFaceVerts.size()
      << " safeSize " << picSafeTags.size() << " nowners " << picOwners.size() << "\n";

  //retrieve face data from elsewhere
  o::Read<o::LO> inFidPtrs; //TODO use GO
  o::Read<o::LO> inBdryFids; //GO
  //getBdryFacesData(gidList, inFidPtrs, inBdryFids);
  //TODO replace this by parallel netcdf read
  if(USE_READIN_CSR_BDRYFACES) {
    inFidPtrs = gm->getBdryCsrReadInPtrs();//  bdryCsrReadInDataPtrs;
    inBdryFids = gm->getBdryCsrReadInFids(); // bdryCsrReadInData;
  } else {
    inFidPtrs = gm->getBdryCsrCalculatedPtrs(); //bdryFacePtrsSelected;
    inBdryFids = gm->getBdryFidsCalculated(); // bdryFacesSelectedCsr;
    OMEGA_H_CHECK(inFidPtrs.size());
  }
  
  if(!rank && debug> 1)
    std::cout << rank << ": nBfidPtrs " << inFidPtrs.size() << " nBdryFids "
      << inBdryFids.size() << "\n marking bdry vertices and face gids\n";

  o::Write<o::LO> dataNums(nelems, 0, "dataNums");
  o::Write<o::LO> faceGid_flags(nfAll, 0, "faceGid_flags");
  o::Write<o::LO> gFaceVert_flags(nvAll, 0, "faceGVert_flags");
  auto rank = this->rank;

  //copy mesh data for elements of this picpart.
  auto makePtrs = OMEGA_H_LAMBDA(const o::LO& e) {
    //for explicit storage of individual face vertex coordinates
    //input face ptrs and fids are based on global element ids
    auto gid = globalIds[e];
    if(!rank && debug>1 && e >1000 && e < 1003)
      printf("%d:e %d gid %ld \n", rank, e, gid);
    auto beg = inFidPtrs[gid];
    auto end = inFidPtrs[gid+1];
    dataNums[e] = end - beg;
    if(!rank && debug>1 && e < 1)
      printf(" %d : dataNums %d e %d \n", rank, dataNums[e], e);
    for(auto i=beg; i<end; ++i) {
      //global face id
      auto fid = inBdryFids[i];
      Kokkos::atomic_exchange(&(faceGid_flags[fid]), 1);
      //o::Few<o::LO, 3>
      const auto fv2v = o::gather_verts<3>(allFaceVerts, fid);
      for(int j=0; j<3; ++j) {
        auto vid = fv2v[j];
        Kokkos::atomic_exchange(&(gFaceVert_flags[vid]), 1);
      }
    }
  };
  o::parallel_for(nelems, makePtrs, "makePicpartDist2BdryPtrs");
  
  bdryFaceIdPtrsPic = std::make_shared<o::LOs>(o::offset_scan(o::LOs(dataNums),
                       "bdryFaceIdPtrsPic"));

  auto nPtrsPic = (*bdryFaceIdPtrsPic).size();
  auto nBdryFids = o::HostRead<o::LO>(*bdryFaceIdPtrsPic)[nPtrsPic-1];
  auto glob2locFVertmap = utils::makeLocalIdMap<o::LO>(gFaceVert_flags, "glob2locFVertmap");
  auto nLocVerts = o::get_sum(o::LOs(gFaceVert_flags));
  
  if(!rank && debug > 1)
    std::cout << " storing relevant bdry vertices nLocVerts " << nLocVerts
      << " nBdryFids " << nBdryFids << " nPtrs " <<  nPtrsPic << "\n";  

  //Convert global fids to that of separate local storage.
  auto glob2locFidMap = utils::makeLocalIdMap<o::LO>(faceGid_flags, "glob2locFidMap");
  o::Write<o::Real> bdryFaceVertCoords_w(nLocVerts*3, 0, "bdryFaceVertCoordsPic");
  //copy neded face vertex coords
  auto copyCoords = OMEGA_H_LAMBDA(const o::LO& id) {
    if(gFaceVert_flags[id]) {
      auto lid = glob2locFVertmap[id];
      OMEGA_H_CHECK(lid >=0);
      //auto v = o::gather_vectors<1, 3>(coords, id);
      for(int i=0; i<3; ++i)
        Kokkos::atomic_exchange(&(bdryFaceVertCoords_w[lid*3+i]), allCoords[3*id+i]);
    }
  };
  o::parallel_for(nvAll, copyCoords, "copyFaceCoords");
  bdryFaceVertCoordsPic = std::make_shared<o::Reals>(bdryFaceVertCoords_w);

  if(!rank && debug>1)
    std::cout << " copying vertex ids\n";
  //copy face vert Ids
  auto nFaceLids = o::get_sum(o::LOs(faceGid_flags));
  o::Write<o::LO> bdryFaceVerts_w(nFaceLids*3, 0, "bdryFaceVertsPic");
  auto copyVerts = OMEGA_H_LAMBDA(const o::LO& id) {
    if(faceGid_flags[id]) {
      //face verts
      const auto fv2v = o::gather_verts<3>(allFaceVerts, id);
      auto lid = glob2locFidMap[id];
      for(int i=0; i<3; ++i) {
        auto vid = fv2v[i];
        bdryFaceVerts_w[lid*3+i] = glob2locFVertmap[vid];
      }
    }
  };
  o::parallel_for(nfAll, copyVerts, "copyFaceVerts");
  bdryFaceVertsPic = std::make_shared<o::LOs>(bdryFaceVerts_w);

  if(!rank && debug>1)
    std::cout << " copying bdry face ids nUniqIds " << nFaceLids << " total "
      << nBdryFids << "\n";
  //copy faceids
  auto bdryFaceIdPtrsPic_ = o::LOs(*bdryFaceIdPtrsPic);
  o::Write<o::LO> bdryFaceIds_w(nBdryFids, -1, "bdryFaceIdsPic");
  auto copyFids = OMEGA_H_LAMBDA(const o::LO& e) {
    //map from global (since data is stored so) to local
    auto gid = globalIds[e];
    auto beg = inFidPtrs[gid];
    auto end = inFidPtrs[gid+1];
    auto b = bdryFaceIdPtrsPic_[e];

    for(o::LO i=0; i<end - beg; ++i) {
      auto gfid = inBdryFids[beg+i];
      if(!rank && debug >1 && e <1)
        printf("%d: %d %d %d %d %d %d \n", rank, e, gid, beg, end, b, gfid);
      bdryFaceIds_w[b+i] = glob2locFidMap[gfid];
    }
  };
  o::parallel_for(nelems, copyFids, "copyBdryFids");
  bdryFaceIdsPic = std::make_shared<o::LOs>(bdryFaceIds_w);
  if(!rank && debug>1)
    std::cout << " done " << __FUNCTION__ << "\n";
  return true;
}


