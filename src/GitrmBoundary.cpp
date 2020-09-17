#include <iostream>

#include "GitrmBoundary.hpp"


GitrmBoundary::GitrmBoundary(GitrmMesh* gm, FaceFieldFillCoverage fill):
   gm(gm), coverage(fill) {
  picparts = gm->getPicparts();
  mesh = picparts->mesh();
  fullMesh = gm->getFullMesh();
  rank = picparts->comm()->rank();
  if(!rank)
    std::cout << rank << ": Initializing GitrmBoundary\n";
  initBoundaries();
}

void GitrmBoundary::setBoundaryInput() {
  isBiasedSurf = gm->isBiasedSurface();
  biasPotential = gm->getBiasPotential();
  useAllMatBdryFaces = gm->isUsingAllBdryFaces();
  if(!rank)
    std::cout << rank << ": biased " << isBiasedSurf << " potential " <<
      biasPotential << " useAllMatBdryFaces " << useAllMatBdryFaces << "\n";
  //debugging. This is not connected to storing boundary faces
  if(useAllMatBdryFaces)
    setAllBdryFacesOfModelIds();

}

void GitrmBoundary::initBoundaries() {
  // if any of the other picpart has less than all the elements of fullMesh
  MPI_Barrier(picparts->comm()->get_impl());
  int min = (mesh->nelems() == fullMesh->nelems());
  int minAll = 0;
  int stat = MPI_Allreduce(&min, &minAll, 1, MPI_INT, MPI_MIN,
                           picparts->comm()->get_impl());
  OMEGA_H_CHECK(!stat);

  //NOTE:if all elements of fullMesh included in picpart, the picpart keeps original
  // numbering, making the re-numbering of core or owning elements invalid.
  useOrigBdryData = minAll;

  // PICPART: add tags on all faces of all elements of the picpart. For this option to be
  // valid, all the required bdry faces have to be part of the picpart. That occurs
  // only in the case of fullMesh picpart.
  // BDRY: separate bdry data stored, by selecting relevant data and re-numbering. 
  // The original numbering is not valid in this case.
  if(coverage == PICPART) {
    //set flag for enabling tag fields
    useBdryTagFields= true;
    if(!minAll) {
      printf("Error: The option to store boundary data on the picpart is incompatible\n");
      Omega_h_fail("ERROR: Cannot store the whole boundary data on smaller picpart\n");
    }
    useStoredBdryData = false;
  } else if(coverage == BDRY) { //separat bdry data stored
    useBdryTagFields = false;
    useStoredBdryData = true;
  } else {
    Omega_h_fail("%d: Error : Dist-to-bdry coverage not known\n", rank);
  }

  std::cout << rank << ": Setting bdry storage, use tag? " << useBdryTagFields
            << " useOrigBdryData "<< useOrigBdryData << " nel " << mesh->nelems()
            << " fullNelems " << fullMesh->nelems() << "\n";
}

o::Reals GitrmBoundary::getBdryFaceVertCoordsPic() const {
  OMEGA_H_CHECK(initData);
  if(useOrigBdryData || useAllMatBdryFaces)
    return fullMesh->coords();
  return bdryFaceVertCoordsPic;
}

o::LOs GitrmBoundary::getBdryFaceVertsPic() const {
  OMEGA_H_CHECK(initData);
  if(useOrigBdryData || useAllMatBdryFaces)
    return fullMesh->ask_verts_of(2);
  return bdryFaceVertsPic;
}

//selection criteria is used from the mesh class. For brute-force run
void GitrmBoundary::setAllBdryFacesOfModelIds() {
  //All material bdry faceIds of fullMesh. The data not selected for picpart.
  //In this case fields calculated and stored for all mesh faces.
  //Note that this option is independent of storing bdryFaceIdsPic and 
  //bdryFaceIdPtrsPic, which are set if that option is used, irrespective
  //of that data is used later or not.
  allMaterialBdryFids = gm->selectBdryFacesOfModelIds(fullMesh->nfaces());
  // each ptr is to use all of allMaterialBdryFids if value is -1
  allMaterialBdryFidPtrs = o::LOs(fullMesh->nfaces()+1, -1, "allMaterialBdryFidPtrs");
}

o::LOs GitrmBoundary::getBdryFaceIdsPic() const {
  OMEGA_H_CHECK(initData);
  if(useAllMatBdryFaces)
    return allMaterialBdryFids;
  if(useOrigBdryData)
    return gm->getBdryCsrReadInFids();
  return bdryFaceIdsPic;
}

o::LOs GitrmBoundary::getBdryFaceIdPtrsPic() const {
  OMEGA_H_CHECK(initData);
  if(useAllMatBdryFaces)
    return allMaterialBdryFidPtrs;
  if(useOrigBdryData)
    return gm->getBdryCsrReadInPtrs();
  return bdryFaceIdPtrsPic;
}

o::GOs GitrmBoundary::getOrigGlobalIds() const {
  return origGlobalIds;
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

// store bdry data separately. Otherwise call calculateBdryData directly
//to be called from mesh class, after calculating electron, ion fields
void GitrmBoundary::storeBdryData() {
  OMEGA_H_CHECK(useStoredBdryData);
  if(!initData)
    storeBdrySurfaceMeshForPicpart();
  storeBdryFaceFieldsForPicpart(initData);
}

//to be called from gm, if tags added on mesh, after calculating el, ion fields
void GitrmBoundary::calculateBdryData(GitrmMeshFaceFields& ff) {
  if(useStoredBdryData)
    Omega_h_fail("%d: Error: In dist-to-bdry data storage\n", rank);
  OMEGA_H_CHECK(useBdryTagFields);
  calculateBdryFaceFields(ff);
}

void GitrmBoundary::storeBdryFaceFieldsForPicpart(bool pre, int debug) {
  if(!pre)
    Omega_h_fail("%d: Required Bdry Mesh should be stored prior to %s\n", rank, __FUNCTION__);
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
//If tags added on mesh, then this is called from mesh class to return fields.
//Otherwise, the bdry data stored separate local order in this class.
//NOTE: tags fields calculated on only picpart faces, not full mesh, which
//will not include the bdry faces needed.
bool GitrmBoundary::calculateBdryFaceFields(GitrmMeshFaceFields& ff, int debug) {
  std::cout << rank << ": CalculateBdryFaceFields-bdry\n";
  if(!(useBdryTagFields || initData))
    Omega_h_fail("Error: Store bdry data before this function\n");
  o::LO nf = (useBdryTagFields) ? mesh->nfaces() : (getBdryFaceVertsPic()).size()/3;
  if(!nf)
    Omega_h_fail("Error : getting number of bdry face ids\n");
  o::LOs face_verts = (useBdryTagFields) ? mesh->ask_verts_of(2): getBdryFaceVertsPic();
  o::Read<o::I8> exposed = (useBdryTagFields) ? mark_exposed_sides(mesh) :
                           o::Read<o::I8>(o::Write<o::I8>(nf, 1, "dummyExposed"));
  o::Reals coords = (useBdryTagFields) ? mesh->coords() : getBdryFaceVertCoordsPic();
  //tag added only if the fields are added on the picpart mesh,
  //not when separate data is stored in bdry class.

  //if addTag=true, make sure density,temp tags are added prior to this
  //The addTag flag sanity check already done in the selectMeshData(..)
  auto density = getIonDensity();
  auto ne = getElDensity();
  auto te = getElTemp();
  auto ti = getIonTemp();

  const o::LO background_Z = gm->getBackgroundZ();
  const o::Real background_amu = gm->getBackgroundAmu();
  auto useConstantBField = gm->isUsingConstBField();
  //TODO
  //OMEGA_H_CHECK(useConstantBField);
  o::Reals bField_const(3);
  if(useConstantBField)
    bField_const = utils::getConstBField();

  o::Real potential = biasPotential;
  auto biased = isBiasedSurf;

  if(debug >1 && !rank)
    std::cout << "BIAS_POTENTIAL " << biased << " " << potential << " nf " << nf << "\n";

  auto bGrids = o::Reals(o::Write<o::Real>(o::HostWrite<o::Real>(
    {gm->bGridX0, gm->bGridZ0, gm->bGridDx, gm->bGridDz})));
  auto bGridsN = o::LOs(o::Write<o::LO>(o::HostWrite<o::LO>({gm->bGridNx, gm->bGridNz})));
  const auto bfield_2dm = gm->getBfield2d();
  const auto bGrid1 = gm->getBfield2dGrid(1);
  const auto bGrid2 = gm->getBfield2dGrid(2);

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
      if(useConstantBField) {
        for(auto i=0; i<3; ++i)
          B[i] = bField_const[i];
      } else {
        p::interp2dVector_wgrid(bfield_2dm, bGrid1, bGrid2, fcent, B, true);
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

      const auto tion = ti[fid];
      const auto tel = te[fid];
      const auto nel = ne[fid];
      const auto dens = density[fid];

      o::Real dlen = 0;
      if(o::are_close(nel, 0.0)){
        dlen = 1.0e12;
      }
      else { //TODO use common constants
        dlen = sqrt(8.854187e-12*tel/(nel*pow(background_Z,2)*1.60217662e-19));
      }
      debyeLength_d[fid] = dlen;
      larmorRadius_d[fid] = 1.44e-4*sqrt(background_amu*tion/2)/(background_Z*magB);
      flux_d[fid] = 0.25* dens *sqrt(8.0*tion*1.60217662e-19/(3.1415*background_amu));
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
      if(!rank && debug > 2) {
        printf(" fid %d centroid  %.15f %.15f %.15f \n", fid, fcent[0], fcent[1], fcent[2]);
        printf(" fid %d angleBS %g theta %g angle %g\n", fid, angleBS, theta,
            angle_d[fid]);
        printf(" fid %d ti %g te %g ne %g dens %g\n", fid, tion, tel, nel, dens);
        printf(" fid %d: B %g %g %g surfNormOut:%.15f %.15f %.15f \n",
          B[0], B[1], B[2], surfNormOut[0], surfNormOut[1], surfNormOut[2]);

      }

      if(debug > 1)
        printf("%d: fid %d DL %g LR %g flux %g impact %g CLD %g pot %g\n",
        rank, fid, dlen, larmorRadius_d[fid], flux_d[fid], impacts_d[fid],
        childLangmuirDist_d[fid], pot);
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
  std::cout << rank << " :" <<  __FUNCTION__ << " done\n";
  return true;
}

o::Reals GitrmBoundary::calculateScalarFieldOnBdryFacesFromFile(const std::string& fieldName,
    const std::string &file, Field3StructInput& fs, int debug) {
  debug = 1;
  auto rank = this->rank;
  if(!rank && debug)
    std::cout << rank << " :" << __FUNCTION__ << " " << fieldName << "\n";

  if(!initData)
    storeBdrySurfaceMeshForPicpart();
  if(!(useBdryTagFields || initData))
    Omega_h_fail("Error: Store bdry data before this step\n");

  readInputDataNcFileFS3(file, fs);

  if(debug>1)
    std::cout << "Read fields from file. Getting mesh tag fields " << useBdryTagFields << "\n";
  o::LO nf = (useBdryTagFields) ? mesh->nfaces() : (getBdryFaceVertsPic()).size()/3;
  o::LOs face_verts = (useBdryTagFields) ? mesh->ask_verts_of(2): getBdryFaceVertsPic();
  o::Read<o::I8> exposed = (useBdryTagFields) ? mark_exposed_sides(mesh) :
                           o::Read<o::I8>(o::Write<o::I8>(nf, 1, "dummyExposed"));
  o::Reals coords = (useBdryTagFields) ? mesh->coords() : getBdryFaceVertCoordsPic();
  if(debug > 1)
    std::cout << rank << ": selecting Data done nf " << nf << " nv " << face_verts.size() <<
      " ncoords " << coords.size() << " addTags " << useBdryTagFields << "\n";

  auto addTag = useBdryTagFields;

  int nR = fs.getNumGrids(0);
  int nZ = fs.getNumGrids(1);
  o::Real rMin = fs.getGridMin(0);
  o::Real zMin = fs.getGridMin(1);
  o::Real dr = fs.getGridDelta(0);
  o::Real dz = fs.getGridDelta(1);
  bool cylSymm = true;

  if(!rank && debug > 1)
    printf("%d: nR %d nZ %d dr %g, dz %g, rMin %g, zMin %g cylSymm %d\n",
      rank, nR, nZ, dr, dz, rMin, zMin, cylSymm);

  //Interpolate at vertices and Set tag
  o::Write<o::Real> tag_d(nf, 0, fieldName);
  const auto readInData_d = o::Reals(fs.data.write());
  const auto gridx = o::Reals(fs.grid1);
  const auto gridz = o::Reals(fs.grid2);
  auto len = readInData_d.size();
  if(!rank && debug >1)
    std::cout << rank << ": " << fieldName << " file " << file << " data-size " << len << "\n";

  auto lambda = OMEGA_H_LAMBDA(const o::LO& fid) {
    //NOTE only exposed faces considered
    if(!exposed[fid]) {
      tag_d[fid] = 0;
      return;
    }
    auto pos = p::face_centroid_of_tet(fid, coords, face_verts);
    auto val = p::interpolate2d_wgrid(readInData_d, gridx, gridz, pos, cylSymm, 1, 0);
    tag_d[fid] = val;

    if(!rank && debug > 2 && fid < 1) {
      for(int i=0 ; i < 20 && i<len; ++i)
        printf("%d: i %d readInData: %g\n",rank, i, readInData_d[i]);
    }
    if(!rank && debug > 2 && val > 0 && fid < 100)
      printf("%d: fid %d tag %g pos %g %g %g\n",rank, fid, val, pos[0], pos[1], pos[2]);
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
  if(debug>1)
    std::cout << rank << ": done " << __FUNCTION__ << " " << fieldName << "\n";
  return o::Reals(tag_d);
}

bool GitrmBoundary::checkBdryDataUse() {
  //Retreive the original gid tag from picpart. It was set in fullMesh
  origGlobalIds = mesh->get_array<o::GO>(o::REGION, "origGids");

  //if all elements of fullMesh included in picpart, the picpart keeps original
  // numbering, making the re-numbering of core elements invalid. TODO Update this
  if(useOrigBdryData) {
    if(fullMesh->nelems() > mesh->nelems()) {
      printf("Error: the option to use the whole bdry face data requires "
        "all elements of fullmesh in the picpart. Use USE_STORED_BDRYDATA option\n");
      Omega_h_fail("Error: use USE_STORED_BDRYDATA for distributed mesh\n");
    }
    initData = true;
  }
  return initData;
}

template<typename T>
OMEGA_H_DEVICE T getElemIdUsedInBdryData(T el, bool readInData, o::GOs& origGlobalIds) {
  //based on the read in data from file as stored per global el id
  return (readInData) ? origGlobalIds[el] : el;
}

/* To get the original global ids, the tag origGids added to the full mesh
 * before passing it to the picpart creation.
 * If the buffer elements are skipped from bdry storage, the rebuild
 * would need to be done once in every time step.
*/
bool GitrmBoundary::storeBdrySurfaceMeshForPicpart(int debug) {
  debug = 2;
  if(debug)
    std::cout << rank << ": " <<  __FUNCTION__ << " Bdry\n";

  if(debug)
    MPI_Barrier(MPI_COMM_WORLD);
  auto rank = this->rank;

  if(checkBdryDataUse())
    return true;

  bool useGidBasedData = USE_READIN_CSR_BDRYFACES;

  const auto nelAll = fullMesh->nelems();
  const auto nfAll = fullMesh->nfaces();
  const auto nvAll = fullMesh->nverts();
  const auto allFaceVerts = fullMesh->ask_verts_of(2);
  const auto allCoords = fullMesh->coords();
  const auto dim = mesh->dim();
  const auto nelems = mesh->nelems();

  //create local to original global id map ONLY for CORE elements
  auto ownersAll = gm->getOwnersAll();

  if(debug > 1) {
    const auto picOwners = picparts->entOwners(dim);
    const auto picSafeTags = picparts->safeTag();
    const auto picLocalIds = picparts->rankLocalIndex(dim);
    std::cout << rank << ": nelems " << nelems << " nfaces " << mesh->nfaces()
      << " safeSize " << picSafeTags.size()
      << " nAllFaceVerts " << allFaceVerts.size() << " nfullELelms " << nelAll << "\n";
  }

  if(debug > 2) {
    o::HostRead<o::GO> gids_h(origGlobalIds);
    printf("useReadInData %d (if 1 then bdry data is based on origGid\n", useGidBasedData);
    for(int i=0; i<nelAll; ++i)
      if(gids_h[i] >=0)
        printf("%d: origGid %d \n", rank, gids_h[i]);
  }

  //retrieve face data from elsewhere
  o::LOs inFidPtrs; //TODO use GO
  o::LOs inBdryFids; //GO
  //TODO replace this by parallel netcdf read
  if(useGidBasedData) {
    inFidPtrs = gm->getBdryCsrReadInPtrs();
    inBdryFids = gm->getBdryCsrReadInFids();
  } else {
    inFidPtrs = gm->getBdryCsrCalculatedPtrs(); //bdryFacePtrsSelected;
    inBdryFids = gm->getBdryFidsCalculated(); // bdryFacesSelectedCsr;
    OMEGA_H_CHECK(inFidPtrs.size());
  }

  //if(debug> 1)
    std::cout << rank << ": nBfidPtrs " << inFidPtrs.size() << " nBdryFids "
      << inBdryFids.size() << ". Marking bdry vertices and face gids\n";

  o::Write<o::LO> dataNums(nelems, 0, "dataNums");
  o::Write<o::LO> faceGid_flags(nfAll, 0, "faceGid_flags");
  o::Write<o::LO> gFaceVert_flags(nvAll, 0, "faceGVert_flags");
  o::LO exchange = 1; //identifier non default (0) value

  //copy mesh data from the boundaries of fullMesh for this picpart.
  //For explicit storage of individual face vertex coordinates
  // the input face ptrs and fids are based on global element ids
  auto makePtrs = OMEGA_H_LAMBDA(const o::LO& e) {
    o::LO eid = getElemIdUsedInBdryData(e, useGidBasedData, origGlobalIds); //TODO auto for GO
    auto beg = inFidPtrs[eid];
    auto end = inFidPtrs[eid+1];
    o::LO nf = end - beg;
    dataNums[e] = nf;

    if(!nf)
      printf("%d:e %d eid %d  ptr %d %d\n", rank, e, eid, beg, end);
    OMEGA_H_CHECK(nf);

    for(o::LO i=beg; i<end; ++i) {
     //global face id
      auto fid = inBdryFids[i];
      Kokkos::atomic_exchange(&(faceGid_flags[fid]), exchange);
      //o::Few<o::LO, 3>
      const auto fv2v = o::gather_verts<3>(allFaceVerts, fid);
      for(int j=0; j<3; ++j) {
        auto vid = fv2v[j];
        Kokkos::atomic_exchange(&(gFaceVert_flags[vid]), exchange);
      }
    }
  };
  o::parallel_for(nelems, makePtrs, "makePicpartDist2BdryPtrs");
  auto bfPtrs = o::offset_scan(o::LOs(dataNums), "bdryFaceIdPtrsPic");
  bdryFaceIdPtrsPic = o::LOs(o::deep_copy(bfPtrs));
  if(debug > 1)
    std::cout << rank <<": Created faceId csr ptrs\n";

  auto nPtrsPic = bdryFaceIdPtrsPic.size();
  auto nBdryFids = o::HostRead<o::LO>(bdryFaceIdPtrsPic)[nPtrsPic-1];
  int nLocVerts = 0;
  auto glob2locFVertmap = utils::makeLocalIdMap<o::LO>(gFaceVert_flags, exchange,
    nLocVerts, "glob2locFVertmap");

  if(debug > 1)
    std::cout << rank << " storing relevant bdry vertices nLocVerts " << nLocVerts
      << " nBdryFids " << nBdryFids << " nPtrs " <<  nPtrsPic << "\n";

  //Convert global fids to that of separate local storage.
  int nums = 0;
  auto glob2locFidMap = utils::makeLocalIdMap<o::LO>(faceGid_flags, exchange,
                        nums, "glob2locFidMap");
  o::Write<o::Real> bdryFaceVertCoords_w(nLocVerts*3, 0, "bdryFaceVertCoordsPic");
  //copy neded face vertex coords
  auto copyCoords = OMEGA_H_LAMBDA(const o::LO& id) {
    if(gFaceVert_flags[id] > 0) {
      auto lid = glob2locFVertmap[id];
      OMEGA_H_CHECK(lid >=0);
      for(int i=0; i<3; ++i)
        Kokkos::atomic_exchange(&(bdryFaceVertCoords_w[lid*3+i]), allCoords[3*id+i]);
    }
  };
  o::parallel_for(nvAll, copyCoords, "copyFaceCoords");
  bdryFaceVertCoordsPic = o::Reals(bdryFaceVertCoords_w);

  if(debug > 2)
    std::cout << rank << ": copying vertex ids\n";
  //copy face vert Ids
  auto nFaceLids = o::get_sum(o::LOs(faceGid_flags));
  o::Write<o::LO> bdryFaceVerts_w(nFaceLids*3, 0, "bdryFaceVertsPic");
  auto copyVerts = OMEGA_H_LAMBDA(const o::LO& fid) {
    if(faceGid_flags[fid] > 0) {
      //face verts
      const auto fv2v = o::gather_verts<3>(allFaceVerts, fid);
      auto lid = glob2locFidMap[fid];
      for(int i=0; i<3; ++i) {
        auto vid = fv2v[i];
        bdryFaceVerts_w[lid*3+i] = glob2locFVertmap[vid];
      }
    }
  };
  o::parallel_for(nfAll, copyVerts, "copyFaceVerts");
  bdryFaceVertsPic = o::LOs(bdryFaceVerts_w);

  if(debug>1)
    std::cout << rank << " copying bdry face ids nUniqIds " << nFaceLids << " total "
      << nBdryFids << "\n";
  //copy faceids
  auto bfidPtrs = bdryFaceIdPtrsPic;
  o::Write<o::LO> bdryFaceIds_w(nBdryFids, -1, "bdryFaceIdsPic");
  auto copyFids = OMEGA_H_LAMBDA(const o::LO& e) {
    //map from local to global
    o::LO eid = getElemIdUsedInBdryData(e, useGidBasedData, origGlobalIds); //TODO GO
    auto beg = inFidPtrs[eid];
    auto end = inFidPtrs[eid+1];
    auto b = bfidPtrs[e];
    for(o::LO i=0; i < (end - beg); ++i) {
      auto fid = inBdryFids[beg+i];
      bdryFaceIds_w[b+i] = glob2locFidMap[fid];
      if(debug >2) {
        auto f = p::get_face_coords_of_tet(allFaceVerts, allCoords, fid);
        printf("%d: e %d eid %d ind %d fid %d f %g %g %g :"
          " %g %g %g : %g %g %g\n", rank, e, eid, b+i, fid,
          f[0][0],f[0][1],f[0][2], f[1][0],f[1][1],f[1][2], f[2][0],f[2][1],f[2][2]);
      }
    }
  };

  o::parallel_for(nelems, copyFids, "copyBdryFids");
  bdryFaceIdsPic = o::LOs(bdryFaceIds_w);
  if(debug>1)
    std::cout << rank << ": done " << __FUNCTION__ << "\n";
  initData = true;
  if(debug)
    MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

//Move to unit tests
void GitrmBoundary::testDistanceToBdry(int debug) {
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << rank << " Testing DistanceToBdry \n";

  const auto coords = fullMesh->coords();
  const auto face_verts = fullMesh->ask_verts_of(2);
  const auto mesh2verts = fullMesh->ask_elem_verts();
  const auto lmesh2verts = mesh->ask_elem_verts();
  const auto lcoords = mesh->coords();
  const auto nel = mesh->nelems();

  const auto bdryFaceVertCoords = bdryFaceVertCoordsPic;
  const auto bdryFaceVerts = bdryFaceVertsPic;
  const auto bdryFaceIds = bdryFaceIdsPic;
  const auto bdryFaceIdPtrs =  bdryFaceIdPtrsPic ;
  const auto origGIds = origGlobalIds;

  const auto bdryCsrReadInDataPtrs = gm->getBdryCsrReadInPtrs();
  const auto bdryCsrReadInData = gm->getBdryCsrReadInFids();

  const auto owners = picparts->entOwners(3);

  //test makeLocalIdMap function
  auto ownersAll = gm->getOwnersAll();
  auto nelAll = fullMesh->nelems();
  o::Write<o::GO> origMeshGIds(nelAll, -1, "origMeshIds");
  int num=0;
  auto coreIds = utils::makeLocalIdMap(ownersAll, rank, num, "coreIds");
  o::parallel_for(nelAll, OMEGA_H_LAMBDA(const o::LO& e) {
    auto lid = coreIds[e];
    if(lid >= 0) {
      origMeshGIds[lid] = e;
    }
  });
  if(false) {
  //test if the origMeshGIds is same as origGIds
  o::parallel_for(nel, OMEGA_H_LAMBDA(const o::LO& e) {
    auto id = origMeshGIds[e];
    auto gid = origGIds[e];
    //if(id >= 0)
    //  OMEGA_H_CHECK(id == gid);
  });
  }

  std::cout << rank << ": Test Dist2Bdry \n";
  //NOTE: since bdry mesh data (reduced 2d data) is not standard mesh, bdry fid
  //cannot be used to get bdry element.
  auto lambda = OMEGA_H_LAMBDA(const int& e) {
    o::LO lbeg = bdryFaceIdPtrs[e];
    o::LO lend = bdryFaceIdPtrs[e+1];
    int lnFaces = lend - lbeg;

    //compare full read-in data with seperate copy per core. Only fids for core checked
    o::LO gid = origGIds[e];
    o::LO gbeg = bdryCsrReadInDataPtrs[gid];
    o::LO gend = bdryCsrReadInDataPtrs[gid+1];
    int gnFaces = gend - gbeg;
    if(lnFaces != gnFaces)
      printf(" %d:FIDS_UNEQUAL e %d  gid %d gbeg %d lbeg %d lnFaces %d gnFaces %d \n",
        rank, e, gid, gbeg, lbeg, lnFaces, gnFaces);
    OMEGA_H_CHECK(lnFaces == gnFaces);

    for(int i=0; i<lnFaces; ++i) {
      o::LO lbfid = bdryFaceIds[lbeg+i];
      o::LO gbfid = bdryCsrReadInData[gbeg+i];
      if(lbfid < 0)
        printf(" %d:FIDS<0 e %d  gid %d gbeg %d lbeg %d lbfid %d gbfid %d\n",
          rank, e, gid, gbeg, lbeg, lbfid, gbfid);
      OMEGA_H_CHECK(lbfid >= 0);
      OMEGA_H_CHECK(gbfid >= 0);

      auto lf = p::get_face_coords_of_tet(bdryFaceVerts, bdryFaceVertCoords, lbfid);
      auto gf = p::get_face_coords_of_tet(face_verts, coords, gbfid);
      for(int j=0; j<3;++j) {
        for(int k=0; k<3; ++k) {
          auto eq = p::almost_equal(lf[j][k], gf[j][k]);
          if(!eq || (debug >2 && e%10000==0))
            printf(" %d: BDRY_COORDS_VAL_CHECK e %d gid %d  %f %f\n",
              rank, e, gid, lf[j][k], gf[j][k]);
          OMEGA_H_CHECK(eq);
        }
      } //for
      //test dist calculation
      int region = -1;
      auto ref = p::centroid_of_tet(e, lmesh2verts, lcoords);
      auto pt = p::closest_point_on_triangle(lf, ref, &region);
      auto dist = o::norm(pt - ref);
      OMEGA_H_CHECK(dist >= 0);
    }
  };
  o::parallel_for(nel, lambda, "testDist2bdry");
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << rank << " Done testDist2Bdry\n";
}

