#include <fstream>
#include <cstdlib>
#include <vector>
#include <set>
#include <random>
#include <chrono>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_scan.hpp>
#include "Omega_h_shape.hpp"
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
#include "GitrmInputOutput.hpp"
#include <fstream>

int iTimePlusOne = 0;


GitrmParticles::GitrmParticles(GitrmMesh* gm, p::Mesh& picparts, long int nPtcls,
   int nIter, double dT,
   bool cuRnd, unsigned long int seed, unsigned long int seq, bool gitrRnd):
   gm(gm), picparts(picparts), mesh(*picparts.mesh()), ptcls(nullptr) {
  //move to where input is handled
  useCudaRnd = cuRnd;
  useGitrRndNums = gitrRnd;
  totalPtcls = nPtcls;
  numIterations = nIter;
  timeStep = dT;

  int rn, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rn);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  rank = rn; //picparts.comm()->rank();
  commSize = size;//picparts.comm()->size();
  isFullMesh = picparts.isFullMesh();

  elemOwners = picparts.entOwners(3);

  ptclSplitRead = PTCLS_SPLIT_READ;
  if(!useGitrRndNums)
    initRandGenerator(seed);
  mustFindAllPtcls = MUST_FIND_ALL_PTCLS;

  auto geomName = gm->getGeometryName();

  if(geomName == "pisces")
    geometryId = PISCES_ID;
  else if(geomName == "Iter")
    geometryId = ITER_ID;
  else
    geometryId = INVALID;

  int dataSize = gm->getNumDetectorModelIds();
  initPtclDetectionData(dataSize);
  if(false)
    initPtclsOnGeometricBoundarys(*gm);
}

GitrmParticles::~GitrmParticles() {
  delete ptcls;
  if(useCudaRnd)
    cudaFree(cudaRndStates);
}

void GitrmParticles::initRandGenerator(unsigned long int seed) {
  bool debug = true;
  if(useCudaRnd) {
   //Caution : CUDA states for total particles initialized in all ranks.
   //This is to match with that in GITR, but ONLY VALID when migration is off.
    auto nPtcls = totalPtcls;
    curandState* cudaRndStates;
    cudaMalloc((void **)&cudaRndStates, nPtcls*sizeof(curandState));
    if(!rank)
      std::cout <<"\n\n ******* Initializing CUDA random numbers for "
                << nPtcls << " ptcls\n\n";
    Omega_h::parallel_for(nPtcls, OMEGA_H_LAMBDA(const o::LO& id) {
      curand_init(id, 0, 0, &cudaRndStates[id]);
    });
    this->cudaRndStates = cudaRndStates;
    bool deb = false;
    if(deb) {
      Omega_h::parallel_for(1, OMEGA_H_LAMBDA(const o::LO& id){
        auto localState = cudaRndStates[id];
        double r;
        for(int i=0; i<1000; ++i) {
          r = curand_uniform(&localState);
          printf("gitrm-rnd %d %.15f\n", i, r);
        }
        cudaRndStates[id] = localState;
      });
    }
  } else {
    // kokkos initialze generators for maximum threads in the device.
    // The seed=0 not accepted since in kokkos it results using a default seed.
    // No sequence accepted in kokkos. Note that the reusing a seed doesn't
    // guarantee that the previous result will be reproduced. Because, a particle
    // is not certain to be processed by the same thread or set of threads,
    // in order to get the same rand num at all steps even though the threads get
    // the same rand initial states. Any instance of repoduced result could be
    // because the threads keeping the same order; which is not guaranteed.
    if(!seed) {
      auto time0 = std::chrono::high_resolution_clock::now();
      seed = time0.time_since_epoch().count();
    }
    std::mt19937 gen(seed);
    unsigned long int seedThis;
    for(int i=0; i <= rank; ++i)
      seedThis = gen();
    if(!rank || debug)
      std::cout << " rank " << rank << " initialized Kokkos rnd with seed "
                << seed << " => " << seedThis << "\n";
    rand_pool = Kokkos::Random_XorShift64_Pool<>(seedThis);
  }
}

void GitrmParticles::resetPtclWallCollisionData() {
  auto capacity = ptcls->capacity();
  wallCollisionPts_w = o::Write<o::Real>(3*capacity, 0, "wallXpts");
  wallCollisionFaceIds_w = o::Write<o::LO>(capacity, -1, "wallXfids");
}

void GitrmParticles::initPtclWallCollisionData() {
  resetPtclWallCollisionData();
}

void GitrmParticles::convertPtclWallCollisionData() {
  wallCollisionPts = o::Reals(wallCollisionPts_w);
  wallCollisionFaceIds = o::LOs(wallCollisionFaceIds_w);
}

void GitrmParticles::initPtclEndPoints() {
  OMEGA_H_CHECK(ptcls->nPtcls() == numInitPtcls);
  ptclEndPoints = o::Write<o::Real>(3*totalPtcls, 0, "ptclEndPoints");
}

//ptclsInElem: ptcls in global elem ids of picpart mesh, not full-mesh
void GitrmParticles::defineParticles(const o::LOs& ptclsInElem, int elId) {
  bool debug = 1;
  o::Int ne = picparts.nelems();
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  std::cout << rank << " : To define ptcls: ne " << ne << " nGids "
    << mesh_element_gids.size() << " picpart-nel " << picparts.nelems() << "\n";
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const o::LO& i) {
    element_gids(i) = mesh_element_gids[i];
  });
  if(elId>=0) { //only to init in single elem
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const o::LO& i) {
      ptcls_per_elem(i) = 0;
      if (i == elId) {
        ptcls_per_elem(i) = numInitPtcls;
        printf(" Ptcls in elId %d\n", elId);
      }
    });
  } else {
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const o::LO& i) {
      ptcls_per_elem(i) = ptclsInElem[i];
    });
  }
  if(debug > 2) {
    printf(" SCS ptcls/elem= rank: lid gid np\n");
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const o::LO& i) {
      const int np = ptcls_per_elem(i);
      o::GO gid = element_gids(i);
      if (np > 0)
        printf("%d: lid %d gid %ld np %d\n", rank, i, gid, np);
    });
    printf("\n");
  }
  //'sigma', 'V', and the 'policy' control the layout of the PS structure
  const int sigma = INT_MAX; //full sorting
  const int V = 128;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  printf("%d: Constructing %d Particles with sigma %d V %d\n",
    rank, numInitPtcls, sigma, V);
  //Create the particle structure
  ptcls = new SellCSigma<Particle>(policy, sigma, V, ne, numInitPtcls,
                   ptcls_per_elem, element_gids);
}

// for distributing particles among picparts. For the current non-overlapping
// partitions this is just selecting all particles that are found in this picpart
void GitrmParticles::assignParticles(const o::LOs& elemIdOfPtclsAll,
   o::LOs& numPtclsInElems, o::LOs& elemIdOfPtcls, o::LOs& ptclDataInds) {
  int commSize = this->commSize;
  int rank = this->rank;
  OMEGA_H_CHECK(totalPtcls > 0);

  int pBegin = 0;
  auto nPtcls = elemIdOfPtclsAll.size();

  //for distributed mesh, the case of particle read from file in chunks is not handled
  if(ptclSplitRead) {
    if(!isFullMesh)
      Omega_h_fail("The case of ptclSplitRead not handled for distributed mesh\n");
    int rest = totalPtcls%commSize;
    nPtcls = totalPtcls/commSize;
    //last one gets the rest !
    if(rank == commSize-1)
      nPtcls += rest;
    pBegin = rank * totalPtcls/commSize;
  }

  // most of these have no effect in the case of fullMesh with all ptcls read
  o::Write<o::LO> elemIdOfPtcls_w(nPtcls, -1, "elemIdOfPtcls");
  o::Write<o::LO> ptclDataInds_w(nPtcls, -1, "ptclDataInds");
  o::Write<o::LO> numPtclsInElems_w(picparts.nelems(), 0, "numPtclsInElems");
  o::Write<o::LO> validPtcls(nPtcls, 0, "validPtcls");
  o::LO offset = pBegin;
  auto lambda = OMEGA_H_LAMBDA(const o::LO& i) {
    auto el = elemIdOfPtclsAll[i];
    if(el>=0) {
      elemIdOfPtcls_w[i] = el;
      ptclDataInds_w[i] = offset+i;
      Kokkos::atomic_increment(&numPtclsInElems_w[el]);
      validPtcls[i] = 1;
    }
  };
  o::parallel_for(nPtcls, lambda, "assign_init_ptcls");
  elemIdOfPtcls = o::LOs(elemIdOfPtcls_w);
  ptclDataInds = o::LOs(ptclDataInds_w);
  numPtclsInElems = o::LOs(numPtclsInElems_w);
  //reset count of particles to be initialized
  numInitPtcls = o::get_sum(o::LOs(validPtcls));

  //verify
  if(isFullMesh)
    OMEGA_H_CHECK(numInitPtcls == nPtcls);

  const long int nPtcls_ = numInitPtcls;
  long int totalPtcls_ = 0;
  int collectRank = 0;
  MPI_Barrier(MPI_COMM_WORLD);//picparts.comm()->get_impl());
  MPI_Reduce(&nPtcls_, &totalPtcls_, 1, MPI_LONG, MPI_SUM, collectRank, MPI_COMM_WORLD);
//  if(!rank)
    printf("%d: isFullMesh %d Particles %d total %ld valid %ld\n",
       rank, isFullMesh, numInitPtcls, totalPtcls, totalPtcls_);
  if(rank == collectRank && mustFindAllPtcls)
    OMEGA_H_CHECK(totalPtcls_ == totalPtcls);
}

o::LOs findInvalidPtcls(o::Reals& data, int dof= 6) {
  int np = data.size()/dof;
  o::Write<o::LO> valids(np, 0, "validPtclIds");
  o::parallel_for(np, OMEGA_H_LAMBDA(const o::LO& ip) {
    bool stat = true;
    for(int j=0; j<6 && j<dof; ++j) {
      stat = (stat && !isnan(data[j*np+ip]));
    }
    valids[ip] = stat;
  });
  return o::LOs(valids);
}


void GitrmParticles::initPtclsFromFile(const std::string& fName,
   o::LO maxLoops, bool printSource) {
  if(!rank)
    std::cout << rank << ": Loading initial particle data from file: " << fName << " \n";
  o::HostWrite<o::Real> readInData_h;
  // TODO piscesLowFlux/updated/input/particleSource.cfg has r,z,angles, CDF, cylSymm=1
  //read total particles by all picparts, since parent element not known
  auto each_chunk = (ptclSplitRead) ? totalPtcls/commSize: totalPtcls;
  size_t each_chunk_pos = (ptclSplitRead) ? each_chunk : 0;
  if (rank == commSize-1 && ptclSplitRead)
    each_chunk = each_chunk + totalPtcls%commSize;

  //Reading all simulating particles in all ranks. replaceNaN false to skip those later
  bool replaceNaN = false;
  auto stat = readParticleSourceNcFile(fName, readInData_h, numPtclsRead,
     size_t(each_chunk), each_chunk_pos, replaceNaN);

  //OMEGA_H_CHECK( stat && (numPtclsRead >= totalPtcls));
  auto readInData_r = o::Reals(o::Write<o::Real>(readInData_h));
  o::LOs validPtcls;
  int dof = 6; //pos,vel
  if(!replaceNaN)
    validPtcls = findInvalidPtcls(readInData_r, dof);
  else
    validPtcls = o::LOs(o::Write<o::LO>(numPtclsRead, 1, "valids"));

  o::LOs elemIdOfPtcls;
  o::LOs ptclDataInds;
  o::LOs numPtclsInElems;
  if(!rank)
    printf("%d: partices read %d. Finding ElemIds of Ptcls \n", rank, numPtclsRead);
  findElemIdsOfPtclCoordsByAdjSearch(readInData_r, elemIdOfPtcls, ptclDataInds,
    numPtclsInElems, validPtcls);

  defineParticles(numPtclsInElems, -1);

  initPtclWallCollisionData();
  //note:rebuild to get mask if elem_ids changed
  if(!rank)
    printf("%d: Setting Ptcl InitCoords \n", rank);
  auto ptclIdPtrsOfElem = o::offset_scan(o::LOs(numPtclsInElems), "ptclIdPtrsOfElem");

  o::LOs ptclIdsInElem;
  convertInitPtclElemIdsToCSR(numPtclsInElems, ptclIdPtrsOfElem, elemIdOfPtcls,
    ptclIdsInElem);

  setPidsOfPtclsLoadedFromFile(ptclIdPtrsOfElem, ptclIdsInElem, elemIdOfPtcls,
    ptclDataInds);

  setPtclInitData(readInData_r);

  initPtclChargeIoniRecombData();
  initPtclSurfaceModelData();
  initPtclEndPoints();
}


bool GitrmParticles::searchPtclInAllElems(const o::Reals& data, const o::LO pind,
   o::LO& parentElem, o::LOs& validPtcls) {
  int debug = 0;
  if(debug > 2 && !rank)
    std::cout << rank << ": " << __FUNCTION__ << "\n";
  const int rank = this->rank;
  MESHDATA(mesh);
  auto nPtclsRead = numPtclsRead;

  if(pind >= nPtclsRead) {
    std::cout << rank << "*** Warning: skipping searching pind > nPtclsRead\n";
    parentElem = -1;
    return false;
  }

  auto owners = elemOwners;
  o::Write<o::LO> elemDet(1, -1, "elemDet_searchPtclInAllElems");
  auto lamb = OMEGA_H_LAMBDA(const o::LO& elem) {
    if(validPtcls[pind] > 0) {
      bool select = (rank == owners[elem]);
      if(select) {
        auto pos = o::zero_vector<3>();
        auto bcc = o::zero_vector<4>();
        for(int j=0; j<3; ++j)
          pos[j] = data[j*nPtclsRead+pind];
        if(p::isPointWithinElemTet(mesh2verts, coords, pos, elem, bcc)) {
          auto prev = Kokkos::atomic_exchange(&(elemDet[0]), elem);
          if(prev >= 0)
            printf("%d: InitParticle %d in rank %d was found in multiple elements %d %d \n",
              rank, pind, rank, prev, elem);
        }
      }
    }
  };
  o::parallel_for(picparts.nelems(), lamb, "search_parent_of_ptcl");
  auto result = o::HostWrite<o::LO>(elemDet);
  parentElem = result[0];
  return parentElem >= 0;
}

// totalPtcls are searched in each rank
o::LO GitrmParticles::searchAllPtclsInAllElems(const o::Reals& data,
   o::Write<o::LO>& elemIdOfPtcls, o::Write<o::LO>& numPtclsInElems, o::LOs& validPtcls) {
  printf("%d: %s \n", rank, __FUNCTION__);
  MESHDATA(mesh);
  int rank = this->rank;
  auto owners = elemOwners;
  auto nPtclsRead = numPtclsRead;
  auto nPtcls = elemIdOfPtcls.size();

  auto lamb = OMEGA_H_LAMBDA(const o::LO& elem) {
    bool select = (rank == owners[elem]);;
    if(select) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto tet = p::gatherVectors4x3(coords, tetv2v);
      auto pos = o::zero_vector<3>();
      for(auto pind=0; pind < nPtcls; ++pind) {
        if(validPtcls[pind] <= 0)
          continue;
        for(int j=0; j<3; ++j) {
          pos[j] = data[j*nPtclsRead+pind];
          auto vel = data[(j+3)*nPtclsRead+pind];
        }

        bool valid = true;
        auto bcc = o::zero_vector<4>();
        p::find_barycentric_tet(tet, pos, bcc);
        valid = (valid && p::all_positive(bcc, 1.0e-10));
        if(valid) {
          Kokkos::atomic_increment(&numPtclsInElems[elem]);
          Kokkos::atomic_exchange(&(elemIdOfPtcls[pind]), elem);
        }
      }
    }
  };
  o::parallel_for(picparts.nelems(), lamb, "search_parents_of_ptcls");
  return o::get_min(o::LOs(elemIdOfPtcls));
}

o::LO GitrmParticles::searchPtclsByAdjSearchFromParent(const o::Reals& data,
   const o::LO parentElem, o::Write<o::LO>& numPtclsInElemsAll,
   o::Write<o::LO>& elemIdOfPtclsAll, o::LOs& validPtcls) {
  printf("%d: %s \n", rank, __FUNCTION__);
  int debug = 0;
  int rank = this->rank;
  MESHDATA(mesh);
  auto owners = elemOwners;
  auto nPtclsRead = numPtclsRead;
  int maxSearch = 1000;

  //search all particles starting with this element
  auto lambda = OMEGA_H_LAMBDA(const o::LO& ip) {
  if(validPtcls[ip] > 0) {
      bool found = false;
      auto pos = o::zero_vector<3>();
      auto bcc = o::zero_vector<4>();
      o::LO elem = parentElem;
      int isearch = 0;
      while(!found) {
        auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
        auto tet = p::gatherVectors4x3(coords, tetv2v);
        for(int j=0; j<3; ++j) {
          pos[j] = data[j*nPtclsRead+ip];
        }
        p::find_barycentric_tet(tet, pos, bcc);

        bool select = (rank == owners[elem]);
        if(select && p::all_positive(bcc, 0)) {
          elemIdOfPtclsAll[ip] = elem;
          if(debug > 2)
            printf("%d: Final particle %d elid %d\n",rank, ip, elemIdOfPtclsAll[ip]);
          Kokkos::atomic_increment(&numPtclsInElemsAll[elem]);
          found = true;
        } else {
          o::LO minInd = p::min_index(bcc, 4);
          auto dual_elem_id = dual_faces[elem];
          o::LO findex = 0;

          int beg = elem*4;
          for(auto iface = beg; iface < beg+4; ++iface) {
            auto face_id = down_r2fs[iface];
            bool exposed = side_is_exposed[face_id];
            if(!exposed) {
              if(findex == minInd)
                elem = dual_elems[dual_elem_id];
              ++dual_elem_id;
            }
            ++findex;
          }
        }
        if(isearch > maxSearch) {
          if(debug > 1)
            printf("%d: Error finding particle %d\n", rank, ip);
          break;
        }
        ++isearch;
      }
    }
  };
  auto nPtcls = elemIdOfPtclsAll.size();
  o::parallel_for(nPtcls, lambda, "init_ptcl2");
  return o::get_min(o::LOs(elemIdOfPtclsAll));
}

void setDefaultOfInvalidPtcls(o::Write<o::LO>& elemIdOfPtclsAll, o::LOs& validPtcls, int invalidMin=-1) {
  o::parallel_for(elemIdOfPtclsAll.size(), OMEGA_H_LAMBDA(const o::LO& i) {
    if(validPtcls[i] <= 0)
      elemIdOfPtclsAll[i] = invalidMin;
  });
}


// search all particles read. TODO update this for partitioning
void GitrmParticles::findElemIdsOfPtclCoordsByAdjSearch(const o::Reals& data,
   o::LOs& elemIdOfPtcls, o::LOs& ptclDataInds, o::LOs& numPtclsInElems, o::LOs& validPtcls) {
  bool debug = true;
  int rank = this->rank;

  auto nPtcls = (numPtclsRead > totalPtcls) ? totalPtcls : numPtclsRead;
  //try one in first few particles
  int maxLoop = 20;
  maxLoop = (nPtcls > maxLoop) ? maxLoop : nPtcls;
  MESHDATA(mesh);
  o::Write<o::LO> elemDet(1, -1);
  o::LO parentElem = -1;
  o::LO pstart = 0;
  o::LO ptcl = -1;
  for(auto ip = pstart; ip < maxLoop; ++ip) {
    searchPtclInAllElems(data, ip, parentElem, validPtcls);
    if(parentElem >= 0) {
      ptcl = ip;
      break;
    }
  }
  if(debug && parentElem >=0 && !rank)
   printf("%d: Beginning Ptcl search: ptcl %d found in elem %d\n",
    rank, ptcl, parentElem);
  if(debug && parentElem < 0 && !rank)
   printf("%d: Parent elements not found for %d particles\n", rank, maxLoop);

  //find all particles starting from parent
  o::Write<o::LO> numPtclsInElemsAll(nel, 0, "numPtclsInElemsAll");
  int defMin = -2;
  int invalidMin = -1; // more than defMin, but less than valid elemId
  o::Write<o::LO> elemIdOfPtclsAll(nPtcls, defMin, "elemIdOfPtclsAll");
  setDefaultOfInvalidPtcls(elemIdOfPtclsAll, validPtcls, invalidMin);
  o::LO min = defMin;

  if(parentElem >= 0)
    min = searchPtclsByAdjSearchFromParent(data, parentElem, numPtclsInElemsAll,
      elemIdOfPtclsAll, validPtcls);

  std::cout << "min of searchPtclsByAdjSearchFromParent " << min << "\n";

  //if not all found, do brute-force search for all
  if(min == defMin && mustFindAllPtcls) //FIXME
    min = searchAllPtclsInAllElems(data, elemIdOfPtclsAll, numPtclsInElemsAll, validPtcls);
  if(debug && !rank)
    printf("%d: done adjacency search for particles\n",rank);

  if(debug > 3 && min < 0) {
    o::parallel_for(nPtcls, OMEGA_H_LAMBDA(const o::LO& i) {
      if(elemIdOfPtclsAll[i] < 0) {
        double v[6];
        for(int j=0; j<6; ++j)
          v[j] = data[j*nPtcls+i];
        printf("%d: NOTdet i %d %g %g %g :vel: %g %g %g\n",
          rank, i, v[0], v[1], v[2], v[3], v[4], v[5] );
      }
    },"kernel_findElemIdsOfPtclCoordsByAdjSearch");
  }

  if(isFullMesh)
    OMEGA_H_CHECK(min >=0);

  assignParticles(o::LOs(elemIdOfPtclsAll), numPtclsInElems, elemIdOfPtcls,
    ptclDataInds);
  if(debug && !rank)
    printf("%d: done assigning particles \n", rank);

  //TODO handle this
  numOfPtclsInElems = o::LOs(numPtclsInElems);
}


//TODO handle numPtclsInElems which was added as a member for this function call
void GitrmParticles::printNumPtclsInElems(const std::string& fname) {
  std::cout << "Printing Nums of ptcls in elements in " << fname << "\n";
  auto nps_h = o::HostWrite<o::LO>(o::deep_copy(numOfPtclsInElems));

  auto origGids = mesh.get_array<o::GO>(o::REGION, "origGids");
  auto origGids_h = o::HostWrite<o::GO>(o::deep_copy(origGids));

  auto file = fname + "_" + std::to_string(rank);
  std::ofstream ofs(file);
  for(int i=0; i<nps_h.size(); ++i) {
    auto gid = origGids_h[i];
    ofs << gid << " " << nps_h[i] << "\n";
  }
}

// using ptcl sequential numbers 0..numPtcls
void GitrmParticles::convertInitPtclElemIdsToCSR(const o::LOs& numPtclsInElems,
   const o::LOs& ptclIdPtrsOfElem, const o::LOs& elemIdOfPtcls, o::LOs& ptclIdsInElem) {
  o::LO debug = 0;
  auto nel = picparts.nelems();
  // csr data
  o::Write<o::LO> ptclIdsInElem_w(numInitPtcls, -1, "ptclIdsInElem");
  o::Write<o::LO> ptclsFilledInElem(nel, 0, "ptclsFilledInElem");
  int rank = this->rank;
  auto lambda = OMEGA_H_LAMBDA(const o::LO& id) {
    auto el = elemIdOfPtcls[id];
    if(el >= 0) {
      auto old = Kokkos::atomic_fetch_add(&(ptclsFilledInElem[el]), 1);
      auto nLimit = numPtclsInElems[el];
      OMEGA_H_CHECK(old < nLimit);
      //elemId is sequential from 0 .. nel
      auto beg = ptclIdPtrsOfElem[el];
      auto pos = beg + old;
      auto idLimit = ptclIdPtrsOfElem[el+1];
      OMEGA_H_CHECK(pos < idLimit);
      auto prev = Kokkos::atomic_exchange(&(ptclIdsInElem_w[pos]), id);
      if(debug && !rank)
        printf("%d: id:el %d %d old %d beg %d pos %d previd %d maxPtcls %d \n",
          rank, id, el, old, beg, pos, prev, ptclIdsInElem_w[id] );
    }
  };
  o::parallel_for(elemIdOfPtcls.size(), lambda, "Convert to CSR write");
  ptclIdsInElem = o::LOs(ptclIdsInElem_w);
}


void GitrmParticles::setPidsOfPtclsLoadedFromFile(const o::LOs& ptclIdPtrsOfElem,
  const o::LOs& ptclIdsInElem,  const o::LOs& elemIdOfPtcls, const o::LOs& ptclDataInds) {
  int debug = 0;
  int rank = this->rank;
  int pBegin = 0;// TODO rank*int(totalPtcls/commSize);
  //if(debug && !rank)
    printf("%d: numInitPtcls %d numPtclsRead %d \n", rank, numInitPtcls, numPtclsRead);
  auto nel = picparts.nelems();
  o::Write<o::LO> nextPtclInd(nel, 0, "nextPtclInd");
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto pid_ps_global=ptcls->get<PTCL_ID_GLOBAL>();
  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask > 0) {
      auto thisInd = Kokkos::atomic_fetch_add(&(nextPtclInd[elem]), 1);
      auto idPtr = ptclIdPtrsOfElem[elem];
      OMEGA_H_CHECK(idPtr >= 0);
      auto ind = idPtr + thisInd;
      auto limit = ptclIdPtrsOfElem[elem+1];
      //Set checks separately to avoid possible Error
      OMEGA_H_CHECK(ind >= 0);
      OMEGA_H_CHECK(ind < limit);
      const int ip_g = ptclIdsInElem[ind] + pBegin; // TODO
      const int ip = ptclIdsInElem[ind];
      if(debug > 2)
        printf("%d: **** elem %d pid %d ind %d thisInd %d nextInd %d indlim %d ip %d\n",
          rank, elem, pid, ind, thisInd, nextPtclInd[elem], ptclIdPtrsOfElem[elem+1], ip);
      OMEGA_H_CHECK(ip >= 0);
      pid_ps(pid)         = ip;
      pid_ps_global(pid)  = ip_g;
    }
  };
  p::parallel_for(ptcls, lambda, "setPidsOfPtcls");
}

// Construct PS with  #particles in each elem passed in, otherwise newly added
//  particles in originally empty elements won't show up in PS_LAMBDA iterations.
// ie. their mask will be 0. If mask is not used, invalid particles may
// show up from other threads in the launch group.
void GitrmParticles::setPtclInitData(const o::Reals& data) {
  int debug = 0;
  int rank = this->rank;
  if(debug && !rank)
    printf("%d: numPtclsRead %d \n", rank, numPtclsRead);
  auto npRead = numPtclsRead;
  const auto coords = mesh.coords();
  const auto mesh2verts = mesh.ask_elem_verts();
  auto next_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto pos_ps_d = ptcls->get<PTCL_POS>();
  auto vel_d = ptcls->get<PTCL_VEL>();
  auto pid_ps = ptcls->get<PTCL_ID>();

  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);
      auto bcc = o::zero_vector<4>();
      auto vel = o::zero_vector<3>();
      auto pos = o::zero_vector<3>();
      auto ip = pid_ps(pid);
      for(int i=0; i<3; ++i){
        pos[i] = data[i*npRead+ip];
      }
      for(int i=0; i<3; ++i)
        vel[i] = data[(3+i)*npRead+ip];
      if(debug>2)
        printf("Rank %d ip %d pid %d pos %g %g %g vel %g %g %g \n",rank, ip, pid,
         pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
      p::find_barycentric_tet(M, pos, bcc);
      OMEGA_H_CHECK(p::all_positive(bcc, 0));
      for(int i=0; i<3; i++) {
        pos_ps_d(pid,i) = pos[i];
        vel_d(pid, i) = vel[i];
        next_ps_d(pid,i) = 0;
      }
    }
  };
  p::parallel_for(ptcls, lambda, "setPtclInitData");
}

void GitrmParticles::initPtclChargeIoniRecombData() {
  auto charge_ps = ptcls->get<PTCL_CHARGE>();
  auto first_ionizeZ_ps = ptcls->get<PTCL_FIRST_IONIZEZ>();
  auto prev_ionize_ps = ptcls->get<PTCL_PREV_IONIZE>();
  auto first_ionizeT_ps = ptcls->get<PTCL_FIRST_IONIZET>();
  auto prev_recomb_ps = ptcls->get<PTCL_PREV_RECOMBINE>();

  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask > 0) {
      charge_ps(pid) = 0;
      first_ionizeZ_ps(pid) = 0;
      prev_ionize_ps(pid) = 0;
      first_ionizeT_ps(pid) = 0;
      prev_recomb_ps(pid) = 0;
    }
  };
  p::parallel_for(ptcls, lambda, "initPtclChargeIoniRecombData");
}

void GitrmParticles::initPtclSurfaceModelData() {
  auto ps_weight = ptcls->get<PTCL_WEIGHT>();
  auto ps_hitNum = ptcls->get<PTCL_HIT_NUM>();
  auto ps_newVelMag = ptcls->get<PTCL_VMAG_NEW>();
  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask > 0) {
      ps_weight(pid) = 1;
      ps_hitNum(pid) = 0;
      ps_newVelMag(pid) = 0;
    }
  };
  p::parallel_for(ptcls, lambda, "initPtclSurfaceModelData");
}

void GitrmParticles::initPtclsInADirection(o::Real theta, o::Real phi, o::Real r,
   o::LO maxLoops, o::Real outer) {
  int debug = 0;
  o::Write<o::LO> elemAndFace(3, -1);
  o::LO initEl = -1;
  findInitialBdryElemIdInADir(theta, phi, r, initEl, elemAndFace, maxLoops, outer);
  o::LOs temp;
  defineParticles(temp, initEl);

  //note:rebuild if particles to be added in new elems, or after emptying any elem.
  if(!rank && debug)
    printf("%d: Setting ImpurityPtcl InitCoords \n", rank);
  setPtclInitRndDistribution(elemAndFace);
}

void GitrmParticles::setPtclInitRndDistribution(o::Write<o::LO> &elemAndFace) {
  MESHDATA(mesh);
  int rank = this->rank;
  //Set particle coordinates. Initialized only on one face. TODO confirm this ?
  auto x_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto x_ps_prev_d = ptcls->get<PTCL_POS>();
  auto vel_d = ptcls->get<PTCL_VEL>();
  int psCapacity = ptcls->capacity();
  o::HostWrite<o::Real> rnd1(psCapacity, "rnd1");
  o::HostWrite<o::Real> rnd2(psCapacity, "rnd2");
  o::HostWrite<o::Real> rnd3(psCapacity, "rnd3");
  std::srand(time(NULL));
  for(auto i=0; i<psCapacity; ++i) {
    rnd1[i] = (o::Real)(std::rand())/RAND_MAX;
    rnd2[i] = (o::Real)(std::rand())/RAND_MAX;
    rnd3[i] = (o::Real)(std::rand())/RAND_MAX;
  }
  o::Reals rand1 = o::Reals(o::Write<o::Real>(rnd1));
  o::Reals rand2 = o::Reals(o::Write<o::Real>(rnd2));
  o::Reals rand3 = o::Reals(o::Write<o::Real>(rnd3));

  o::Write<o::LO> elem_ids(psCapacity,-1, "elem_ids");
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
      o::LO verbose =1;
      // TODO if more than an element  ?
      const auto faceId = elemAndFace[2];
      const auto fv2v = o::gather_verts<3>(face_verts, faceId);
      const auto face = p::gatherVectors3x3(coords, fv2v);
      auto fcent = p::face_centroid_of_tet(faceId, coords, face_verts);
      auto tcent = p::centroid_of_tet(elem, mesh2verts, coords);
      auto diff = tcent - fcent;
      if(verbose >3 && !rank)
        printf(" elemAndFace[1]:%d, elem:%d face%d beg%d\n",
          elemAndFace[1], elem, elemAndFace[2], elemAndFace[0]);

      o::Vector<4> bcc;
      o::Vector<3> pos;
      auto rn1 = rand1[pid];
      auto rn2 = rand2[pid];
      do {
        o::Real bc1 = (rn1 > rn2) ? rn2: rn1;
        o::Real bc2 = std::abs(rn1 - rn2);
        o::Real bc3 = 1.0 - bc1 - bc2;
        o::Vector<3> fpos = bc1*face[0] + bc2*face[1] + bc3*face[2];
        auto fnorm = p::face_normal_of_tet(faceId, elem, coords, mesh2verts,
                                        face_verts, down_r2fs);
        pos = fpos - 1.0e-6*fnorm;
        auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
        auto M = p::gatherVectors4x3(coords, tetv2v);
        p::find_barycentric_tet(M, pos, bcc);
        rn1 /= 2.0;
      } while(!p::all_positive(bcc, 0));

      double amu = 2.0; //TODO
      double energy[] = {4.0, 4, 4}; //TODO actual [4,0,0]
      double vel[] = {0,0,0};
      for(int i=0; i<3; i++) {
        x_ps_prev_d(pid,i) = pos[i];
        x_ps_d(pid,i) = pos[i];
        auto en = energy[i];
        //if(! o::are_close(energy[i], 0))
        vel[i] = std::sqrt(2.0 * abs(en) * 1.60217662e-19 / (amu * 1.6737236e-27));
        vel[i] *= rand3[pid];  //TODO
      }

      for(int i=0; i<3; i++)
        vel_d(pid, i) = vel[i];

      elem_ids[pid] = elem;

      if(verbose >2 && !rank)
        printf("elm %d : pos %.4f %.4f %.4f : vel %.1f %.1f %.1f Mask%d\n",
          elem, x_ps_prev_d(pid,0), x_ps_prev_d(pid,1), x_ps_prev_d(pid,2),
          vel[0], vel[1], vel[2], mask);
    }
  };
  p::parallel_for(ptcls, lambda, "setPtclInitRndDistribution");
}

// spherical coordinates (wikipedia), radius r=1.5m, inclination theta[0,pi] from the z dir,
// azimuth angle phi[0, 2π) from the Cartesian x-axis (so that the y-axis has phi = +90°).
void GitrmParticles::findInitialBdryElemIdInADir(o::Real theta, o::Real phi, o::Real r,
     o::LO &initEl, o::Write<o::LO> &elemAndFace, o::LO maxLoops, o::Real outer){
  int rank = this->rank;
  o::LO debug = 4;
  MESHDATA(mesh);

  theta = theta * o::PI / 180.0;
  phi = phi * o::PI / 180.0;

  const o::Real x = r * sin(theta) * cos(phi);
  const o::Real y = r * sin(theta) * sin(phi);
  const o::Real z = r * cos(theta);

  o::Real endR = r + outer; //meter, to be outside of the domain
  const o::Real xe = endR * sin(theta) * cos(phi);
  const o::Real ye = endR * sin(theta) * sin(phi);
  const o::Real ze = endR * cos(theta);
  if(!rank)
    printf("Direction:x,y,z: %f %f %f\n xe,ye,ze: %f %f %f\n", x,y,z, xe,ye,ze);

  // Beginning element id of this x,y,z
  auto lamb = OMEGA_H_LAMBDA(const o::LO& elem) {
    auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
    auto M = p::gatherVectors4x3(coords, tetv2v);

    o::Vector<3> orig;
    orig[0] = x;
    orig[1] = y;
    orig[2] = z;
    o::Vector<4> bcc;
    p::find_barycentric_tet(M, orig, bcc);
    if(p::all_positive(bcc, 0)) {
      elemAndFace[0] = elem;
      if(debug > 3 && !rank)
        printf(" ORIGIN detected in elem %d \n", elem);
    }
  };
  o::parallel_for(picparts.nelems(), lamb, "init_impurity_ptcl1");
  Kokkos::fence();
  o::HostRead<o::LO> elemId_bh(elemAndFace);
  if(!rank)
    printf(" ELEM_beg %d \n", elemId_bh[0]);

  if(elemId_bh[0] < 0) {
    Omega_h_fail("Failed finding initial element in given direction\n");
  }

  // Search final elemAndFace on bdry, on 1 thread on device(issue [] on host)
  o::Write<o::Real> xpt(3, -1);
  auto lamb2 = OMEGA_H_LAMBDA(const o::LO& e) {
    auto elem = elemAndFace[0];
    o::Vector<3> dest;
    dest[0] = xe;
    dest[1] = ye;
    dest[2] = ze;
    o::Vector<3> orig;
    orig[0] = x;
    orig[1] = y;
    orig[2] = z;
    o::Vector<4> bcc;
    bool found = false;
    o::LO loops = 0;

    while (!found) {
      if(debug > 4 && !rank)
        printf("\n****ELEM %d : ", elem);

      // Destination should be outisde domain
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);

      p::find_barycentric_tet(M, dest, bcc);
      if(p::all_positive(bcc, 0)) {
        printf("Wrong guess of destination in initImpurityPtcls");
        OMEGA_H_CHECK(false);
      }

      // Start search
      auto dual_elem_id = dual_faces[elem];
      const auto beg_face = elem *4;
      const auto end_face = beg_face +4;
      o::LO fIndex = 0;

      for(auto iface = beg_face; iface < end_face; ++iface) {
        const auto face_id = down_r2fs[iface];

        o::Vector<3> xpoint = o::zero_vector<3>();
        const auto face = p::get_face_coords_of_tet(mesh2verts, coords, elem, fIndex);
        o::Real dproj = 0;
        bool detected = p::line_triangle_intx_simple(face, orig, dest, xpoint, dproj);
        if(debug > 4 && !rank) {
          printf("iface %d faceid %d detected %d\n", iface, face_id, detected);
        }

        if(detected && side_is_exposed[face_id]) {
          found = true;
          elemAndFace[1] = elem;
          elemAndFace[2] = face_id;

          for(o::LO i=0; i<3; ++i)
            xpt[i] = xpoint[i];

          if(debug && !rank) {
            printf(" faceid %d detected on exposed\n",  face_id);
          }
          break;
        } else if(detected && !side_is_exposed[face_id]) {
          auto adj_elem  = dual_elems[dual_elem_id];
          elem = adj_elem;
          if(debug >4 && !rank) {
            printf(" faceid %d detected on interior; next elm %d\n", face_id, elem);
          }
          break;
        }
        if(!side_is_exposed[face_id]){
          ++dual_elem_id;
        }
        ++fIndex;
      } // faces

      if(loops > maxLoops) {
        if(!rank)
          printf("Tried maxLoops iterations in initImpurityPtcls");
        OMEGA_H_CHECK(false);
      }
      ++loops;
    }
  };
  o::parallel_for(1, lamb2, "init_impurity_ptcl2");
  Kokkos::fence();
  o::HostRead<o::Real> xpt_h(xpt);

  o::HostRead<o::LO> elemId_fh(elemAndFace);
  initEl = elemId_fh[1];
  if(!rank)
    printf(" ELEM_final %d xpt: %.3f %.3f %.3f\n\n", elemId_fh[1], xpt_h[0], xpt_h[1], xpt_h[2]);
  OMEGA_H_CHECK((initEl>=0) && (elemId_fh[0]>=0));
}

void GitrmParticles::initPtclsOnGeometricIdSurfaces() {
}

// Read GITR particle step data of all time steps; eg: rand numbers.
int GitrmParticles::readGITRPtclStepDataNcFile(const std::string& ncFileName,
  int& maxNPtcls, bool debug) {
  OMEGA_H_CHECK(useGitrRndNums == 1);
  if(!rank)
    std::cout << "Reading Test GITR step data : " << ncFileName << "\n";
  OMEGA_H_CHECK(!ncFileName.empty());
  // re-order the list in its constructor to leave out empty {}
  Field3StructInput fs({"intermediate"}, {}, {"nP", "nTRun", "dof"}, 0,
    {"RndIoni_at", "RndRecomb_at", "RndCollision_n1_at", "RndCollision_n2_at",
     "RndCollision_xsi_at", "RndCrossField_at", "RndReflection_at",
     "Opt_IoniRecomb", "Opt_Diffusion", "Opt_Collision", "Opt_SurfaceModel"});
  auto stat = readInputDataNcFileFS3(ncFileName, fs, maxNPtcls, numPtclsRead,
      "nP", debug);
  printf("Reading done %d\n",stat);
  testGitrPtclStepData = o::Reals(fs.data);
  testGitrDataIoniRandInd = fs.getIntValueOf("RndIoni_at");
  testGitrDataRecRandInd = fs.getIntValueOf("RndRecomb_at");
  testGitrStepDataDof = fs.getIntValueOf("dof"); // or fs.getNumGrids(2);
  testGitrStepDataNumTsteps = fs.getIntValueOf("nTRun");
  testGitrStepDataNumPtcls = fs.getIntValueOf("nP");
  testGitrCrossFieldDiffRndInd = fs.getIntValueOf("RndCrossField_at");
  testGitrCollisionRndn1Ind = fs.getIntValueOf("RndCollision_n1_at");
  testGitrCollisionRndn2Ind = fs.getIntValueOf("RndCollision_n2_at");
  testGitrCollisionRndxsiInd = fs.getIntValueOf("RndCollision_xsi_at");
  testGitrReflectionRndInd = fs.getIntValueOf("RndReflection_at");
  testGitrOptIoniRec = fs.getIntValueOf("Opt_IoniRecomb");
  testGitrOptDiffusion =  fs.getIntValueOf("Opt_Diffusion");
  testGitrOptCollision =  fs.getIntValueOf("Opt_Collision");
  testGitrOptSurfaceModel = fs.getIntValueOf("Opt_SurfaceModel");
  if(!rank)
    printf(" TestGITRdata: dof %d nT %d nP %d Index: rndIoni %d rndRec %d \n"
      " rndCrossFieldDiff %d rndColl_n1 %d rndColl_n2 %d rndColl_xsi %d "
      " rndReflection %d\n GITR run Flags: ioniRec %d diffusion %d "
      " collision %d surfmodel %d \n", testGitrStepDataDof, testGitrStepDataNumTsteps,
      testGitrStepDataNumPtcls, testGitrDataIoniRandInd, testGitrDataRecRandInd,
      testGitrCrossFieldDiffRndInd, testGitrCollisionRndn1Ind,
      testGitrCollisionRndn2Ind, testGitrCollisionRndxsiInd,
      testGitrReflectionRndInd, testGitrOptIoniRec, testGitrOptDiffusion,
      testGitrOptCollision, testGitrOptSurfaceModel);

  return stat;
}

//timestep >0
void GitrmParticles::checkCompatibilityWithGITRflags(int timestep) {
  if(timestep==0 && !rank)
    printf("ERROR: random number GITR data doesn't match with flags \n");
  OMEGA_H_CHECK(timestep>0);
  if(ranIonization||ranRecombination)
    OMEGA_H_CHECK(testGitrOptIoniRec);
  else
    OMEGA_H_CHECK(!testGitrOptIoniRec);

  if(ranCoulombCollision)
    OMEGA_H_CHECK(testGitrOptCollision);
  else
    OMEGA_H_CHECK(!testGitrOptCollision);

  if(ranDiffusion)
    OMEGA_H_CHECK(testGitrOptDiffusion);
  else
    OMEGA_H_CHECK(!testGitrOptDiffusion);
  if(ranSurfaceReflection)
    OMEGA_H_CHECK(testGitrOptSurfaceModel);
  else
    OMEGA_H_CHECK(!testGitrOptSurfaceModel);
}


namespace gitrm {

template<typename T>
o::Write<T> extractDataColumns (const o::Write<T>& data, const int nRows,
 const o::LOs& cols, const std::string& name="") {
  auto nCols = data.size()/nRows;
  auto nc = cols.size();
  //printf("nc %d \n", nc);
  //o::Write<T> d(nRows*nCols, 0, name);
  o::Write<T> d(nRows*nc, 0, name);
  //std::cout << name << "\n";
  o::parallel_for(nRows, OMEGA_H_LAMBDA(const int& i) {
    for(int j=0; j<nc; ++j) {
      auto col = cols[j];
      d[i*nc+j] = data[i*nCols + col];
      //printf(" %d %d %f \n",  i, j, d[i*nc+j]);
    }
  });
  return d;
}

template<typename T>
o::Write<T> sumDataColumns (const o::Write<T>& data, const int nRows,
 const o::LOs& cols, const std::string& name="") {
  auto nCols = data.size()/nRows;
  auto nc = cols.size();
  //printf("nc %d \n", nc);
  o::Write<T> d(nRows*nCols, 0, name);
  o::Write<T> sum(nRows, 0, name);
  //std::cout << name << "\n";
  o::parallel_for(nRows, OMEGA_H_LAMBDA(const int& i) {
    o::Real tot=0;
    for(int j=0; j<nc; ++j) {
      auto col = cols[j];
      d[i*nc+j] = data[i*nCols + col];
      tot=tot+d[i*nc+j];
    }
    sum[i]=tot;
    //printf("Sum is %f \n",sum[i]);
  });
  return sum;
}

template<typename T>
 o::Write<T> interp1D (const o::Write<T>& xdata, const o::Write<T>& ydata, const o::Write<T>& xpoints) {


  /* EXAMPLE
  Kokkos::parallel_reduce(dataTot_d.size(), OMEGA_H_LAMBDA(const int i, o::LO& lsum) {
    lsum += dataTot_d[i];
  }, total);
  */
  double xsum =0.0;
  double x2sum=0.0;
  double ysum =0.0;
  double xysum=0.0;

  Kokkos::parallel_reduce(xdata.size(), OMEGA_H_LAMBDA(const int i, o::Real& lsum1) {
    lsum1 += xdata[i];
  }, xsum);

  Kokkos::parallel_reduce(xdata.size(), OMEGA_H_LAMBDA(const int i, o::Real& lsum2) {
    lsum2 += xdata[i]*xdata[i];
  }, x2sum);


  Kokkos::parallel_reduce(xdata.size(), OMEGA_H_LAMBDA(const int i, o::Real& lsum3) {
    lsum3 += ydata[i];
  }, ysum);

  Kokkos::parallel_reduce(xdata.size(), OMEGA_H_LAMBDA(const int i, o::Real& lsum4) {
    lsum4 += xdata[i]*ydata[i];
  }, xysum);


  int n=xdata.size();
  double b=(n*xysum-xsum*ysum)/(n*x2sum-xsum*xsum);
  double a=(ysum/n)-(b/n)*xsum;



  auto sz=xpoints.size();
  o::Write<T> ypoints(sz, 0);
  o::parallel_for(sz, OMEGA_H_LAMBDA(const int& i) {
    ypoints[i]=a+b*xpoints[i] ;
    //printf("Ypoints %f\n", ypoints[i]);
  });

 return ypoints;
 }
}


void GitrmParticles::initPtclsOnGeometricBoundarys(const GitrmMesh& gm) {
  std::cout << __FUNCTION__ << "\n";
  //TODO get from config


  // GETTING MAGNETIC FIELD
  const auto& BField_2d = gm.getBfield2d();
  const auto bX0 = gm.bGridX0;
  const auto bZ0 = gm.bGridZ0;
  const auto bDx = gm.bGridDx;
  const auto bDz = gm.bGridDz;
  const auto bGridNx = gm.bGridNx;
  const auto bGridNz = gm.bGridNz;

  const int rank=this->rank;
  const int comm_size=this->commSize;
  auto owners=elemOwners;


  //GET SOLPS DATA
  o::HostWrite<o::Real> solps_h{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  o::LO nCols = 5;
  o::Write<o::Real> solps(solps_h.write());
  auto nRows = solps.size() / nCols;

  //rmrs = SOLPS[:,0] //1st column
  //r = SOLPS[:,1] //2nd column
  //z = SOLPS[:,2] //3rd column
  o::HostWrite<o::LO> rmrsCols_h{0}; //TODO get these from input
  auto rmrs = gitrm::extractDataColumns(solps, nRows, rmrsCols_h.write(), "rmrs_solps");
  o::HostWrite<o::LO> rCols_h{1}; //TODO get these from input
  auto rSolps = gitrm::extractDataColumns(solps, nRows, rCols_h.write(), "r_solps");
  o::HostWrite<o::LO> zCols_h{2}; //TODO get these from input
  auto zSolps = gitrm::extractDataColumns(solps, nRows, zCols_h.write(), "z_solps");


  //get flux indices
  //fluxInds = [28,30,32,33,35,36,37,38,40,41,42,43,44,45,46,47,48,49]
  o::HostWrite<o::LO> fluxInds_h{2,3,4};
  o::Write<o::LO> fluxInds(fluxInds_h.write());
  //selected solps flux for columns in fluxInds
  //fi  = SOLPS[:,fluxInds] //these columns
  //sum along columns, giving array of same number of entries as rows as SOLPS
  //[.....] column vector of 36 entries
  //reshape to (36,1) .  [[x1], [x2], ...[xn]] 36 rows, 1 column
  //spFlux = np.reshape(np.sum(fi,axis=1),(36,1)) // spFlux = 0*spyl zeros of same shape as spyl

  auto nflux = fluxInds.size();
  o::Write<o::Real> sFluxAll(nRows*nflux, 0, "sFlux");
  o::Write<o::Real> sFlux(nRows, 0, "sFlux");
  OMEGA_H_CHECK(nflux <= nCols);
  o::parallel_for(nRows, OMEGA_H_LAMBDA(const int& i) {
    o::Real tot = 0;
    for(int j=0; j<nflux; ++j) {
      auto fi = fluxInds[j];
      sFluxAll[i*nflux + j] = solps[i*nCols+fi];
      tot += abs(solps[i*nCols+fi]);
    }
    sFlux[i] = tot;
    //printf("Sumflux is %f \n", sFlux[i]);
  });

  //get sputter field file data
  //spyl = np.loadtxt(spylFile)
  o::HostWrite<o::Real> sputYld_h{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8};

  o::Write<o::Real> sputYld(sputYld_h.write());
  //spFlux = np.reshape(np.sum(fi,axis=1),(36,1))*spyl
  //result in shape (len(spyl), 36)
  //creating sputter flux of shape nRows x nSpYld
  auto nSpYld = sputYld.size();
  o::Write<o::Real> spFlux(nRows*nSpYld, 0, "spFlux");
  o::parallel_for(nRows, OMEGA_H_LAMBDA(const int& i) {
    for(int j=0; j<nSpYld; ++j) {
      spFlux[i*nSpYld+j] = sFlux[i]*sputYld[j];
      //printf("Sputter Flux:%f \n ",spFlux[i*nSpYld+j]);
    }
  });
  //std::cout << "got spFlux \n";
  auto nSpFlux = spFlux.size();


  //heTotal = np.sum(spFlux[:,2:4],axis=1)
  //get helium columns

  o::HostWrite<o::LO> helCols_h{2,3,4};//{2,3}; //TODO get these from input
  auto helFlux = gitrm::sumDataColumns(spFlux, nRows, helCols_h.write(), "HelTotal");

  //beTotal = np.sum(spFlux[:,4:8],axis=1)
  o::HostWrite<o::LO> beCols_h{4,5,6,7};//{4,5,6,7};
  auto beFlux = gitrm::sumDataColumns(spFlux, nRows, beCols_h.write(), "BeTotal");
  //neTotal = np.sum(spFlux[:,8:18],axis=1)
  o::HostWrite<o::LO> neCols_h{5,6,7};
  auto neFlux = gitrm::sumDataColumns(spFlux, nRows, neCols_h.write(), "NeTotal");


  //total flux
  o::HostWrite<o::LO> first2cols_h{0,1};
  o::Write<o::Real> totalFlux(nRows, 0, "totalFlux");
  auto spFlux2 = gitrm::sumDataColumns(spFlux, nRows, first2cols_h.write(), "spFlux2");
  o::parallel_for(nRows, OMEGA_H_LAMBDA(const o::LO& i) {
    totalFlux[i] = spFlux2[i] + helFlux[i] + neFlux[i];
    //printf("total flux %f \n", totalFlux[i]);
  });


  //Interpolation is between zSolps and total flux to find the fitting equation
  //The points at which the y point cordiantes are to be found are the z coordinate centroids
  //Finding the  z-coords of the mesh faces of the geometric face

  o::HostWrite<o::LO> ptclInitModelIds{138}; //PISCES

  MESHDATA(mesh);
  const auto f2r_ptr = mesh.ask_up(o::FACE, o::REGION).a2ab;
  const auto f2r_elem = mesh.ask_up(o::FACE, o::REGION).ab2b;

  //FInding the total no of bdry faces in the suface used for particle generation
  auto geomIds = o::LOs(ptclInitModelIds);
  o::Write<o::LO> ptclBdries_all(mesh.nfaces(), 0, "ptclInitBdryFaces");
  auto size=ptclBdries_all.size();
  printf("Rank %d Ptcl_Bdries_All Size %d \n",rank, size);
  gitrm::markBdryFacesOnGeomModelIds(mesh, geomIds, ptclBdries_all, 1, false);


  o::Write<o::LO> ptclBdries(mesh.nfaces(), 0, "ptclInitBdryFaces");
  o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {

    if (ptclBdries_all[fid]==1){
      auto elem = p::elem_id_of_bdry_face_of_tet(fid, f2r_ptr, f2r_elem);
        if (rank==owners[elem])
          ptclBdries[fid]=1;
    }
  });

  auto scan_r_all=o::offset_scan(o::LOs(ptclBdries_all));
  auto nsrc_all=o::get_sum(o::LOs(ptclBdries_all));
  auto scan_r=o::offset_scan(o::LOs(ptclBdries));
  auto nsrc=o::get_sum(o::LOs(ptclBdries));


  printf("RANK %d NSRC_total %d NsRC_picpart %d \n", rank, nsrc_all, nsrc);




  o::Write<o::Real> areas_w(mesh.nfaces(), 0, "areasBFaces");
  o::Write<o::Real> z_cent (mesh.nfaces(),0, "z_centroids");
  //FInding areas and z_centroids of the mesh triangular elements. WIll include zerso in between.
  o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {

    if (ptclBdries[fid]==1){
      //printf("nSrc is %d \n", nsrc);
      //printf("nSCAN is %d \n", scan_r.size());
      //printf("last of o_scan is %d \n", scan_r[scan_r.size()-1]);
      auto elem = p::elem_id_of_bdry_face_of_tet(fid, f2r_ptr, f2r_elem);
      const auto fv2v = o::gather_verts<3>(face_verts, fid);
      const auto face = p::gatherVectors3x3(coords, fv2v);
      auto cross = 1/2.0 * o::cross(face[1] - face[0], face[2] - face[0]);
      auto fcent = p::face_centroid_of_tet(fid, coords, face_verts);
      areas_w[fid] = o::inner_product(o::normalize(cross), cross);
      z_cent[fid]=fcent[2];
    }
  });


  //Removing 0 entries i.e. taking the array of elements that belong to only the geometric id --mesh faces.
   o::Write<o::Real> z_cent_trimmed(nsrc,0, "z_centroids_trimmed");
   o::Write<o::Real> area_trimmed(nsrc,0, "area_trimmed");
   o::Write<o::Real> surface_particle_rate(nsrc,0, "surface_particle_rate");
   o::Write<o::Real> particle_surf_track(mesh.nfaces(),0, "particle_track_id1");

   o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {
    if (ptclBdries[fid]){
      z_cent_trimmed[scan_r[fid]] = z_cent[fid];
      area_trimmed[scan_r[fid]] = areas_w[fid];
      particle_surf_track[fid]=fid; //TRACKING====================================1
    }
   });

  //printf("Going for 1d linear fit \n");
  //o::Write<o::Real> zSolps{-23, -2, 41}; ///For verification of 1d fit
  auto ypoints=gitrm::interp1D(zSolps, totalFlux, z_cent_trimmed);

  o::Write<o::Real> surface_rate_track(mesh.nfaces(),0, "particle_track_id2");
  o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {
    if (ptclBdries[fid]){
      surface_particle_rate[scan_r[fid]] = area_trimmed[scan_r[fid]]*ypoints[scan_r[fid]];

      if (surface_particle_rate[scan_r[fid]]>0)
      surface_rate_track[fid]=fid; //TRACKING======================================2
    }

  });


  //Finding the final surfaces
    o::Write<o::LO> final_surfaces(mesh.nfaces(),0, "final_surfaces");
    o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {
    if (surface_rate_track[fid] && particle_surf_track[fid] ){
      final_surfaces[fid]=1; //Eroded surfaces??
    }
  });

  auto size1xx=o::get_sum(o::LOs(final_surfaces));
  auto final_surf_offset=o::offset_scan(o::LOs(final_surfaces));
  o::Write<o::LO> final_surfaces_trimmed(size1xx, 0, "final_surfaces");
  printf("No of final surfaces %d rank %d \n", size1xx, rank);
  o::parallel_for(mesh.nfaces(), OMEGA_H_LAMBDA(const o::LO& fid) {
    if (final_surfaces[fid]){
      final_surfaces_trimmed[final_surf_offset[fid]]=fid; //Eroded surfaces              //IMPORTANT NO 1
      //printf("Final surfaces %d \n",final_surfaces_trimmed[final_surf_offset[fid]]);
    }
  });


  // Finding surface particle rates with values greater than 0
  o::Write<o::LO> surface_particle_rate_trimmed_index(nsrc,0, "surface_particle_rate_trimmed_index");
  o::parallel_for(nsrc, OMEGA_H_LAMBDA(const o::LO& fid) {
    if (surface_particle_rate[fid]>0)
    surface_particle_rate_trimmed_index[fid]=1;
  });

  // get the number and array cumsum of surface_particle_rate array_trimmed

  auto surface_particle_rate_sum=o::get_sum(o::LOs(surface_particle_rate_trimmed_index));
  auto surface_particle_rate_scan_r=o::offset_scan(o::LOs(surface_particle_rate_trimmed_index));

  o::Write<o::Real> surface_particle_rate_trimmed(surface_particle_rate_sum,0, "surface_particle_rate_trimmed");


  o::parallel_for(nsrc, OMEGA_H_LAMBDA(const o::LO& fid) {
   if (surface_particle_rate_trimmed_index[fid]){
    surface_particle_rate_trimmed[surface_particle_rate_scan_r[fid]] = surface_particle_rate[fid];//IMPORTANT NO 2
   }
  });

  //This array above just caluclated is the trimmed (>0) particle_rate array
  //Problem is cannot do offset_scan for cumsum as it's a float
  //So going for inclusive scan but it will contain one more lemnt than the parent element.
  //the initial element of the array is iitilaizedto 0 so no need to change
  o::Write<o::Real> out(surface_particle_rate_trimmed.size() + 1, 0, "surface_particle_rate_trimmed_offset");
  printf("Surface paarticle rate greter 0 == rank %d %d \n",rank, surface_particle_rate_trimmed.size());
  o::inclusive_scan(surface_particle_rate_trimmed.begin(), surface_particle_rate_trimmed.end(), out.begin()+1);

  o::Write<o::Real> particleCDF(surface_particle_rate_trimmed.size(), 0, "surface_particle_rate_trimmed_offset");
  o::parallel_for(surface_particle_rate_trimmed.size(), OMEGA_H_LAMBDA(const o::LO& fid) {

    particleCDF[fid]=out[fid+1]/out[surface_particle_rate_trimmed.size()];
    //printf("particleCDF %f \n", particleCDF[fid]);

  });

  printf("particleCDF_size rank %d: %d \n", rank, particleCDF.size());
  //GENERATE_PARTICLE_PART
  Kokkos::Random_XorShift64_Pool<> randy_pool;
  randy_pool=Kokkos::Random_XorShift64_Pool<>(102);



  int temp = nsrc*totalPtcls;
  int np_create = temp/nsrc_all;
  if (rank==comm_size-1)
    np_create=np_create ;
  printf("RANK %d np_create %d \n", rank, np_create);

  /*
  o::Write<o::Real> rand_ptcl_gen(np_create, 0,"Random numbers");
  o::Write<o::Real> rand_ptcl_gen2(np_create, 0,"Random numbers");
  o::Write<o::Real> rand_ptcl_gen3(np_create, 0,"Random numbers");
  o::Write<o::Real> rand_ptcl_gen4(np_create, 0,"Random numbers");
  */
  o::Write<o::Real> diff(np_create*particleCDF.size(), 0,"differences");
  o::Write<o::LO> mins(np_create, 0,"Minimum indices");

  /*
  o::parallel_for(np_create, OMEGA_H_LAMBDA(const o::LO& ptid) {
    auto state=randy_pool.get_state();
    rand_ptcl_gen[ptid]=state.drand();
    rand_ptcl_gen2[ptid]=state.drand();
    rand_ptcl_gen3[ptid]=state.drand();
    rand_ptcl_gen4[ptid]=state.drand();
    //printf("PTCL_RANDOM_NUMBER IS %f \n",rand_ptcl_gen[ptid]);
    randy_pool.free_state(state);
  });
  */


  o::parallel_for(np_create, OMEGA_H_LAMBDA(const o::LO& ptid) {
  int min;


    auto state=randy_pool.get_state();
    auto rand_ptcl_gen=state.drand();
    randy_pool.free_state(state);

    for (int i=0; i<particleCDF.size();i++){

       diff[ptid*particleCDF.size()+i]=particleCDF[i]-rand_ptcl_gen;
          if (diff[ptid*particleCDF.size()+i]<0)
       diff[ptid*particleCDF.size()+i]=100;
    }

    min=ptid*particleCDF.size();
    for (int i=0; i<particleCDF.size()-1; i++){
      if (diff[ptid*particleCDF.size()+i+1]<diff[min])
        min=ptid*particleCDF.size()+i+1;

      if(ptid==0 && false ){
        printf("value is %f \n", diff[ptid*particleCDF.size()+i]);
        printf("Min %d index %d\n", min, i);
      }

    }
    mins[ptid]=min-ptid*particleCDF.size(); //Surface id for the particle

  });


  //o::Write<o::Real> abcd_d(4*np_create, 0, "abcd");
  //o::Write<o::Real> planeNorm_d(np_create, 0, "planeNormal");
  //o::Write<o::Real> pos_created(3*np_create,0,"particlePositions");
  o::Write<o::Real> ptcl_created_data_w(6*np_create,0,"particleCreateddata");
  printf("Created ptcl_created_data with size %d rank %d \n", rank, ptcl_created_data_w.size());


    o::LO n_E=5;
    o::LO n_phi=5;
    o::LO n_theta=5;
    o::LO n_Loc=3;

    o::HostWrite<o::Real>phiGrid_h{0.25, 0.45, 0.2, 0.2, 0.6};     //READ THEM FROM FILES is a TODO
    o::HostWrite<o::Real>thetaGrid_h{0.125, 0.445, 0.70, 0.56, 0.908};
    o::HostWrite<o::Real>EGrid_h{0.25, 0.45, 0.25, 0.75, 0.4};


    o::Write<o::Real> phiGrid(phiGrid_h.write());
    o::Write<o::Real> thetaGrid(thetaGrid_h.write());
    o::Write<o::Real> EGrid(EGrid_h.write());


    o::HostWrite<o::Real>EDist_h(n_Loc*n_E);
    o::HostWrite<o::Real>thetaDist_h(n_Loc*n_theta);
    o::HostWrite<o::Real>phiDist_h(n_Loc*n_phi);

    for (int i=0; i<n_phi; ++i){
      for (int j=0; j<n_Loc; ++j){

        EDist_h[i*n_Loc+j]=i*n_Loc+j*n_phi;
        thetaDist_h[i*n_Loc+j]=2+i+j;
        phiDist_h[i*n_Loc+j]=2-i-j;
      }
    }
    o::Write<o::Real> thetaDist(thetaDist_h.write());
    o::Write<o::Real> phiDist(phiDist_h.write());
    o::Write<o::Real> EDist(EDist_h.write());



    //LIKE AN INCLUSIVE SCAN OVER THE 2D ARRAY TO FIND 2D CUMSUM
    // eCDF = np.cumsum(Edist,axis=0) Indicates sum over rows for each column
    o::Write<o::Real> thetaCDF(thetaDist.size(), 0, "thetaCDF");
    o::parallel_for(n_Loc, OMEGA_H_LAMBDA(const o::LO& fid) {

      for(int i=0; i<n_theta;i++){

        if(i==0)

          thetaCDF[fid+i*n_Loc]=thetaDist[fid+i*n_Loc];
        else

          thetaCDF[fid+i*n_Loc]=thetaCDF[fid+(i-1)*n_Loc] + thetaDist[fid+i*n_Loc];

        //if (fid==1) printf("THETA_ROW_WISE_CDF fid %d %f \n", fid, thetaCDF[fid+i*n_Loc]);
      }

    });


    o::Write<o::Real> phiCDF(phiDist.size(), 0, "phiCDF");
    o::parallel_for(n_Loc, OMEGA_H_LAMBDA(const o::LO& fid) {

      for(int i=0; i<n_phi;i++){

        if(i==0)

          phiCDF[fid+i*n_Loc]=phiDist[fid+i*n_Loc];
        else

          phiCDF[fid+i*n_Loc]=phiCDF[fid+(i-1)*n_Loc] + phiDist[fid+i*n_Loc];

        //if (fid==0) printf("PHI_ROW_WISE_CDF fid %d %f \n", fid, thetaCDF[fid+i*n_Loc]);
      }

    });


    o::Write<o::Real> ECDF(EDist.size(), 0, "ECDF");
    o::parallel_for(n_Loc, OMEGA_H_LAMBDA(const o::LO& fid) {

      for(int i=0; i<n_E;i++){

        if(i==0)

          ECDF[fid+i*n_Loc]=EDist[fid+i*n_Loc];
        else

          ECDF[fid+i*n_Loc]=ECDF[fid+(i-1)*n_Loc] + EDist[fid+i*n_Loc];

        //if (fid==0) printf("E_ROW_WISE_CDF fid %d %f \n", fid, ECDF[fid+i*n_Loc]);
      }

    });



  o::Write<o::Real> asd_theta(np_create*n_theta, 0, "asd_theta");
  o::Write<o::Real> diff_1(np_create*n_theta, 0, "diff_1");

  o::Write<o::Real> asd_phi(np_create*n_phi, 0, "asd_phi");
  o::Write<o::Real> diff_2(np_create*n_phi, 0, "diff_2");

  o::Write<o::Real> asd_e(np_create*n_E, 0, "asd_e");
  o::Write<o::Real> diff_3(np_create*n_E, 0, "diff_3");

  //o::Write<o::Real> v_sampled(np_create*3, -1, "v_sampled");
  //o::Write<o::Real> v_final(np_create*3, -1, "v_sampled");

  o::parallel_for(np_create, OMEGA_H_LAMBDA(const o::LO& ptid) {

    int minsi = mins[ptid];
    const auto fv2v = o::gather_verts<3>(face_verts, final_surfaces_trimmed[minsi]);      //Both do the sme thing
    const auto face = p::gatherVectors3x3(coords, fv2v);

    auto abc = p::get_face_coords_of_tet(face_verts, coords, final_surfaces_trimmed[minsi]);

    int elem_face=p::elem_id_of_bdry_face_of_tet(final_surfaces_trimmed[minsi], f2r_ptr, f2r_elem); //GET element id from face

    if (ptid<20&& false) printf("Element id %d is %d \n", ptid, elem_face);
    //if (ptid<100) printf("face id %d is %d \n", ptid, final_surfaces_trimmed[minsi]);

    if(ptid==0 && false)
    for (int i=0; i<3;i++){
      for (int j=0; j<3;j++)
      printf("abc %f \n", abc[i][j]);
    }

    auto ab = abc[1] - abc[0];
    auto ac = abc[2] - abc[0];
    auto bc = abc[2] - abc[1];

    auto abc_transform0=abc[0]-abc[0];
    auto abc_transform1=abc[1]-abc[0];
    auto abc_transform2=abc[2]-abc[0];

    if(ptid==0 && false)
    for (int i=0; i<3;i++){
      printf("abc_transform %f \n", abc_transform1[i]);
    }


    auto state=randy_pool.get_state();
    auto a1=state.drand();
    auto a2=state.drand();
    //randy_pool.free_state(state);

    auto samples=a1*abc_transform1 + a2*abc_transform2;
    //abc_transform2 = -abc_transform2;
    samples=-samples+abc_transform1;
    samples= samples+abc_transform2;
    samples=samples+abc[0];

    auto normalVec = o::cross(ab, ac);

    /*
    for(auto i=0; i<3; ++i) {
      abcd_d[ptid*4+i] = normalVec[i];
    }
    abcd_d[ptid*4+3] = -(o::inner_product(normalVec, abc[0]));
    planeNorm_d[ptid] = o::norm(normalVec);
    */

    o::Vector<4> abcd_d;
    for(auto i=0; i<3; ++i) {
      abcd_d[i] = normalVec[i];
    }
    abcd_d[3] = -(o::inner_product(normalVec, abc[0]));
    auto planeNorm_d=o::norm(normalVec);


    for (int i=0; i<3; i++){
      //pos_created[ptid*3+i]=samples[i]-abcd_d[ptid*4+i]/planeNorm_d[ptid]*0.00001;
      //ptcl_created_data_w[i*np_create+ptid]=samples[i]-abcd_d[ptid*4+i]/planeNorm_d[ptid]*0.00001;
      ptcl_created_data_w[i*np_create+ptid]=samples[i]-abcd_d[i]/planeNorm_d*0.00001;
      //ptcl_created_data_w[i*np_create+ptid]=(abc[0][i]+abc[1][i]+abc[2][i])/3; //Centroids
    }


     //The velocities: 2ND PART OF INITIALIZATION

    //FInding the z nearest to the initialized point
    double diff_pos[3];
    for (int i=0; i<3; i++){
      //diff_pos[i]= pos_created[ptid*3+2]-abc[i][2];
        diff_pos[i]= ptcl_created_data_w[2*np_create+ptid]-abc[i][2]; //z-zz[i]
        if (ptid<20 && false)
        printf("ptid %d diff_pos %f \n", ptid, diff_pos[i]);
    }

    int rzdiff=0;
    for (int i=0; i<2; i++){
      if(std::abs(diff_pos[i+1])<std::abs(diff_pos[rzdiff]))
        rzdiff=i+1;
    }


    if (ptid<20 && false){
     printf("ptid %d rzdiff %d \n",ptid, rzdiff);
    }



    //All rows 1 column of eDIST, PHIDIST AND THETADIT cumsum

    //PHI_DIST
    for (int i=0; i<n_phi;i++){
      auto col = rzdiff;
      asd_phi[ptid*n_phi+i]=phiCDF[i*n_Loc + col];


      if(ptid==0 && false)
        printf("ptid %d asd_phi is %f \n",ptid, asd_phi[ptid*n_phi+i]);
    }


    for (int i=0; i<n_theta;i++){
      auto col = rzdiff;
      asd_theta[ptid*n_theta+i]=thetaCDF[i*n_Loc + col];
    }

    for (int i=0; i<n_E;i++){
      auto col = rzdiff;
      asd_e[ptid*n_E+i]=ECDF[i*n_Loc + col];
    }




    int  min_theta, min_phi, min_E;

    //auto state=randy_pool.get_state();
    auto rand_ptcl_gen2=state.drand();
    //randy_pool.free_state(state);

    for (int i=0; i<n_E;i++){
       diff_3[ptid*n_E+i]=std::abs(asd_e[ptid*n_E+i]-rand_ptcl_gen2);
    }
    min_E=ptid*n_E;

    for (int i=0; i<n_E-1; i++){
      if (diff_3[ptid*n_E+i+1]<diff_3[min_E])
        min_E=ptid*n_E+i+1;
    }
    double pE=EGrid[min_E-ptid*n_E];
      if (pE>20)
        pE=6;
    double pvel=sqrt(2*pE*1.602e-19/184/1.66e-27);


    if (ptid<20 && false){
     printf("ptid %d pvel %f \n",ptid, pvel);
    }


    //auto state=randy_pool.get_state();
    auto rand_ptcl_gen4=state.drand();
    //randy_pool.free_state(state);
    for (int i=0; i<n_theta;i++){
       diff_1[ptid*n_theta+i]=std::abs(asd_theta[ptid*n_theta+i]-rand_ptcl_gen4);
    }
    min_theta=ptid*n_theta;
    for (int i=0; i<n_theta-1; i++){
      if (diff_1[ptid*n_theta+i+1]<diff_1[min_theta])
        min_theta=ptid*n_theta+i+1;
    }
    double pTheta=thetaGrid[min_theta-ptid*n_theta];


    if (ptid<20 && false){
     printf("ptid %d pTheta %f \n", ptid, pTheta);
    }



    //auto state=randy_pool.get_state();
    auto rand_ptcl_gen3=state.drand();
    randy_pool.free_state(state);
    for (int i=0; i<n_phi;i++){
       diff_2[ptid*n_phi+i]=std::abs(asd_phi[ptid*n_phi+i]-rand_ptcl_gen3);
    }
    min_phi=ptid*n_phi;
    for (int i=0; i<n_phi-1; i++){
      if (diff_2[ptid*n_phi+i+1]<diff_2[min_phi])
        min_phi=ptid*n_phi+i+1;
    }
    double pPhi=phiGrid[min_phi-ptid*n_phi];


    if (ptid<20 && false){
     printf("ptid %d min_phi %d, pPhi %f \n", ptid, min_phi-ptid*n_phi, pPhi);
    }

    //v_sampled[ptid*3]=pvel*sin(pPhi*3.1415/180)*cos(pTheta*3.1415/180);
    //v_sampled[ptid*3+1]=pvel*sin(pPhi*3.1415/180)*sin(pTheta*3.1415/180);
    //v_sampled[ptid*3+2]=pvel*cos(pPhi*3.1415/180);

    //Replace above three lines by
    o::Vector<3> v_sampled;

    v_sampled[0]=pvel*sin(pPhi*3.1415/180)*cos(pTheta*3.1415/180);
    v_sampled[1]=pvel*sin(pPhi*3.1415/180)*sin(pTheta*3.1415/180);
    v_sampled[2]=pvel*cos(pPhi*3.1415/180);


    if (ptid<20 && false){
      printf("vsampled 0 is %f \n",v_sampled[0]);
      printf("vsampled 1 is %f \n",v_sampled[1]);
      printf("vsampled 2 is %f \n",v_sampled[2]);
    }



    o::Vector<3> b_field;
    o::Vector<3> pos_b;


    //pos_b[0]=sqrt(pos_created[ptid*3+0]*pos_created[ptid*3+0] + pos_created[ptid*3+1]*pos_created[ptid*3+1]);
    pos_b[0]=sqrt(ptcl_created_data_w[0*np_create+ptid]*ptcl_created_data_w[0*np_create+ptid] + ptcl_created_data_w[1*np_create+ptid]*ptcl_created_data_w[1*np_create+ptid]);
    pos_b[1]=0;
    pos_b[2]=ptcl_created_data_w[2*np_create+ptid];

    p::interp2dVector(BField_2d, bX0, bZ0, bDx, bDz, bGridNx, bGridNz,
      pos_b, b_field, true);

    if (ptid==0 && false){

      printf("bfield is %f",b_field[0]);
      printf("bfield is %f",b_field[1]);
      printf("bfield is %f \n",b_field[2]);

    }

    o::Vector<3> surfNorm;
    o::Vector<3> surfPar;
    o::Vector<3> surfParX;

    //surfNorm[0]=-abcd_d[ptid*4]/planeNorm_d[ptid];
    surfNorm[0]=-abcd_d[0]/planeNorm_d;
    //surfNorm[1]=-abcd_d[ptid*4+1]/planeNorm_d[ptid];
    surfNorm[1]=-abcd_d[1]/planeNorm_d;
    //surfNorm[2]=-abcd_d[ptid*4+2]/planeNorm_d[ptid];
    surfNorm[2]=-abcd_d[2]/planeNorm_d;
    surfPar = o::cross(surfNorm,o::cross(b_field,surfNorm));

    if(ptid==0 && false){
    for (int i= 0; i<3;i++)
        printf("abcd %f \n", abcd_d[i]);
    //surfPar =o::cross(b_field,surfNorm);
      for (int i= 0; i<3;i++)
        printf("SURF_PAR %f \n", surfPar[i]);

    }
    double surfPar_norm=o::norm(surfPar);
    surfPar=surfPar/surfPar_norm;
    surfParX=o::cross(surfPar,surfNorm);

    for (int i=0; i<3;i++){

      //v_final[ptid*3+i]=v_sampled[ptid*3]*surfParX[i]+surfPar[i]*v_sampled[ptid*3+1]+surfNorm[i]*v_sampled[ptid*3+2];
      //ptcl_created_data_w[(i+3)*np_create+ptid]=v_sampled[ptid*3]*surfParX[i]+surfPar[i]*v_sampled[ptid*3+1]+surfNorm[i]*v_sampled[ptid*3+2];
      ptcl_created_data_w[(i+3)*np_create+ptid]=v_sampled[0]*surfParX[i]+surfPar[i]*v_sampled[1]+surfNorm[i]*v_sampled[2];
    }
  });

  printf("ptcl_created_data INSIDE is %d \n", ptcl_created_data_w.size());
  ptcl_created_data=ptcl_created_data_w;

  o::parallel_for(np_create, OMEGA_H_LAMBDA(const o::LO& ind) {

      if (ind<10)
      for(int i=0; i<3;i++){
        printf("IND %d i %d , PTCL_POSITIONS %f \n ", ind, i, ptcl_created_data_w[i*np_create+ind]);
        printf("IND %d i %d , PTCL_VELOCITIES %f \n ", ind, i, ptcl_created_data_w[(i+3)*np_create+ind]);
      }

  });


  //writeGeneratedParticleDataNc(ptcl_created_data_host, np_create, totalPtcls, "ptcl_gen_file0.nc");

  bool write_output=false;
  if (write_output){

    int nThistory=1;
    OutputNcFileFieldStruct outStruct({"nP", "nT"}, {"x", "y", "z", "vx", "vy", "vz"},
                                       {np_create, nThistory});
    if (rank==0){
        o::HostWrite<o::Real> ptcl_created_data_host0(ptcl_created_data_w);
        std::ofstream outfile;
        outfile.open("file0.txt");
        for (int i=0; i<ptcl_created_data_host0.size();i++){
            outfile << ptcl_created_data_host0[i] << std::endl;
        }
        outfile.close();

    }
    if (rank==1){
        o::HostWrite<o::Real> ptcl_created_data_host1(ptcl_created_data_w);
        std::ofstream outfile1;
        outfile1.open("file1.txt");
        for (int i=0; i<ptcl_created_data_host1.size();i++){
          outfile1 << ptcl_created_data_host1[i] << std::endl;
        }
        outfile1.close();
    }
  }
  //writeOutputNcFile(ptcl_created_data_host0, np_create, 6, outStruct, "ptcl_gen_file0.nc");

}

//  END_ OF_PARTILCLE_INITIALIZATION

//allocated for totalPtcls in each picpart
void GitrmParticles::initPtclHistoryData(int hstep) {
  OMEGA_H_CHECK(hstep > 0);
  bool debug = true;
  histInterval = hstep;
  if(histInterval > numIterations)
    histInterval = numIterations;
  dofHistory = 6; //x,y,z,vx,vy,vz
  histStep = 0;
  //1 extra for initial
  nThistory = 1 + numIterations/histInterval;
  auto rem = numIterations%histInterval;
  if(rem)
    ++nThistory;
 // printf("While initialization of data my rank is %d \n",rank);
  nFilledPtclsInHistory = 0;

  int size = totalPtcls*dofHistory*nThistory;
  int nph = totalPtcls*nThistory;

  if(!rank)
    printf("History: Allocating %d doubles %d + %d ints\n", size, nph, totalPtcls);
  if(debug)
    gitrm::printCudaMemInfo();
  ptclHistoryData = o::Write<o::Real>(size, 0, "ptclHistoryData");
  if(debug)
    gitrm::printCudaMemInfo();
  ptclIdsOfHistoryData = o::Write<o::LO>(nph, -1, "ptclIdsOfHistoryData");
  lastFilledTimeSteps = o::Write<o::LO>(totalPtcls, 0); //init 0 if mpi reduce
}


//all are collected by rank=0
void GitrmParticles::writePtclStepHistoryFile(std::string ncfile, bool debug) const {
  if(!rank)
    printf("writing Particle history NC file\n");
  o::HostWrite<o::Real> histData_in(ptclHistoryData);
  o::HostWrite<o::Real> histData_h(histData_in.size(), "histData_h");

  auto size = totalPtcls*dofHistory*nThistory;

  if(!rank)
    printf("MPI gather history size %d totalPtcls %d  nThistory %d allocated"
      " history size %d\n", size, totalPtcls, nThistory, ptclHistoryData.size());
  MPI_Reduce(histData_in.data(), histData_h.data(), histData_in.size(), MPI_DOUBLE,
    MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  o::HostWrite<o::LO> lastFilled_in(lastFilledTimeSteps);
  o::HostWrite<o::LO> lastFilled_h(lastFilled_in.size(), "lastFilled_h");
  //note: make sure default init with 0's for SUM
  MPI_Reduce(lastFilled_in.data(), lastFilled_h.data(), lastFilled_in.size(),
    MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  //only rank 0 is collecting
  if(!rank) {
    if(debug)
      for(int i =0 ; i < 100 && i< histData_h.size() && i<size; ++i) {
        printf(" %g", histData_h[i]);
        if(i%6==5) printf("\n");
      }
    auto dof = dofHistory;
    int numPtcls = totalPtcls;
    auto nTh = nThistory;
    auto histData_d = o::Write<o::Real>(histData_h);
    auto lastFilled_d = o::Write<o::LO>(lastFilled_h);
    //fill empty elements with last filled values
    //if(debug && !rank)
      printf("numPtcls %d lastFilled_h.size %d histData_d.size %d np@rank0 %d\n",
        numPtcls, lastFilled_h.size(), histData_d.size(), lastFilledTimeSteps.size());
    auto lambda = OMEGA_H_LAMBDA(const o::LO& ptcl) {
      auto ts = lastFilled_d[ptcl];
      printf("ts is last filled %d \n",ts);
      for(int idof=0; idof<dof; ++idof) {
        auto ref = ts*numPtcls*dof + ptcl*dof + idof;
        auto dat = histData_d[ref];
        for(int it = ts+1; it < nTh; ++it) {
          auto ind = it*numPtcls*dof + ptcl*dof + idof;
          histData_d[ind] = dat;
        }
      }
    };
    o::parallel_for(numPtcls, lambda,"write_history_nc");
    if(!rank)
      printf("writing history nc file \n");
    o::HostWrite<o::Real> histData(histData_d);
    OutputNcFileFieldStruct outStruct({"nP", "nT"}, {"x", "y", "z", "vx", "vy", "vz"},
                                       {numPtcls, nThistory});
    writeOutputNcFile(histData, numPtcls, dofHistory, outStruct, ncfile);
  }
}

/*
void GitrmParticles::writePtclStepHistoryFile(std::string ncfile, bool debug) const {

  if(!rank)
    printf("writing Particle history NC file\n");

  o::HostWrite<o::Real> histData_in(ptclHistoryData);
  o::HostWrite<o::Real> histData_h(histData_in.size(), "histData_h");

  int each_chunk=totalPtcls/commSize;
  if(rank==commSize-1) each_chunk=each_chunk+totalPtcls%commSize;

  auto size = totalPtcls*dofHistory*nThistory;

  o::HostWrite<o::LO> lastFilled_in(lastFilledTimeSteps);
  o::HostWrite<o::LO> lastFilled_h(lastFilled_in.size(), "lastFilled_h");

  auto dof = dofHistory;
  int numPtcls = each_chunk;
  auto nTh = nThistory;
  auto histData_d = o::Write<o::Real>(histData_in);
  auto lastFilled_d = o::Write<o::Real>(lastFilled_in);

    auto lambda = OMEGA_H_LAMBDA(const o::LO& ptcl) {
      auto ts = lastFilled_d[ptcl];
      printf("ts for ptcl %d is last filled %d \n",ptcl, ts);
      for(int idof=0; idof<dof; ++idof) {
        //auto ref = ts*numPtcls*dof + ptcl*dof + idof;
        auto ref = idof*numPtcls*nTh + ptcl*nTh + ts;
        //auto dat = histData_d[ref];
        auto dat = histData_d[ref];
        for(int it = ts+1; it < nTh; ++it) {
          //auto ind = it*numPtcls*dof + ptcl*dof + idof;
          auto ind =idof*numPtcls*nTh + ptcl*nTh + it;
          histData_d[ind] = dat;
        }
      }
    };
    o::parallel_for(numPtcls, lambda);
    if(!rank)
      printf("writing history nc file \n");
    o::HostWrite<o::Real> histData(histData_d);
    OutputNcFileFieldStruct outStruct({"nP", "nT"}, {"x", "y", "z", "vx", "vy", "vz"},
                                       {numPtcls, nThistory});
    writeOutputNcFile(histData, numPtcls, totalPtcls, nThistory, ncfile);

}

*/

//if elem_ids invalid then store xpoints, else store final position.
//if iter is before timestep starts, store init position, which makes
//total history step +1. Stored in aray for totalPtcls

/*
void GitrmParticles::updatePtclHistoryData(int iter, int nT, const o::LOs& elem_ids) {
  bool debug = false;
  if(!histInterval || (iter < nT-1 && iter>=0 && iter%histInterval))
    return;

  int each_chunk=totalPtcls/commSize;
  int each_chunk_pos=each_chunk;
  if(rank==commSize-1) each_chunk=each_chunk+totalPtcls%commSize;

  int iThistory = (iter<0) ? 0: (1 + iter/histInterval);
  auto size = ptclIdsOfHistoryData.size();
  auto dof = dofHistory;
  auto nPtcls = each_chunk;
  auto ndi = dofHistory*iThistory;
  auto nh = nThistory;
  auto ptclIds = ptclIdsOfHistoryData;
  auto historyData = ptclHistoryData;
  if(historyData.size() <= nPtcls*dof*iThistory) {
      printf("History storage is insufficient @ t %d histStep %d\n", iter, iThistory);
    return;
  }
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto pos_next = ptcls->get<PTCL_NEXT_POS>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto xfaces_d = wallCollisionFaceIds;
  auto xpts = wallCollisionPts;
  auto lastSteps = lastFilledTimeSteps;

  if(debug && !rank)
    printf(" histupdate iter %d iThistory %d size %d ndi %d nh %d nelid %d nfid %d\n",
      iter, iThistory, size, ndi, nh, elem_ids.size(), xfaces_d.size());
  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      auto vel = p::makeVector3(pid, vel_ps);
      auto pos = p::makeVector3(pid, pos_next);
      auto data_comb={pos,vel};

      if(iter < 0)
        pos = p::makeVector3(pid, pos_ps);
      else {
        auto check = (elem_ids[pid] >= 0 || xfaces_d[pid] >=0);
        OMEGA_H_CHECK(check);
      }
      if(iter >=0 && xfaces_d[pid] >=0) {
        for(int i=0; i<3; ++i) {
          pos[i] = xpts[pid*3+i];
        }
      }
      printf("Printed till last steps %d %d \n", rank, ptcl);
      //note: index on global ptcl id
      lastSteps[ptcl] = iThistory;
      int beg = nPtcls*dof*iThistory + ptcl*dof;
      for(int i=0; i<3; ++i) {

        historyData[i*nPtcls*nThistory+ptcl*nThistory+iThistory] = pos[i];
        historyData[(i+3)*nPtcls*nThistory+ptcl*nThistory+iThistory] = vel[i];

      }
      if(debug && !rank)
        printf("iter %d ptcl %d pid %d beg %d pos %g %g %g vel %g %g %g\n", iter,
          ptcl, pid, beg, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
      Kokkos::atomic_exchange(&(ptclIds[ptcl]), ptcl);
    }// mask
  };
  p::parallel_for(ptcls, lambda, "updateStepData");

}

*/

void GitrmParticles::updatePtclHistoryData(int iter, int nT, const o::LOs& elem_ids) {
  bool debug = false;
  if(!histInterval || (iter < nT-1 && iter>=0 && iter%histInterval))
    return;

  int iThistory = (iter<0) ? 0: (1 + iter/histInterval);
 // printf("iThistory is %d \n", iThistory);
  auto size = ptclIdsOfHistoryData.size();
  auto dof = dofHistory;
  auto nPtcls = totalPtcls;
  auto ndi = dofHistory*iThistory;
  auto nh = nThistory;
  auto ptclIds = ptclIdsOfHistoryData;
  auto historyData = ptclHistoryData;
  if(historyData.size() <= nPtcls*dof*iThistory) {
    if(!rank)
      printf("History storage is insufficient @ t %d histStep %d\n", iter, iThistory);
    return;
  }
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto pos_next = ptcls->get<PTCL_NEXT_POS>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto pid_ps_global=ptcls->get<PTCL_ID_GLOBAL>();
  auto xfids = wallCollisionFaceIds;
  auto xpts = wallCollisionPts;
  auto lastSteps = lastFilledTimeSteps;
  if(debug && !rank)
    printf(" histupdate iter %d iThistory %d size %d ndi %d nh %d nelid %d nfid %d\n",
      iter, iThistory, size, ndi, nh, elem_ids.size(), xfids.size());
  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      auto ptcl_global=pid_ps_global(pid);
      auto vel = p::makeVector3(pid, vel_ps);
      auto pos = p::makeVector3(pid, pos_next);
      if(iter < 0)
        pos = p::makeVector3(pid, pos_ps);
      else {
        auto check = (elem_ids[pid] >= 0 || xfids[pid] >=0);
        OMEGA_H_CHECK(check);
      }
      if(iter >=0 && xfids[pid] >=0) {
        for(int i=0; i<3; ++i) {
          pos[i] = xpts[pid*3+i];
        }
      }
      //note: index on global ptcl id
      lastSteps[ptcl_global] = iThistory;
      int beg = nPtcls*dof*iThistory + ptcl_global*dof;
      for(int i=0; i<3; ++i) {
        historyData[beg+i] = pos[i];
        //printf("The particle is %d positions are %e \n", ptcl, historyData[beg+i]);
        historyData[beg+3+i] = vel[i];
        //printf("The particle is %d velocities are %e \n",ptcl, historyData[beg+3+i]);
      }
      if(debug && !rank)
        printf("iter %d ptcl %d pid %d beg %d pos %g %g %g vel %g %g %g\n", iter,
          ptcl, pid, beg, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
      Kokkos::atomic_exchange(&(ptclIds[ptcl_global]), ptcl_global);
    }// mask
  };
  p::parallel_for(ptcls, lambda, "updateStepData");
}


void GitrmParticles::writeDetectedParticles(std::string fname, std::string header) const {
  int rank = this->rank;
  if(!rank)
    printf("writing #Detected Particles\n");
  o::HostWrite<o::LO> dataTot_in(collectedPtcls);
  o::HostWrite<o::LO> dataTot(collectedPtcls.size());

  //NOTE: based on the array of same size on all picparts
  if(!rank)
    printf("Detected size %d this %d \n", dataTot.size(),dataTot_in.size());
  int stat = MPI_Reduce(dataTot_in.data(), dataTot.data(), dataTot_in.size(),
    MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  OMEGA_H_CHECK(!stat);
  if(rank)
    return;

  printf("rank %d Detected size %d this %d \n", rank, dataTot.size(), dataTot_in.size());
  int total = 0;
  auto dataTot_d = o::Write<o::LO>(dataTot);
  Kokkos::parallel_reduce(dataTot_d.size(), OMEGA_H_LAMBDA(const int i, o::LO& lsum) {
    lsum += dataTot_d[i];
  }, total);
  Kokkos::fence();
  std::ofstream outf;
  outf.open(fname);
  outf << header << "\n";
  outf << "total collected " <<  total << "\n";
  outf << "index   total\n";
  for(int i=0; i<dataTot.size(); ++i)
    outf <<  i << " " << dataTot[i] << "\n";
  outf.close();
}

//TODO write nc file
void GitrmParticles::writeOutPtclEndPoints(const std::string& file) const {
  if(!rank)
    printf("writing particle End-points file\n");
  o::HostWrite<o::Real> pts_in(ptclEndPoints);
  o::HostWrite<o::Real> pts(pts_in.size(), "ptclEndPoints_h");
  //TODO handle if different size in ranks
  int stat = MPI_Reduce(pts_in.data(), pts.data(), pts_in.size(), MPI_DOUBLE, MPI_SUM,
             0, MPI_COMM_WORLD);
  OMEGA_H_CHECK(!stat);
  if(!rank) {
    std::ofstream outf(file);
    //Pos( 1,:) = [ 0.00158164 0.0105611 0.45 ];
    for(int i=0; i<pts.size(); i=i+3)
      outf <<  "Pos( " << i/3 << ",:) = [ " <<  pts[i] << " " << pts[i+1] << " "
        << pts[i+2] << " ];\n";
  }
  if(rank) {
    std::string name = "positions-rank_" + std::to_string(rank) + ".txt";
    std::ofstream outf(name);
    for(int i=0; i<pts_in.size(); i=i+3)
      outf <<  "Pos " << i/3 << " = " <<  pts_in[i] << " " << pts_in[i+1] << " "
      << pts_in[i+2] << "\n";
  }
}

//detector grids
void GitrmParticles::initPtclDetectionData(int numGrid) {
  collectedPtcls = o::Write<o::LO>(numGrid, 0, "collectedPtcls");
}


OMEGA_H_DEVICE int calculateDetectorIndex(const int geomId, o::Vector<3>& xpt) {
  if(geomId != PISCES_ID)
    return -1;
  //only for pisces case
  double radMax = 0.05; //m 0.0446+0.005
  double zMin = 0; //m height min
  double zMax = 0.15; //m height max 0.14275
  double htBead1 =  0.01275; //m ht of 1st bead
  double dz = 0.01; //m ht of beads 2..14
  auto x = xpt[0];
  auto y = xpt[1];
  auto z = xpt[2];
  o::Real rad = sqrt(x*x + y*y);
  int zInd = -1;
  if(rad < radMax && z <= zMax && z >= zMin)
    zInd = (z > htBead1) ? (1+(o::LO)((z-htBead1)/dz)) : 0;
  return zInd;
}

//TODO dimensions set for pisces to be removed
//call this before re-building, since mask of exiting ptcl removed from origin elem
void GitrmParticles::updateParticleDetection(const o::LOs& elem_ids, o::LO iter,
   bool last, bool debug) {
  int rank = this->rank;
  const int geomId = geometryId; // PISCES_ID,  ITER_ID
  auto& data_d = collectedPtcls;
  auto detIds = mesh.get_array<o::LO>(o::FACE, "detectorSurfaceIndex");

  //auto pid_ps_global=ptcls->get<PTCL_ID_GLOBAL>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto pos_next = ptcls->get<PTCL_NEXT_POS>();
  const auto& xpoints = wallCollisionPts;
  auto& xfaces_d = wallCollisionFaceIds;
  //array of global id won't work for large no of ptcls
  auto& ptclEndPts = ptclEndPoints;
  //o::Write<o::LO> nPtclsInElems(mesh.nelems(), 0, "nPtclsInElems");
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      //auto ptcl_global=pid_ps_global(pid);
      auto pos = p::makeVector3(pid, pos_next);
      auto fid = xfaces_d[pid];
      auto elm = elem_ids[pid];
      if(last) {
        //for(int i=0; i<3; ++i)
        //  ptclEndPts[3*ptcl+i] = pos[i];
          //ptclEndPts[3*ptcl_global+i] = pos[i];
        if(debug)
          printf("rank %d p %d %g %g %g \n", rank, ptcl, pos[0], pos[1], pos[2]);
      }
      if(elm < 0 && fid>=0) {
        auto xpt = o::zero_vector<3>();
        for(o::LO i=0; i<3; ++i) {
          xpt[i] = xpoints[pid*3+i]; //ptcl = 0..numPtcls
          //TODO secondary wall collision
          //ptclEndPts[3*ptcl+i] = xpt[i];
          //ptclEndPts[3*ptcl_global+i] = xpt[i];
          if(debug && !rank)
            printf("rank %d p %d %g %g %g \n",rank, ptcl, xpt[0], xpt[1], xpt[2]);
        }
        o::LO zInd = (debug) ? calculateDetectorIndex(geomId, xpt) : -1;

        auto detId = detIds[fid];
        if(detId >= 0) { //TODO
          if(debug)
            printf("%d: ptclID %d zInd %d detId %d pos %.5f %.5f %.5f iter %d\n",
              rank, pid_ps(pid), zInd, detId, xpt[0], xpt[1], xpt[2], iter);
          Kokkos::atomic_increment(&data_d[detId]);
        }
      }
    }
  };
  p::parallel_for(ptcls, lamb, "StoreDetectedData");
}

// gridsR gets preference. If >1 then gridsZ not taken
void GitrmParticles::storePtclDataInGridsRZ(o::LO iter, o::Write<o::LO>& data_d,
   int gridsR, int gridsZ, double radMax, double zMax, double radMin, double zMin) {
  auto dz = (zMax - zMin)/gridsZ;
  auto dr = (radMax - radMin)/gridsR;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto tgt_ps = ptcls->get<PTCL_NEXT_POS>();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask >0) {
      auto x = tgt_ps(pid, 0);
      auto y = tgt_ps(pid, 1);
      auto z = tgt_ps(pid, 2);
      auto rad = sqrt(x*x + y*y);
      int ind = -1;
      if(rad < radMax && radMin >= radMin && z <= zMax && z >= zMin)
        if(gridsR >1) //prefer radial
          ind = (int)((rad - radMin)/dr);
        else if(gridsZ >1)
          ind = (int)((z - zMin)/dz);
      //int dir = (gridsR >1)?1:0;
      if(ind >=0) {
        Kokkos::atomic_increment(&data_d[ind]);
      }
    } //mask
  };
  p::parallel_for(ptcls, lamb);
}
