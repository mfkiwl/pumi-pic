#include <fstream>
#include <cstdlib>
#include <vector>
#include <set>
#include <random>
#include <chrono>
#include <Omega_h_int_scan.hpp>
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
  //if(!useGitrRndNums) //TODO testing
    initRandGenerator(seed);
  requireFindingAllPtcls = REQUIRE_FINDING_ALL_PTCLS;

  //TODO handle this neatly
  auto geomName = gm->getGeometryName();
 
  if(geomName == "pisces")
    geometryId = PISCES_ID;
  else if(geomName == "Iter")
    geometryId = ITER_ID;
  else
    geometryId = INVALID;

  //TODO set using mesh class data
  int dataSize = (geometryId == PISCES_ID) ? 14 : mesh.nfaces();
  initPtclDetectionData(dataSize);
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
    printf("%d: isFullMesh %d Particles %d total %ld reduced %ld\n",
       rank, isFullMesh, numInitPtcls, totalPtcls, totalPtcls_);
  if(rank == collectRank && requireFindingAllPtcls)
    OMEGA_H_CHECK(totalPtcls_ == totalPtcls);
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

  //Reading all simulating particles in all ranks
  auto stat = readParticleSourceNcFile(fName, readInData_h, numPtclsRead,
     size_t(each_chunk), each_chunk_pos, true);
  //OMEGA_H_CHECK( stat && (numPtclsRead >= totalPtcls));
  auto readInData_r = o::Reals(o::Write<o::Real>(readInData_h));
  o::LOs elemIdOfPtcls;
  o::LOs ptclDataInds;
  o::LOs numPtclsInElems;
  if(!rank)
    printf("%d: partices read %d. Finding ElemIds of Ptcls \n", rank, numPtclsRead);
  findElemIdsOfPtclCoordsByAdjSearch(readInData_r, elemIdOfPtcls, ptclDataInds,
    numPtclsInElems);

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
   o::LO& parentElem) {
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
  };
  o::parallel_for(picparts.nelems(), lamb, "search_parent_of_ptcl");
  auto result = o::HostWrite<o::LO>(elemDet);
  parentElem = result[0];
  return parentElem >= 0;
}

// totalPtcls are searched in each rank
o::LO GitrmParticles::searchAllPtclsInAllElems(const o::Reals& data,
   o::Write<o::LO>& elemIdOfPtcls, o::Write<o::LO>& numPtclsInElems) {
  MESHDATA(mesh);
  int rank = this->rank;
  auto owners = elemOwners;
  auto nPtclsRead = numPtclsRead;
  auto nPtcls = elemIdOfPtcls.size();

  auto lamb = OMEGA_H_LAMBDA(const o::LO& elem) {
    bool select = (rank == owners[elem]);
    //if(elem==443015) printf("rank %d  elem %d owner %d\n", rank, elem, owners[elem]);
    if(select) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto tet = p::gatherVectors4x3(coords, tetv2v);
      auto pos = o::zero_vector<3>();
      for(auto pind=0; pind < nPtcls; ++pind) {
        for(int j=0; j<3; ++j)
          pos[j] = data[j*nPtclsRead+pind];
        auto bcc = o::zero_vector<4>();
        p::find_barycentric_tet(tet, pos, bcc);
        if(p::all_positive(bcc, 1.0e-10)) {
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
   o::Write<o::LO>& elemIdOfPtclsAll) {
  int debug = 0;
  int rank = this->rank;
  MESHDATA(mesh);
  auto owners = elemOwners;
  auto nPtclsRead = numPtclsRead;
  int maxSearch = 1000;

  //search all particles starting with this element
  auto lambda = OMEGA_H_LAMBDA(const o::LO& ip) {
    bool found = false;
    auto pos = o::zero_vector<3>();
    auto bcc = o::zero_vector<4>();
    o::LO elem = parentElem;
    int isearch=0;
    while(!found) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto tet = p::gatherVectors4x3(coords, tetv2v);
      for(int j=0; j<3; ++j) {
        pos[j] = data[j*nPtclsRead+ip];
      }
      p::find_barycentric_tet(tet, pos, bcc);

      bool select = (rank == owners[elem]);
      if(select && p::all_positive(bcc, 0)) {
        Kokkos::atomic_exchange(&(elemIdOfPtclsAll[ip]), elem);
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
          printf("%d: Error finding particle %d in rank %d \n", rank, ip);
        break;
      }
      ++isearch;
    }
  };
  auto nPtcls = elemIdOfPtclsAll.size();
  o::parallel_for(nPtcls, lambda, "init_ptcl2");
  auto status=o::get_min(o::LOs(elemIdOfPtclsAll));
  return o::get_min(o::LOs(elemIdOfPtclsAll));
}


// search all particles read. TODO update this for partitioning
void GitrmParticles::findElemIdsOfPtclCoordsByAdjSearch(const o::Reals& data,
   o::LOs& elemIdOfPtcls, o::LOs& ptclDataInds, o::LOs& numPtclsInElems) {
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
    searchPtclInAllElems(data, ip, parentElem);
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
  o::Write<o::LO> elemIdOfPtclsAll(nPtcls, -1, "elemIdOfPtclsAll");
  o::LO min = -1;

  if(parentElem >= 0)
    min = searchPtclsByAdjSearchFromParent(data, parentElem, numPtclsInElemsAll,
      elemIdOfPtclsAll);

  //if not all found, do brute-force search for all
  if(min < 0)
    min = searchAllPtclsInAllElems(data, elemIdOfPtclsAll, numPtclsInElemsAll);
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
  std::cout << "Printing Nums of ptcls in elements.\n";
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
  //auto nInitPtcls = numInitPtcls;
  //TODO for full-buffer only
  int rank = this->rank;
  int pBegin = 0;// TODO rank*int(totalPtcls/commSize);
  if(debug && !rank)
    printf("%d: numInitPtcls %d numPtclsRead %d \n", rank, numInitPtcls, numPtclsRead);
  auto nel = picparts.nelems();
  o::Write<o::LO> nextPtclInd(nel, 0, "nextPtclInd");
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto pid_ps_global=ptcls->get<PTCL_ID_GLOBAL>();
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
      auto thisInd = Kokkos::atomic_fetch_add(&(nextPtclInd[elem]), 1);
      auto ind = ptclIdPtrsOfElem[elem] + thisInd;
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

//To use ptcls, PS_LAMBDA is required, not o::parallel_for
// since ptcls in each element is iterated in groups. Construct PS with
// #particles in each elem passed in, otherwise newly added particles in
// originally empty elements won't show up in PS_LAMBDA iterations.
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

  //TODO handle if different size in ranks
  if(!rank)
    printf("Detected size %d this %d \n", dataTot.size(),dataTot_in.size());
  int stat = MPI_Reduce(dataTot_in.data(), dataTot.data(), dataTot_in.size(), MPI_INT, MPI_SUM,
    0, MPI_COMM_WORLD);
  OMEGA_H_CHECK(!stat);
  if(rank)
    return;

  printf("rank %d Detected size %d this %d \n", rank, dataTot.size(),
    dataTot_in.size());
  int total = 0;
  auto dataTot_d = o::Write<o::LO>(dataTot);
  Kokkos::fence();
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
