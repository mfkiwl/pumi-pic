#include <vector>
#include <fstream>
#include <iostream>
#include <Omega_h_file.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_mesh.hpp>

#include <pumipic_mesh.hpp>
#include <pumipic_adjacency.hpp>

#include "GitrmParticles.hpp"
#include "GitrmPush.hpp"
#include "GitrmIonizeRecombine.hpp"
#include "GitrmSurfaceModel.hpp"
#include "GitrmCoulombCollision.hpp"
#include "GitrmThermalForce.hpp"
#include "GitrmCrossDiffusion.hpp"
#include "GitrmSpectroscopy.hpp"
//#include "GitrmInput.hpp"

void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void printTimerResolution() {
  Kokkos::Timer timer;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  fprintf(stderr, "kokkos timer reports 1ms as %f seconds\n", timer.seconds());
}

void updatePtclPositions(PS* ptcls) {
  auto x_ps_d = ptcls->get<0>();
  auto xtgt_ps_d = ptcls->get<1>();
  auto updatePtclPos = PS_LAMBDA(const int& e, const int& pid, const bool& mask) {
    if(mask) {
      x_ps_d(pid,0) = xtgt_ps_d(pid,0);
      x_ps_d(pid,1) = xtgt_ps_d(pid,1);
      x_ps_d(pid,2) = xtgt_ps_d(pid,2);
      xtgt_ps_d(pid,0) = 0;
      xtgt_ps_d(pid,1) = 0;
      xtgt_ps_d(pid,2) = 0;
    }
  };
  ps::parallel_for(ptcls, updatePtclPos, "updatePtclPos");
}

void rebuild(p::Mesh& picparts, PS* ptcls, o::LOs elem_ids,
    const bool output=false) {
  updatePtclPositions(ptcls);
  const int ps_capacity = ptcls->capacity();
  PS::kkLidView ps_elem_ids("ps_elem_ids", ps_capacity);
  PS::kkLidView ps_process_ids("ps_process_ids", ps_capacity);
  Omega_h::LOs is_safe = picparts.safeTag();
  Omega_h::LOs elm_owners = picparts.entOwners(picparts.dim());

  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  //Added to check particle migration
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto pid_ps_global = ptcls->get<PTCL_ID_GLOBAL>(); 
  //delete later
  //
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      int new_elem = elem_ids[pid];
      ps_elem_ids(pid) = new_elem;
      ps_process_ids(pid) = comm_rank;
     
      //Added to check migration
      auto ptcl = pid_ps(pid);
      auto ptcl_global=pid_ps_global(pid);
      if(output)
        printf("Rank:%d ptcl:%d global_particle:%d \n",ps_process_ids(pid), ptcl, ptcl_global);

      if (new_elem != -1 && is_safe[new_elem] == 0) {
        ps_process_ids(pid) = elm_owners[new_elem];
      }
      if(output)
        printf("New rank:%d ptcl:%d global_particle:%d \n",ps_process_ids(pid), ptcl, ptcl_global);
    }
  };
  ps::parallel_for(ptcls, lamb);
  ptcls->migrate(ps_elem_ids, ps_process_ids); //migrate /  rebuild
}


void search(p::Mesh& picparts, GitrmParticles& gp,  o::Write<o::LO>& elem_ids,
    bool debug=false) {
  o::Mesh* mesh = picparts.mesh();
  Kokkos::Profiling::pushRegion("gitrm_search");
  PS* ptcls = gp.ptcls;
  assert(ptcls->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 200;
  auto x_ps = ptcls->get<0>();
  auto xtgt_ps = ptcls->get<1>();
  auto pid_ps = ptcls->get<2>();
  //NOTE: elem_ids are used in rebuild
  bool isFound = p::search_mesh_3d<Particle>(*mesh, ptcls, x_ps, xtgt_ps, pid_ps,
    elem_ids, gp.wallCollisionPts_w, gp.wallCollisionFaceIds_w, maxLoops, debug);
  OMEGA_H_CHECK(isFound);
  Kokkos::Profiling::popRegion();
  gp.convertPtclWallCollisionData();
}

void profileAndInterpolateTest(GitrmMesh& gm, bool debug=false, bool inter=false) {
  gm.printDensityTempProfile(0.05, 20, 0.7, 2);
  if(inter)
    gm.test_interpolateFields(true);
}


o::Mesh readMesh(char* meshFile, o::Library& lib) {
  const auto rank = lib.world()->rank();
  (void)lib;
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    if(!rank)
      std::cout << "reading gmsh mesh " << meshFile << "\n";
    return Omega_h::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    if(!rank)
      std::cout << "reading omegah mesh " << meshFile << "\n";
    return Omega_h::read_mesh_file(meshFile, lib.self());
  } else {
    if(!rank)
      std::cout << "error: unrecognized mesh extension \'" << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  auto start_sim = std::chrono::system_clock::now();
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  const auto comm_rank = lib.world()->rank();
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if(argc < 7) {
    if(comm_rank == 0)
      std::cout << "Usage: " << argv[0]
        << " <mesh> <owners_file> <ptcls_file> <prof_file> <rate_file><surf_file>"
        << " <thermal_gradient_file> [<nPtcls><nIter> <histInterval> <gitrDataInFileName> ]\n";
    exit(1);
  }
  bool chargedTracking = true; //false for neutral tracking
  bool piscesRun = true; // add as argument later

  bool debug = false; //search
  int debug2 = 0;  //routines

  bool surfacemodel = true;
  bool spectroscopy = true;
  bool thermal_force = false; //false in pisces conf
  if(piscesRun)
    OMEGA_H_CHECK(!thermal_force);
  bool coulomb_collision = true;
  bool diffusion = true; //not for diffusion>1
  bool useCudaRnd = false; //replace kokkos rnd
  
//GitrmInput inp("gitrInput.cfg", true);
//  inp.testInputConfig();
 
  auto deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  size_t free, total;
  cudaMemGetInfo(&free, &total);

  if(comm_rank == 0) {
    std::cout << "free " << free/(1024.0*1024.0) << " total " << total/(1024.0*1024.0) << " MB\n";
    printf("device count per process %d\n", deviceCount);
    printf("world ranks %d\n", comm_size);
    printf("particle_structs floating point value size (bits): %zu\n", sizeof(fp_t));
    printf("omega_h floating point value size (bits): %zu\n", sizeof(Omega_h::Real));
    printf("Kokkos execution space memory %s name %s\n",
        typeid (Kokkos::DefaultExecutionSpace::memory_space).name(),
        typeid (Kokkos::DefaultExecutionSpace).name());
    printf("Kokkos host execution space %s name %s\n",
        typeid (Kokkos::DefaultHostExecutionSpace::memory_space).name(),
        typeid (Kokkos::DefaultHostExecutionSpace).name());
    printTimerResolution();
  }
  auto full_mesh = readMesh(argv[1], lib);
  MPI_Barrier(MPI_COMM_WORLD);

  o::CommPtr world = lib.world();
  //Create Picparts with the full mesh
  p::Input::Method bm = p::Input::Method::FULL;
  p::Input::Method safem = p::Input::Method::BFS;
  p::Input pp_input(full_mesh, argv[2], bm, safem, world);
  pp_input.bridge_dim=full_mesh.dim()-1;
  pp_input.safeBFSLayers=3;
  p::Mesh picparts(pp_input);

  o::Mesh* mesh = picparts.mesh();
  mesh->ask_elem_verts(); //caching adjacency info

  if (comm_rank == 0)
    printf("Mesh loaded with verts %d edges %d faces %d elements %d\n",
      mesh->nverts(), mesh->nedges(), mesh->nfaces(), mesh->nelems());
  std::string ptclSource = argv[3];
  std::string profFile = argv[4];
  std::string ionizeRecombFile = argv[5];
  std::string bFile="bFile"; //TODO
  std::string surfModelFile = argv[6];
  std::string thermGradientFile = argv[7];
  if (!comm_rank) {
    if(!chargedTracking)
      printf("WARNING: neutral particle tracking is ON \n");
    printf(" Mesh file %s\n", argv[1]);
    printf(" Particle Source file %s\n", ptclSource.c_str());
    printf(" Temp Density Profile file %s\n", profFile.c_str());
    printf(" IonizeRecomb File %s\n", ionizeRecombFile.c_str());
    printf(" Gradient profile File %s\n", thermGradientFile.c_str());
    printf(" SurfModel File %s\n", surfModelFile.c_str());
    printf(" Gradient profile File %s\n", thermGradientFile.c_str());
  }
  long int totalNumPtcls = 1;
  int histInterval = 0;
  double dTime = 5e-9; //pisces:5e-9 for 100,000 iterations
  int numIterations = 1; //higher beads needs >10K

  if(argc > 8)
    totalNumPtcls = atol(argv[8]);
  if(argc > 9)
    numIterations = atoi(argv[9]);
  if(argc > 10)
    histInterval = atoi(argv[10]);

  std::string gitrDataFileName;
  bool useGitrRndNums = 0;
  if(argc > 11) {
    gitrDataFileName = argv[11];
    useGitrRndNums = true;
    if(!comm_rank)
      printf(" gitr comparison DataFile %s\n", gitrDataFileName.c_str());
  }

  unsigned long int seed = 0; // zero value for seed not considered !
  GitrmParticles gp(picparts, totalNumPtcls, numIterations, dTime, useCudaRnd,
    seed, useGitrRndNums);
  if(histInterval > 0)
    gp.initPtclHistoryData(histInterval);
  if(!comm_rank)
    printf("Initializing Particles\n");
  gp.initPtclsFromFile(ptclSource, 100, true);

  // TODO use picparts
  GitrmMesh gm(*mesh);

  if(CREATE_GITR_MESH)
    gm.createSurfaceGitrMesh();

  if(piscesRun) {
    gm.markDetectorSurfaces(true);
    int dataSize = 14;
    gp.initPtclDetectionData(dataSize);
  }

  int testNumPtcls = 1;
  if(gp.useGitrRndNums) {
    gp.readGITRPtclStepDataNcFile(gitrDataFileName, testNumPtcls);
    printf("Rnd: testNumPtcls %d totalNumPtcls %d \n", testNumPtcls, totalNumPtcls);
    assert(testNumPtcls >= totalNumPtcls);
    if(!comm_rank)
      printf("useGitrRndNums is ON\n");
  }
  auto* ptcls = gp.ptcls;
  gm.initBField(bFile);
  auto initFields = gm.addTagsAndLoadProfileData(profFile, profFile, thermGradientFile);
  OMEGA_H_CHECK(!ionizeRecombFile.empty());
  GitrmIonizeRecombine gir(ionizeRecombFile, chargedTracking);
  printf("initBoundaryFaces\n");
  auto initBdry = gm.initBoundaryFaces(initFields);

  printf("Preprocessing: dist-to-boundary faces\n");
  int nD2BdryTetSubDiv = D2BDRY_GRIDS_PER_TET;
  int readInCsrBdryData = USE_READIN_CSR_BDRYFACES;
  if(readInCsrBdryData) {
    gm.readDist2BdryFacesData("bdryFaces_in.nc");
  } else {
    gm.preprocessSelectBdryFacesFromAll();
  }
  bool writeTextBdryFaces = WRITE_TEXT_D2BDRY_FACES;
  if(writeTextBdryFaces)
    gm.writeBdryFacesDataText(nD2BdryTetSubDiv);
  bool writeBdryFaceCoords = WRITE_BDRY_FACE_COORDS_NC;
  if(writeBdryFaceCoords)
    gm.writeBdryFaceCoordsNcFile(2); //selected
  bool writeMeshFaceCoords = WRITE_MESH_FACE_COORDS_NC;
  if(writeMeshFaceCoords)
    gm.writeBdryFaceCoordsNcFile(1); //all
  int writeBdryFacesFile = WRITE_OUT_BDRY_FACES_FILE;
  if(writeBdryFacesFile && !readInCsrBdryData) {
    std::string bdryOutName = "bdryFaces_" +
      std::to_string(nD2BdryTetSubDiv) + "div.nc";
    gm.writeDist2BdryFacesData(bdryOutName, nD2BdryTetSubDiv);
  }

  GitrmSurfaceModel sm(gm, surfModelFile);
  GitrmSpectroscopy sp;

  if(false)
    profileAndInterpolateTest(gm, true); //move to unit_test
  printf("\ndTime %g NUM_ITERATIONS %d\n", dTime, numIterations);
  fprintf(stderr, "\n*********Main Loop**********\n");
  auto end_init = std::chrono::system_clock::now();
  int np;
  int ps_np;

  gp.updatePtclHistoryData(-1, numIterations, o::LOs(1,-1, "history-dummy"));
  for(int iter=0; iter<numIterations; iter++) {
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    if(comm_rank == 0)
      fprintf(stderr, "=================iter %d===============\n", iter);
    Kokkos::Profiling::pushRegion("dist2bdry");
    gitrm_findDistanceToBdry(gp, gm, debug2);
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("calculateE");
    gitrm_calculateE(gp, *mesh, gm, debug2);
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("borisMove");
    gitrm_borisMove(ptcls, gm, dTime, debug2);
    Kokkos::Profiling::popRegion();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto psCapacity = ptcls->capacity();
    assert(psCapacity > 0);
    o::Write<o::LO> elem_ids(psCapacity,-1);
    search(picparts, gp, elem_ids, debug);
    auto elem_ids_r = o::LOs(elem_ids);
    Kokkos::Profiling::pushRegion("spectroscopy");
    if(spectroscopy)
      gitrm_spectroscopy(ptcls, sp, elem_ids_r, debug2);
    Kokkos::Profiling::popRegion();
    
    Kokkos::Profiling::pushRegion("ionize");
    gitrm_ionize(ptcls, gir, gp, gm, elem_ids_r, debug2);
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("recombine");
    gitrm_recombine(ptcls, gir, gp, gm, elem_ids_r, debug2);  
    Kokkos::Profiling::popRegion();
    if(diffusion) {
      Kokkos::Profiling::pushRegion("diffusion");
      gitrm_cross_diffusion(ptcls, gm, gp,dTime, elem_ids_r, debug2);
      Kokkos::Profiling::popRegion();
      search(picparts, gp, elem_ids, debug);
    }
    Kokkos::Profiling::pushRegion("coulomb_collision");
    if(coulomb_collision)
      gitrm_coulomb_collision(ptcls, &iter, gm, gp, dTime, elem_ids_r, debug2);
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("thermal_force");
    if(thermal_force)
      gitrm_thermal_force(ptcls, &iter, gm, gp, dTime, elem_ids_r, debug2);
    Kokkos::Profiling::popRegion();
    if(surfacemodel) {
      Kokkos::Profiling::pushRegion("surface_refl");
      gitrm_surfaceReflection(ptcls, sm, gp, gm, debug2);
      Kokkos::Profiling::popRegion();
      //reflected ptcl is searched beginning at orig, not from wall hit point.
      search(picparts, gp, elem_ids, debug);
    }

    elem_ids_r = o::LOs(elem_ids);
    gp.updateParticleDetection(elem_ids_r, iter, iter==numIterations-1, false);

    gp.updatePtclHistoryData(iter, numIterations, elem_ids_r);
    Kokkos::Profiling::pushRegion("rebuild");
    rebuild(picparts, ptcls, elem_ids, debug);
    Kokkos::Profiling::popRegion();
    gp.resetPtclWallCollisionData();

    if(comm_rank == 0 && iter%1000 ==0)
      fprintf(stderr, "rank %d  nPtcls %d \n", comm_rank, ptcls->nPtcls());
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
  }
  auto end_sim = std::chrono::system_clock::now();
  std::chrono::duration<double> dur_init = end_init - start_sim;
  std::cout << "\nInitialization duration " << dur_init.count()/60 << " min.\n";
  std::chrono::duration<double> dur_steps = end_sim - end_init;
  std::cout << "Total Main Loop duration " << dur_steps.count()/60 << " min.\n";

  gp.writeOutPtclEndPoints();

  if(piscesRun) {
    std::string fname("piscesCounts.txt");
    gp.writeDetectedParticles(fname, "piscesDetected");
    gm.writeResultAsMeshTag(gp.collectedPtcls);
  }
  if(histInterval >0)
    gp.writePtclStepHistoryFile("gitrm-history_par.nc");

  if(surfacemodel)
    sm.writeSurfaceDataFile("gitrm-surface.nc");
  if(spectroscopy)
    sp.writeSpectroscopyFile("gitrm-spec.nc");

  Omega_h::vtk::write_parallel("meshvtk", mesh, mesh->dim());
  cudaMemGetInfo(&free, &total);
  double mtotal = 1.0/(1024*1024) * (double)total;
  double mfree = 1.0/(1024*1024) * (double)free;
  fprintf(stderr, "GPU memory: total %.0f used %.0f MB\n", mtotal, mtotal-mfree);
  fprintf(stderr, "Done\n");
  return 0;
}

