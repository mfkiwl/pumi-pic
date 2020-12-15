#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include "perfTypes.hpp"
#include "../particle_structs/test/Distribute.h"
#include <string>

PS* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name);
PS* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids);


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  // Default values if not specified on command line
  int test_num = 2;
  int team_size = 32;
  int vert_slice = 1024;

  /* Check commandline arguments */
  //Required arguments
  int num_elems = atoi(argv[1]);
  int num_ptcls = atoi(argv[2]);
  int strat = atoi(argv[3]);
  bool optimal = false;
  double param = 1; 

  //Optional arguments specified with flags
  for(int i = 4; i < argc; i+=2){
    // -n = test_num
    if(std::string(argv[i]) == "-n"){
      test_num = atoi(argv[i+1]);
    }
    // -s = team_size (/chunk width)
    else if(std::string(argv[i]) == "-s"){
      team_size = atoi(argv[i+1]);
    }
    // -v = vertical slicing
    else if(std::string(argv[i]) == "-v"){
      vert_slice = atoi(argv[i+1]);
    }
    else if(std::string(argv[i]) == "--optimal"){
      optimal = true;
      i--;
    }
    else if(std::string(argv[i]) == "-r"){
      param = atof(argv[i+1]);
    }
    else{
      fprintf(stderr, "Illegal argument: %s", argv[i]);
      // insert usage statement
    }
  }

  fprintf(stderr, "Test Command:\n");
  for(int i = 0; i < argc; i++){
    fprintf(stderr, " %s", argv[i]);
  }
  fprintf(stderr, "\n");

  /* Enable timing on every process */
  pumipic::SetTimingVerbosity(0);

  { //Begin Kokkos region

    /* Create initial distribution of particles */
    kkLidView ppe("ptcls_per_elem", num_elems);
    kkLidView ptcl_elems("ptcl_elems", num_ptcls);
    kkGidView element_gids("",0);
    printf("Generating particle distribution with strategy: %s\n", distribute_name(strat));
    distribute_particles(num_elems, num_ptcls, strat, ppe, ptcl_elems,param);
    //printView(ppe); //Uncomment to view distribution of particles

    /* Create particle structure */
    ParticleStructures structures;
    if(test_num % 2 == 0){ //0 or 2
      //note distribution detection is on so team_size has no impace and labels the output possibly wrong
      if(strat == 3)  team_size = 512;
      structures.push_back(std::make_pair("Sell-"+std::to_string(team_size)+"-ne",
                                      createSCS(num_elems, num_ptcls, ppe, element_gids,
                                                team_size, num_elems, vert_slice, "Sell-"+std::to_string(team_size)+"-ne")));
    }
    if(test_num > 0){ //1 or 2
      structures.push_back(std::make_pair("CSR",
                                      createCSR(num_elems, num_ptcls, ppe, element_gids)));
      structures.back().second->setTeamSize(team_size);
      if(optimal) //CSR optimal push may be 256 need to look a little closer still
        structures.back().second->setTeamSize(512);
    }

    for(int i = 0; i < structures.size(); ++i){
      std::string name = structures[i].first;
      PS* ptcls = structures[i].second;

      fprintf(stderr,"%s\t", name.c_str());
      
      // Figure out the size of the major arrays in each structure (assume other data is 
      // insignficant in total mem consumption/is O(1) )
      if(name == "CSR"){
        //only need size of ptcl_data = capacity*size of 1 entry
        size_t size = ptcls->capacity() * (4*sizeof(double)+sizeof(int))/ 2^20 ;
        fprintf(stderr,"%d\n",size);
      }
      else{
        //need size of all the views
      }


    }
    

    for (size_t i = 0; i < structures.size(); ++i)
      delete structures[i].second;
    structures.clear();

  }

  Kokkos::finalize();
  return 0;
}

PS* createSCS(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids, int C, int sigma, int V, std::string name) {
  Kokkos::TeamPolicy<ExeSpace> policy(4, C);
  pumipic::SCS_Input<PerfTypes> input(policy, sigma, V, num_elems, num_ptcls, ppe, elm_gids);
  input.name = name;
  return new pumipic::SellCSigma<PerfTypes, MemSpace>(input);
}
PS* createCSR(int num_elems, int num_ptcls, kkLidView ppe, kkGidView elm_gids) {
  Kokkos::TeamPolicy<ExeSpace> po(32,Kokkos::AUTO);
  return new pumipic::CSR<PerfTypes, MemSpace>(po, num_elems, num_ptcls, ppe, elm_gids);

}


