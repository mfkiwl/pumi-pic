#include <particle_structs.hpp>
#include <ppTiming.hpp>
#include "perfTypes.hpp"
#include "../particle_structs/test/Distribute.h"

int main(int argc, char* argv[]){
  Kokkos::initialize(argc,argv);
  MPI_Init(&argc,&argv);

  {
    /* Create initial distribution of particles */
    int num_elems = atoi(argv[1]);
    int num_ptcls = atoi(argv[2]);
    int strat = atoi(argv[3]);
    kkLidView ppe("ptcls_per_elem", num_elems);
    kkGidView element_gids("",0);
    int* ppe_host = new int[num_elems];
    std::vector<int>* ids = new std::vector<int>[num_elems];
    distribute_particles(num_elems, num_ptcls, strat, ppe_host, ids);
    pumipic::hostToDevice(ppe, ppe_host);
    delete [] ppe_host;
    delete [] ids;

    /* Perform necessary statistics */
    // Max value
    double size = ppe.size();
    int max;
    Kokkos::Max<int> max_reducer(max);
    Kokkos::parallel_reduce("Max Reduce", size, KOKKOS_LAMBDA(const int& i, int& lmax){
      max_reducer.join(lmax, ppe[i]); 
    },max_reducer); 
    
    // Average
    int sum;
    Kokkos::Sum<int> sum_reducer(sum);
    Kokkos::parallel_reduce("Sum Reduce", size, KOKKOS_LAMBDA(const int& i, int& lsum){
      sum_reducer.join(lsum, ppe[i]); 
    },sum_reducer); 
    double mean = sum/size;

    double ratio = max/mean;
    if(ratio < 1.2) printf("Distribution is uniform\n");
    else if (ratio < 4) printf("Distribution is Gaussian\n");
    else printf("Distribution is skewed\n");
  }


  Kokkos::finalize();
  return 0;
}
