#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <particle_structs.hpp>
#include "Distribute.h"
#include <mpi.h>

using particle_structs::CSR;
using particle_structs::MemberTypes;

typedef MemberTypes<int, double[3]> Type;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef CSR<Type, exe_space> _CSR;

bool sendToOne(int ne, int np);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  int comm_rank;
  int comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


  int ne = 5;
  int np = 20;
  int fails = 0;
  particle_structs::gid_t* gids = new particle_structs::gid_t[ne];
  distribute_elements(ne, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 2, ptcls_per_elem, ids);
  delete [] ids;
  {
    _CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
    _CSR::kkGidView element_gids_v("element_gids_v", ne);
    particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
    particle_structs::hostToDevice(element_gids_v, gids);
    delete [] ptcls_per_elem;
    delete [] gids;
    Kokkos::TeamPolicy<exe_space> po(4, 32);
    _CSR* csr = new _CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);

    char rank_str[100];
    sprintf(rank_str,"Format for rank %d", comm_rank);
    csr->printFormat(rank_str);

    typedef _CSR::kkLidView kkLidView;
    kkLidView new_element("new_element", csr->capacity());
    kkLidView new_process("new_process", csr->capacity());

    auto int_slice = csr->get<0>();
    auto double3_slice = csr->get<1>();

    auto setValues = PS_LAMBDA(int elm_id, int ptcl_id, int mask) {
      int_slice(ptcl_id) = comm_rank;
      double3_slice(ptcl_id, 0) = comm_rank;
      double3_slice(ptcl_id, 1) = (comm_rank + 1) * elm_id;
      double3_slice(ptcl_id, 2) = mask;
    };
    csr->parallel_for(setValues);
    //Send half the particles right one process except on rank 0
    if (comm_rank > 0) {
      auto setElmProcess = PS_LAMBDA(int element_id, int particle_id, int mask) {
        new_element(particle_id) = element_id;
        new_process(particle_id) = (comm_rank + (element_id==4)) % comm_size;
      };
      csr->parallel_for(setElmProcess);
    }
    else {
      auto setElmProcess = PS_LAMBDA(int element_id, int particle_id, int mask) {
        new_element(particle_id) = element_id;
        new_process(particle_id) = comm_rank;
      };
      csr->parallel_for(setElmProcess);
    }

    csr->migrate(new_element, new_process);


    int_slice = csr->get<0>();
    double3_slice = csr->get<1>();
    _CSR::kkLidView fail("fail", 1);
    auto checkValues = PS_LAMBDA(int elm_id, int ptcl_id, int mask) {
      if (mask && mask != double3_slice(ptcl_id, 2)) {
        printf("%d mask failure on ptcl %d (%d, %f)\n", comm_rank, ptcl_id, mask,
               double3_slice(ptcl_id,2));
        fail(0) = 1;
      }
      if (mask && comm_rank != int_slice(ptcl_id) && elm_id != 0) {
        printf("%d rank failure on ptcl %d\n", comm_rank, ptcl_id);
        fail(0) = 1;
      }
      if (mask && int_slice(ptcl_id) != double3_slice(ptcl_id, 0)) {
        printf("%d int/double failure on ptcl %d\n", comm_rank, ptcl_id);
        fail(0) = 1;
      }
    };
    csr->parallel_for(checkValues);
    MPI_Barrier(MPI_COMM_WORLD);
    csr->printFormat(rank_str);

    int f = particle_structs::getLastValue(fail);
    if (f == 1) {
      printf("Migration of values failed on rank %d\n", comm_rank);
      fails++;
    }
    delete csr;
  }
  
  if (!sendToOne(5000, 100000)) { // Big Test
    printf("SendToOne failed on rank %d\n", comm_rank);
    fails++;
  }

  Kokkos::finalize();
  int total_fails;
  MPI_Reduce(&fails, &total_fails, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  if (comm_rank == 0 && total_fails == 0)
    printf("All tests passed\n");
  return fails;
}

bool sendToOne(int ne, int np) {
  int comm_rank;
  int comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  particle_structs::gid_t* gids = new particle_structs::gid_t[ne];
  for (int i = 0; i < ne; ++i)
    gids[i] = i;

  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 2, ptcls_per_elem, ids);
  delete [] ids;


  _CSR::kkLidView ptcls_per_elem_v("ptcls_per_elem_v", ne);
  _CSR::kkGidView element_gids_v("element_gids_v", ne);
  particle_structs::hostToDevice(ptcls_per_elem_v, ptcls_per_elem);
  particle_structs::hostToDevice(element_gids_v, gids);
  delete [] ptcls_per_elem;
  delete [] gids;
  Kokkos::TeamPolicy<exe_space> po(4, 32);
  _CSR* csr = new _CSR(po, ne, np, ptcls_per_elem_v, element_gids_v);

  typedef _CSR::kkLidView kkLidView;
  kkLidView new_element("new_element", csr->capacity());
  kkLidView new_process("new_process", csr->capacity());

  auto int_slice = csr->get<0>();
  auto double_slice = csr->get<1>();
  kkLidView ptcls_set("ptcls_set", 1);
  auto setValues = PS_LAMBDA(int elem_id, int ptcl_id, int mask) {
    int_slice(ptcl_id) = comm_rank;
    double_slice(ptcl_id,0) = comm_rank * 5;
    if (Kokkos::atomic_fetch_add(&ptcls_set(0), mask) < np/100) {
      new_process[ptcl_id] = 0;
    }
    else
      new_process[ptcl_id] = comm_rank;
    new_element[ptcl_id] = elem_id;
  };
  csr->parallel_for(setValues);

  csr->migrate(new_element, new_process);

  int nPtcls = csr->nPtcls();
  if (comm_rank == 0 && nPtcls != np + (comm_size - 1) * np * 1.0/100) {
    fprintf(stderr, "Rank 0 has incorrect number of particles (%d != %d)\n",
            nPtcls, np + (comm_size - 1) * np/100);
    return false;
  }
  else if (comm_rank != 0 && nPtcls != np*99/100) {
    fprintf(stderr, "Rank %d has incorrect number of particles (%d != %d)\n", comm_rank,
            nPtcls, np* 99/100);
    return false;
  }

  int_slice = csr->get<0>();
  double_slice = csr->get<1>();
  kkLidView fail("fail", 1);
  auto checkValues = PS_LAMBDA(int elm_id, int ptcl_id, int mask) {
    if (mask) {
      int rank = int_slice(ptcl_id);
      double val = double_slice(ptcl_id, 0);
      if (fabs(rank*5 - val) > .0005) {
        printf("%d Value fails on ptcl %d (%d %.2f)", comm_rank, ptcl_id, rank*5, val);
        fail(0) = 1;
      }
    }
  };
  csr->parallel_for(checkValues);
  int f = particle_structs::getLastValue(fail);
  return f == 0;
}