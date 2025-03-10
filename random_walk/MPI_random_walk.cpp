# include "functions.h"
# include <iostream>

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Define the domain
  int domain_size = 100;
  int max_walk_size = 20;
  int num_walkers_per_proc = 10;

  int subdomain_start, subdomain_size;
  std::vector<Walker> incoming_walkers, outgoing_walkers;

  // Decompose the domain
  decompose_domain(domain_size, world_rank, world_size,
    &subdomain_start, &subdomain_size);
  
  // Initialize the walkers
  initialize_walkers(num_walkers_per_proc, max_walk_size,
    subdomain_start, subdomain_size, &incoming_walkers);

  std::cout << "Process " << world_rank << " initiated " << num_walkers_per_proc
  << " walkers in subdomain " << subdomain_start << " - "
  << subdomain_start + subdomain_size - 1 << std::endl;
  
  // Determine the maximum amount of sends and receives needed to
  // complete all walkers
  int maximum_sends_recvs = max_walk_size / (domain_size / world_size) + 1;

  for (int m = 0; m < maximum_sends_recvs; m++) {
    // Process all incoming walkers
    for (int i = 0; i < incoming_walkers.size(); i++) {
        walk(&incoming_walkers[i], subdomain_start, subdomain_size,
            domain_size, &outgoing_walkers);
    }
    
    std::cout << "Process " << world_rank << " sending " << outgoing_walkers.size()
         << " outgoing walkers to process " << (world_rank + 1) % world_size
         << std::endl;


    if (world_rank % 2 == 0) {
        // Send the outgoing walkers to the next process
        send_outgoing_walkers(&outgoing_walkers, world_rank, world_size);

        // Receive incoming walkers from the previous process
        receive_incoming_walkers(&incoming_walkers, world_rank, world_size);
    } else {
        // Receive incoming walkers from the previous process
        receive_incoming_walkers(&incoming_walkers, world_rank, world_size);

        // Send the outgoing walkers to the next process
        send_outgoing_walkers(&outgoing_walkers, world_rank, world_size);
    }
    std::cout << "Process " << world_rank << " received " << incoming_walkers.size()
         << " incoming walkers" << std::endl;
  }

  MPI_Finalize();

  return 0;
}