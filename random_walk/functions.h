#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <mpi.h>

# include <iostream>
#include <vector>
# include <cstdlib>

void decompose_domain(int domain_size, int world_rank, int world_size,
    int* subdomain_start, int* subdomain_size);

typedef struct {
    int location;
    int num_steps_left_in_walk;
} Walker;  

void initialize_walkers(int num_walkers_per_proc, int max_walk_size,
                       int subdomain_start, int subdomain_size,
                       std::vector<Walker>* incoming_walkers);

void walk(Walker* walker, int subdomain_start, int subdomain_size, 
    int domain_size, std::vector<Walker>* outgoing_walkers);

void send_outgoing_walkers(std::vector<Walker>* outgoing_walkers,
        int world_rank, int world_size);

void receive_incoming_walkers(std::vector<Walker>* incoming_walkers,
        int world_rank, int world_size);

#endif // FUNCTIONS_H