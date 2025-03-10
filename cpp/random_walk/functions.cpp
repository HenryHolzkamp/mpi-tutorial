# include "functions.h"

void decompose_domain(int domain_size, int world_rank, int world_size,
    int* subdomain_start, int* subdomain_size) {
    if (world_size > domain_size) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    *subdomain_start = domain_size / world_size * world_rank;
    *subdomain_size = domain_size / world_size;

    if (world_rank == world_size - 1) {
        // Give remainder to last process
        *subdomain_size += domain_size % world_size;
    }
}


void initialize_walkers(int num_walkers_per_proc, int max_walk_size,
                        int subdomain_start, int subdomain_size,
                        std::vector<Walker>* incoming_walkers) {

    srand(time(NULL) * subdomain_start);
    Walker walker;
    
    for (int i = 0; i < num_walkers_per_proc; i++) {

        // Initialize walkers in the middle of the subdomain
        walker.location = subdomain_start + subdomain_size / 2;
        walker.num_steps_left_in_walk = (rand() / (float)RAND_MAX) * max_walk_size;
        incoming_walkers->push_back(walker);
    }
}


void walk(Walker* walker, int subdomain_start, int subdomain_size,
    int domain_size, std::vector<Walker>* outgoing_walkers) {

    while (walker->num_steps_left_in_walk > 0) {
        if (walker->location == subdomain_start + subdomain_size) {
        // Take care of the case when the walker is at the end
        // of the domain by wrapping it around to the beginning
        if (walker->location == domain_size) {
                walker->location = 0;
            }
            outgoing_walkers->push_back(*walker);
        break;
        } else {
            walker->num_steps_left_in_walk--;
            walker->location++;
        }
    }
}


void send_outgoing_walkers(std::vector<Walker>* outgoing_walkers,
    int world_rank, int world_size) {
    
    // Send the data as an array of MPI_BYTEs to the next process.
    // The last process sends to process zero.
    MPI_Send((void*)outgoing_walkers->data(),
        outgoing_walkers->size() * sizeof(Walker), MPI_BYTE,
        (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
    
        // Clear the outgoing walkers list
    outgoing_walkers->clear();
}


void receive_incoming_walkers(std::vector<Walker>* incoming_walkers,
    int world_rank, int world_size) {
    // Probe for new incoming walkers
    MPI_Status status;

    // Receive from the process before you. If you are process zero,
    // receive from the last process
    int incoming_rank =
    (world_rank == 0) ? world_size - 1 : world_rank - 1;
    MPI_Probe(incoming_rank, 0, MPI_COMM_WORLD, &status);

    // Resize your incoming walker buffer based on how much data is
    // being received
    int incoming_walkers_size;
    MPI_Get_count(&status, MPI_BYTE, &incoming_walkers_size);
    incoming_walkers->resize(incoming_walkers_size / sizeof(Walker));

    MPI_Recv((void*)incoming_walkers->data(), incoming_walkers_size,
    MPI_BYTE, incoming_rank, 0, MPI_COMM_WORLD,
    MPI_STATUS_IGNORE);
}