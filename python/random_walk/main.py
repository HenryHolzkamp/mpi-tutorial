from mpi4py import MPI
from functions import *
import numpy as np


def main():
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()

    domain_size  = 840
    max_walk_size = 10
    num_walkers_per_proc = 100

    subdomain_start, subdomain_size = decompose_domain(domain_size, 
                                                       MPI_rank, MPI_size)
    local_walkers = initialize_walkers(num_walkers_per_proc, 
                            max_walk_size, subdomain_start, subdomain_size)

    max_exchanges = int(max_walk_size / subdomain_size) + 1

    for _ in range(max_exchanges):
        
        local_walkers, outgoing_walkers = walk(local_walkers, subdomain_start, 
                                               subdomain_size, domain_size)

        nr_incoming_walkers = exchange_outgoing_walkers_size(outgoing_walkers, 
                                                MPI_rank, MPI_size, comm)


        incoming_walkers = exchange_walkers(outgoing_walkers, nr_incoming_walkers,
                                                MPI_rank, MPI_size, comm)
        
        local_walkers = np.concat([local_walkers, incoming_walkers], axis=None)

    return None


if __name__ == '__main__':
    main()