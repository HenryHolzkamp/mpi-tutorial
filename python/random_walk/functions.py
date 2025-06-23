import numpy as np
from mpi4py import MPI

class Walker():
    def __init__(self, location, nr_of_steps_left):
        self.location = location
        self.nr_of_steps_left = nr_of_steps_left 


def decompose_domain(domain_size, MPI_rank, MPI_size):
    subdomain_size = domain_size / MPI_size
    subdomain_start = subdomain_size * MPI_rank

    # Give remainder to last process
    if (MPI_rank == MPI_size - 1):
        subdomain_size += domain_size % MPI_size
    
    return subdomain_start, subdomain_size


def initialize_walkers(num_walkers_per_proc, max_walk_size,
        subdomain_start, subdomain_size):

    local_walkers = np.array([None] * num_walkers_per_proc, dtype=Walker)
    
    # Initialize the initial positions of the walkers 
    # inside the subdomain
    for i in range(num_walkers_per_proc):
        location = int(subdomain_start \
            + i / num_walkers_per_proc * subdomain_size)
        

        local_walkers[i] = Walker(location, int(np.random.randint(0, max_walk_size + 1)))
            
    return local_walkers


def walk(local_walkers, subdomain_start, subdomain_size, domain_size):
    outgoing_walkers = np.array([], dtype = Walker)
    i_of_outgoing_walkers = []

    for i, walker in enumerate(local_walkers):
        while (walker.nr_of_steps_left > 0):
            walker.location += 1
            walker.nr_of_steps_left -= 1
            
            if walker.location > subdomain_start + subdomain_size:
                walker.location = walker.location % domain_size
                outgoing_walkers = np.append(outgoing_walkers, walker)
                i_of_outgoing_walkers.append(i)
                break    

    local_walkers = np.delete(local_walkers, i_of_outgoing_walkers)

    return local_walkers, outgoing_walkers


def exchange_outgoing_walkers_size(outgoing_walkers,
            MPI_rank, MPI_size, comm=MPI.COMM_WORLD):
    
    requests = []
    nr_outgoing_walkers = np.array([outgoing_walkers.size])
    nr_incoming_walkers = np.empty(1, dtype=np.int64)

    # First send number of outgoing walkers
    # tag 0: size communication
    requests.append(comm.Isend(nr_outgoing_walkers,
            dest=(MPI_rank + 1) % MPI_size, tag=0))

    # Then receive the number of incoming walkers
    requests.append(comm.Irecv(nr_incoming_walkers,
            source=(MPI_rank - 1) % MPI_size, tag=0))

    MPI.Request.Waitall(requests)

    return nr_incoming_walkers[0]


def exchange_walkers(outgoing_walkers, nr_incoming_walkers, MPI_rank, MPI_size, 
            comm=MPI.COMM_WORLD):
    
    requests = []
    incoming_walkers_locations = np.empty(nr_incoming_walkers, dtype=np.int64)
    incoming_walkers_steps = np.empty(nr_incoming_walkers, dtype=np.int64)

    outgoing_walkers_locations = np.array([walker.location for walker in outgoing_walkers])
    outgoing_walkers_steps = np.array([walker.nr_of_steps_left for walker in outgoing_walkers])

    # First send out outgoing walkers
    # tag 1: walker location communication
    # tag 2: walker steps communication
    if outgoing_walkers.size > 0:
        requests.append(comm.Isend(outgoing_walkers_locations, 
                dest=(MPI_rank + 1) % MPI_size, tag=1))
        
        requests.append(comm.Isend(outgoing_walkers_steps, 
                dest=(MPI_rank + 1) % MPI_size, tag=2))


    # Then receive the attributes of the incoming walkers
    if nr_incoming_walkers > 0:
        requests.append(comm.Irecv(incoming_walkers_locations,
            source=(MPI_rank - 1) % MPI_size, tag=1))

        requests.append(comm.Irecv(incoming_walkers_steps,
            source=(MPI_rank - 1) % MPI_size, tag=2))

    MPI.Request.Waitall(requests)

    if nr_incoming_walkers > 0:
        incoming_walkers = np.array([Walker(incoming_walkers_locations[i],
                                    incoming_walkers_steps[i]) 
                                    for i in range(nr_incoming_walkers)])
    else:
        incoming_walkers = None

    return incoming_walkers