from mpi4py import MPI
import numpy as np

# Setup Communicator
comm = MPI.COMM_WORLD

# Get the number of processes
MPI_size = comm.Get_size()

# Get the rank of the process
MPI_rank = comm.Get_rank()

# ... initialize MPI ... (slide 11) #
nr_rand_ints = 1000
rand_ints = np.random.randint(0, 101, nr_rand_ints)
local_sum = np.sum(rand_ints)
sendbuf = np.array([local_sum], dtype = 'i')

recvbuf = np.empty(1, dtype = 'i')
comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)

if MPI_rank == 0:
    global_sum = recvbuf[0]
    global_average = global_sum / (MPI_size * nr_rand_ints)
    print(f"Global average: {global_average}")