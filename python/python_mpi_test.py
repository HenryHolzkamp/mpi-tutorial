from mpi4py import MPI
import numpy as np

# Setup Communicator
comm = MPI.COMM_WORLD

# Get the number of processes
world_size = comm.Get_size()

# Get the rank of the process
world_rank = comm.Get_rank()

if world_rank == 0:
    number = np.random.rand()
    comm.send(number, dest=1, tag=0)
    print(f"Process {world_rank} sent {number} to process 1.")
else:
    recv_number = comm.recv(source=0, tag=0)
    print(f"Process 1 got {recv_number}.")