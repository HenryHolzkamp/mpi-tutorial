from mpi4py import MPI
import numpy as np

# Init MPI (RUN WITH 4 PROCESSES)
comm = MPI.COMM_WORLD
MPI_rank = comm.Get_rank()
MPI_size = comm.Get_size()

if MPI_size != 4:
    raise ValueError("MPI_size needs to be 4!")

# Prepare buffer
recv_buf = np.zeros((2,2), dtype=np.int64)

# Prepare payload on root process
if MPI_rank == 0:
    test = np.array([[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,15,16]])
    print("0 sends:\n", test)
else:
    test = np.zeros((4,4), dtype=np.int64)

# Scatter test across the processes
comm.Scatter(test, recv_buf, root=0)

# Process 0 doesn't receive the subdomain in a way
# we'd hope for -> We need to solve this
# See subdomain_distr_right.py for the fix
if MPI_rank ==0:
    print("0 received: \n", recv_buf)