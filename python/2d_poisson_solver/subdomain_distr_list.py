from mpi4py import MPI
import numpy as np

# Init MPI (RUN WITH 4 PROCESSES)
comm = MPI.COMM_WORLD
MPI_rank = comm.Get_rank()
MPI_size = comm.Get_size()

if MPI_size != 4:
    raise ValueError("MPI_size needs to be 4!")

# --- Prepare buffer ---
# 2D arrays
subdomain = np.empty((2,2), dtype=np.int64)
domain = np.empty((4,4), dtype=np.int64)

# 2D arrays for communication
# global_sendbuf serves as a list of the subdomains
global_sendbuf = np.empty((2,8), dtype=np.int64)
local_recvbuf = np.empty((2,2), dtype=np.int64)

# --- Communication ---
if MPI_rank == 0:
    # Prepare 2D payload on root process
    domain = np.array([[1,2,5,6],[3,4,7,8],[9,10,13,14],[11,12,15,16]])
    counter = 0 # counter for the 1D array

    # Set up 2D subdomain, insert the flattened subdomain to sendbuf 
    for row_i in range(2):
        for column_j in range(2):
            subdomain = domain[row_i*2:row_i*2+2,column_j*2:column_j*2+2]
            global_sendbuf[0:2, counter:counter+2] = subdomain
            counter += 2
else:
    global_sendbuf = None

# Scatter the sendbuf across the processes
comm.Scatter(global_sendbuf, local_recvbuf, root=0)

# Reshaping the flattened recvbuf yields the right subdomain
subdomain = local_recvbuf.reshape(2,2)
print(f"{MPI_rank} has received the subdomain\n{subdomain}")
