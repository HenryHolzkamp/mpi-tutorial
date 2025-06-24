import numpy as np
from mpi4py import MPI

def scatter_subdomains(f, nx_global, ny_global, nx_local, ny_local, dims, 
                       MPI_rank, comm = MPI.COMM_WORLD):
    # --- Prepare buffer ---
    # 2D arrays
    subdomain = np.empty((nx_local, ny_local), dtype=np.float64)
    
    # 1D arrays for communication
    global_sendbuf = np.empty(nx_global * ny_global, dtype=np.float64)
    local_recvbuf = np.empty(nx_local * ny_local, dtype=np.float64)

    # --- Communication ---
    if MPI_rank == 0:
        # Prepare 2D payload on root process
        domain = f
        counter = 0 # counter for the 1D array

        # Set up 2D subdomain, insert the flattened subdomain to sendbuf 
        for row_i in range(dims[0]):
            for column_j in range(dims[1]):
                subdomain = domain[row_i*nx_local:(row_i+1)*nx_local,
                                   column_j*ny_local:(column_j+1)*ny_local]
                global_sendbuf[counter:counter+nx_local*ny_local] = subdomain.flatten()
                counter += nx_local * ny_local

    comm.Scatter(global_sendbuf, local_recvbuf, root=0)
    subdomain = local_recvbuf.copy().reshape((nx_local, ny_local))

    del global_sendbuf, local_recvbuf

    return subdomain


def gather_subdomains(u_local, nx_global, ny_global, nx_local, ny_local, dims, 
                       MPI_rank, comm = MPI.COMM_WORLD):
    # --- Prepare buffer ---
    # 2D arrays
    domain = np.empty((nx_global, ny_global), dtype=np.float64)

    # 1D arrays for communication
    local_sendbuf = u_local[1:-1,1:-1].copy().flatten()
    global_recvbuf = np.empty(nx_global * ny_global, dtype=np.float64)

    # Gather all subdomains onto root
    comm.Gather(local_sendbuf, global_recvbuf, root=0)

    # --- Communication ---
    if MPI_rank == 0:
        # Prepare 2D payload on root process
        counter = 0 # counter for the 1D array

        # Set up 2D subdomain, insert the flattened subdomain to sendbuf 
        for row_i in range(dims[0]):
            for column_j in range(dims[1]):
                subdomain = global_recvbuf[counter:counter
                                        +nx_local*ny_local].reshape((nx_local, ny_local))
                domain[row_i*nx_local:(row_i+1)*nx_local,
                                   column_j*ny_local:(column_j+1)*ny_local] = subdomain
                
                counter += nx_local * ny_local

    del global_recvbuf, local_sendbuf

    return domain