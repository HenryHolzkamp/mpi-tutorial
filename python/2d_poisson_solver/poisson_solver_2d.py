from mpi4py import MPI
import numpy as np
from functions import *

def solve_poisson_2d(n, l, f, periods=[False,False], tol=1e-8, max_iter=100000):
    # --- Initialize MPI environment --- #
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()

    # --- Create a 2D Cartesian communicator --- #
    # Caution: MPI_size must be a perfect square for this setup!
    if np.sqrt(MPI_size) % 1 == 0.0:
        dims = (int(np.sqrt(MPI_size)), int(np.sqrt(MPI_size)))
    else:
        raise ValueError("MPI size must be a perfect square for this setup.")

    cart_comm = comm.Create_cart(dims, periods=periods, reorder=True)
    north, south = cart_comm.Shift(0, 1)
    west, east = cart_comm.Shift(1, 1)

    # --- Local and global grid parameters --- #
    nx_global = n
    ny_global = n 
    nx_local = n // dims[0]
    ny_local = n // dims[1]
    dx = l / nx_global
    
    # --- Initialize buffers for potential and rhs with ghost nodes --- #
    u_local = np.zeros((nx_local + 2, ny_local + 2))
    f_local = np.zeros((nx_local + 2, ny_local + 2))

    # --- Scatter subdomains --- #
    f_local[1:-1, 1:-1] = scatter_subdomains(f, nx_global, ny_global, nx_local, ny_local, 
                                             dims, MPI_rank, cart_comm)
    

    # --- Jacobi iterations loop --- #
    for _ in range(max_iter):
        reqs = []  # Array for storing the MPI.Requests

        # --- Exchange ghost rows (north/south) --- #
        # Send north, receive south (tag = 0)
        north_send = u_local[1,1:-1].copy()
        south_recv = np.empty_like(north_send)
        reqs.append(cart_comm.Isend(north_send, dest=north, tag=0))
        reqs.append(cart_comm.Irecv(south_recv, source=south, tag=0))
        
        # Send south, receive north (tag = 1)
        south_send = u_local[-2,1:-1].copy()
        north_recv = np.empty_like(south_send)
        reqs.append(cart_comm.Isend(south_send, dest=south, tag=1))
        reqs.append(cart_comm.Irecv(north_recv, source=north, tag=1))
        
        # --- Exchange ghost columns (west/east) --- #
        # Send west, receive east (tag = 2)
        west_send = u_local[1:-1,1].copy()
        east_recv = np.empty_like(west_send)
        reqs.append(cart_comm.Isend(west_send, dest=west, tag=2))
        reqs.append(cart_comm.Irecv(east_recv, source=east, tag=2))
        
        # Send east, receive west (tag = 3)
        east_send = u_local[1:-1,-2].copy()
        west_recv = np.empty_like(east_send)
        reqs.append(cart_comm.Isend(east_send, dest=east, tag=3))
        reqs.append(cart_comm.Irecv(west_recv, source=west, tag=3))

        MPI.Request.Waitall(reqs)

        u_old = u_local.copy()

        # --- Jacobi update for interior points --- #
        u_local[1:-1, 1:-1] = 0.25 * (
            u_old[2:, 1:-1] + u_old[:-2, 1:-1] +
            u_old[1:-1, 2:] + u_old[1:-1, :-2] -
            dx * dx * f_local[1:-1, 1:-1]
        )

        # --- Check for convergence --- #
        local_diff = np.linalg.norm(u_local - u_old, ord=np.inf)
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        if global_diff < tol:
            break

    # --- Gather subdomains --- # 
    u_global = gather_subdomains(u_local, nx_global, ny_global, nx_local, ny_local, dims, 
                    MPI_rank, comm = MPI.COMM_WORLD)
    
    # --- Root process returns result --- #
    if MPI_rank == 0:
        print("global_diff", global_diff)
        return u_global

if __name__ == "__main__":
    N = 200  # Number of grid points per dimension
    L = 10   # Domain length

    # Example: 
    f = np.zeros((N, N))
    f[N // 4, N // 4] = -1
    f[3 * N // 4, N // 4] = 1
    f[N // 4, 3 * N // 4] = 1
    f[3 * N // 4, 3 * N // 4] = -1

    solution = solve_poisson_2d(n = N, l = L, f = f, periods=[True,True])

    # --- Plot solution --- #
    if solution is not None:
        if MPI.COMM_WORLD.Get_rank() == 0:
            import matplotlib.pyplot as plt
            x = np.linspace(0, L, N)
            y = np.linspace(0, L, N)
            X, Y = np.meshgrid(x, y, indexing='ij')
            plt.contourf(X, Y, solution, 50, cmap='RdBu')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.colorbar(label="u(x, y)")
            plt.savefig("solution_2d_plot.png")
            print("Plot saved as solution_2d_plot.png")
