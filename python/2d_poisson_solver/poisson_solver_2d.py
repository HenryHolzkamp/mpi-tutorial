from mpi4py import MPI
import numpy as np

def solve_poisson_2d(N, L, f, tol=1e-8, max_iter=1000):
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()

    # Create a 2D Cartesian communicator
    # Caution: MPI_size must be a perfect square for this setup!
    if np.sqrt(MPI_size) % 1 == 0.0:
        dims = (int(np.sqrt(MPI_size)), int(np.sqrt(MPI_size)))
    else:
        raise ValueError("MPI size must be a perfect square for this setup.")

    # periods = [True, True] # Periodic boundary conditions in both dimensions
    periods = [False, False] 
    cart_comm = comm.Create_cart(dims, periods=periods, reorder=True)
    north, south = cart_comm.Shift(0, 1)
    west, east = cart_comm.Shift(1, 1)

    # Local grid MPI_size (interior points only)
    nx_global = N
    nx_local = nx_global // dims[0]
    
    # Add ghost layers
    u_local = np.zeros((nx_local + 2, nx_local + 2))
    f_local = np.zeros((nx_local + 2, nx_local + 2))

    # ...existing code...

    # Scatter the charge distribution correctly (2D block decomposition)
    if MPI_rank == 0:
        f = np.array(f, dtype='d')
    else:
        f = None

    print(MPI_rank, 'f', np.max(f))
    f_flat_global = np.empty(nx_global * nx_global, dtype='d')
    displs = 0

    # Prepare counts and displacements for Scatterv
    if MPI_rank == 0:
        for i in range(dims[0]):
            for j in range(dims[1]):
                start_row = i * nx_local
                start_col = j * nx_local
                # Each block is nx_local x nx_local, flatten in row-major order
                f_flat_global[displs:displs+nx_local*nx_local] = f[start_row:start_row+nx_local, start_col:start_col+nx_local].flatten()
                displs += nx_local * nx_local
    else:
        f_flat_global = None

    # Scatter 
    f_flat_local = np.zeros(nx_local * nx_local, dtype='d')

    comm.Scatter(f_flat_global, f_flat_local, root=0)
    f_local[1:-1, 1:-1] = f_flat_local.reshape((nx_local, nx_local))
    print(MPI_rank, 'f_local', np.max(np.abs(f_local)))
    
    dx = L / nx_global
    
    for _ in range(max_iter):
        reqs = []

        # Exchange ghost rows (north/south)
        # Send north, receive south
        north_send = u_local[1:-1,1].copy()
        south_recv = np.empty_like(north_send)
        reqs.append(cart_comm.Isend(north_send, dest=north, tag=0))
        reqs.append(cart_comm.Irecv(south_recv, source=south, tag=0))
        
        # Send south, receive north
        south_send = u_local[1:-1,1].copy()
        north_recv = np.empty_like(south_send)
        reqs.append(cart_comm.Isend(south_send, dest=south, tag=1))
        reqs.append(cart_comm.Irecv(north_recv, source=north, tag=1))
        
        # Exchange ghost columns (west/east)
        # Send west, receive east
        west_send = u_local[1:-1,1].copy()
        east_recv = np.empty_like(west_send)
        reqs.append(cart_comm.Isend(west_send, dest=west, tag=2))
        reqs.append(cart_comm.Irecv(east_recv, source=east, tag=2))
        
        # Send east, receive west
        east_send = u_local[1:-1,-2].copy()
        west_recv = np.empty_like(east_send)
        reqs.append(cart_comm.Isend(east_send, dest=east, tag=3))
        reqs.append(cart_comm.Irecv(west_recv, source=west, tag=3))

        MPI.Request.Waitall(reqs)

        u_old = u_local.copy()
        # Jacobi update for interior points
        u_local[1:-1, 1:-1] = 0.25 * (
            u_old[2:, 1:-1] + u_old[:-2, 1:-1] +
            u_old[1:-1, 2:] + u_old[1:-1, :-2] -
            dx * dx * f_local[1:-1, 1:-1]
        )

        # Convergence check
        local_diff = np.linalg.norm(u_local - u_old, ord=np.inf)
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        if global_diff < tol:
            break

    # Gather the solution to the root process
    u_global = None
    if MPI_rank == 0:
        u_global = np.zeros(nx_global * nx_global)

    local_flat = u_local[1:-1, 1:-1].flatten()

    comm.Gather(local_flat, u_global, root=0)

    if MPI_rank == 0:
        u_global = u_global.reshape((nx_global, nx_global))
        print("global_diff", global_diff)
        return u_global

if __name__ == "__main__":
    N = 900 # Number of grid points per dimension
    L = 1.0 # Domain length

    # Example: point charges at four corners
    f = np.zeros((N, N))
    f[N // 2, N // 2] = -1
    # f[N // 4, N // 4] = -1
    # f[3 * N // 4, 3 * N // 4] = 1
    
    solution = solve_poisson_2d(N, L, f)

    if solution is not None:
        print("Solution shape:", solution.shape)
        if MPI.COMM_WORLD.Get_rank() == 0:
            import matplotlib.pyplot as plt
            x = np.linspace(0, L, N)
            y = np.linspace(0, L, N)
            X, Y = np.meshgrid(x, y, indexing='ij')
            plt.contourf(X, Y, solution, 50)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.colorbar(label="u(x, y)")
            plt.savefig("solution_2d_plot.png")
            print("Plot saved as solution_2d_plot.png")
