from mpi4py import MPI
import numpy as np

def solve_poisson_1d(N, L, charge_distribution, tol=1e-10, max_iter=10000):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dx = L / (N - 1)
    local_N = N // size + 2  # Include ghost nodes

    # Initialize local arrays
    u_local = np.zeros(local_N)
    f_local = np.zeros(local_N)

    # Distribute the charge distribution array among the processes
    if rank == 0:
        charge_distribution = np.array(charge_distribution, dtype='d')
    else:
        charge_distribution = None

    local_charge = np.zeros(local_N - 2, dtype='d')
    comm.Scatter(charge_distribution, local_charge, root=0)

    # Set the local charge distribution
    f_local[1:-1] = local_charge

    # Jacobi iteration
    for iteration in range(max_iter):
        # Exchange ghost nodes using non-blocking communication
        send_buf_left = np.array([u_local[1]], dtype='d')
        recv_buf_left = np.array([u_local[0]], dtype='d')

        send_buf_right = np.array([u_local[-2]], dtype='d')
        recv_buf_right = np.array([u_local[-1]], dtype='d')

        reqs = []
        if rank > 0:
            reqs.append(comm.Isend(send_buf_left, dest=rank-1, tag=0))
            reqs.append(comm.Irecv(recv_buf_left, source=rank-1, tag=1))
        else:
            reqs.append(comm.Isend(send_buf_left, dest=size-1, tag=0))
            reqs.append(comm.Irecv(recv_buf_left, source=size-1, tag=1))

        if rank < size - 1:
            reqs.append(comm.Isend(send_buf_right, dest=rank+1, tag=1))
            reqs.append(comm.Irecv(recv_buf_right, source=rank+1, tag=0))
        else:
            reqs.append(comm.Isend(send_buf_right, dest=0, tag=1))
            reqs.append(comm.Irecv(recv_buf_right, source=0, tag=0))

        MPI.Request.Waitall(reqs)

        u_local[0] = recv_buf_left[0]
        u_local[-1] = recv_buf_right[0]

        # Update interior points
        u_old = u_local.copy()
        for i in range(1, local_N - 1):
            u_local[i] = 0.5 * (u_old[i-1] + u_old[i+1] - dx**2 * f_local[i])

        # Check for convergence
        local_diff = np.linalg.norm(u_local - u_old, ord=np.inf)
        global_diff = comm.allreduce(local_diff, op=MPI.MAX)
        if global_diff < tol:
            break

    # Gather results to root process
    u_global = None
    if rank == 0:
        u_global = np.zeros(N)
    comm.Gather(u_local[1:-1], u_global, root=0)

    if rank == 0:
        print("global_diff", global_diff)
        return u_global

if __name__ == "__main__":
    N = 160  # Number of grid points
    L = 1.0  # Length of the domain

    # Example charge distribution: -1.0 at the middle of the domain
    charge_distribution = np.zeros(N)
    charge_distribution[N // 4] = -1.0
    charge_distribution[3 * N // 4] = 1.0

    solution = solve_poisson_1d(N, L, charge_distribution)
    
    if solution is not None:
        print("Solution:", solution)

        if MPI.COMM_WORLD.Get_rank() == 0:
            import matplotlib.pyplot as plt
            x = np.linspace(0, L, N)
            plt.plot(x, solution)
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.savefig("solution_plot.png")  # Save the plot to a file
            print("Plot saved as solution_plot.png")
