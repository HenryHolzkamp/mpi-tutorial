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
        u_old = u_local.copy()

        # Exchange ghost nodes
        if rank > 0:
            send_buf = np.array([u_local[1]], dtype='d')
            recv_buf = np.array([u_local[0]], dtype='d')
            comm.Sendrecv(send_buf, dest=rank-1, sendtag=0,
                          recvbuf=recv_buf, source=rank-1, recvtag=1)
            u_local[0] = recv_buf[0]
        if rank < size - 1:
            send_buf = np.array([u_local[-2]], dtype='d')
            recv_buf = np.array([u_local[-1]], dtype='d')
            comm.Sendrecv(send_buf, dest=rank+1, sendtag=1,
                          recvbuf=recv_buf, source=rank+1, recvtag=0)
            u_local[-1] = recv_buf[0]

        # Update interior points
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
    charge_distribution[N // 2] = -1.0
    
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