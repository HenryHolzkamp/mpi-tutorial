from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cart_comm = comm.Create_cart((3, 3), periods=(True, True), reorder=True)
coords = cart_comm.Get_coords(rank)
north, south = cart_comm.Shift(0, 1)
west, east = cart_comm.Shift(1, 1)

print(f"Rank {rank} at coordinates {coords} has neighbors: "
        f"North: {north}, South: {south}, West: {west}, East: {east}")