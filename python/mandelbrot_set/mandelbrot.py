from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height), dtype=int)
    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j*r2[j], max_iter)
    return n3

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parameters for the Mandelbrot set
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 800, 800
    max_iter = 256

    # Divide the work among processes
    local_width = width // size
    local_xmin = xmin + rank * local_width * (xmax - xmin) / width
    local_xmax = xmin + (rank + 1) * local_width * (xmax - xmin) / width

    # Each process computes its part of the Mandelbrot set
    local_mandelbrot = mandelbrot_set(local_xmin, local_xmax, ymin, ymax, local_width, height, max_iter)

    # Gather the results to the root process
    if rank == 0:
        mandelbrot_image = np.empty((width, height), dtype=int)
    else:
        mandelbrot_image = None

    comm.Gather(local_mandelbrot, mandelbrot_image, root=0)

    # Root process visualizes the result
    if rank == 0:
        plt.imshow(mandelbrot_image.T, extent=[xmin, xmax, ymin, ymax], cmap='hot')
        plt.colorbar()
        plt.title("Mandelbrot Set")
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.savefig("mandelbrot_set.png")
        # plt.show()

if __name__ == "__main__":
    main()