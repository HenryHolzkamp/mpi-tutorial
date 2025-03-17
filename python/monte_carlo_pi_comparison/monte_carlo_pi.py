import numpy
from mpi4py import MPI
import time
import argparse

real_pi = 3.141592653589793

def monte_carlo_pi(n, seed):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    iterations_per_process = n // size

    rng = numpy.random.default_rng(seed * rank)    
    
    count = 0
    for i in range(iterations_per_process):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x * x + y * y < 1:
            count += 1

    count = numpy.array(count, dtype='int')
    count = comm.reduce(count, op=MPI.SUM, root=0)

    if rank == 0:
        return (4 * count / n)
    else:
        return None
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monte Carlo Pi Calculation')
    parser.add_argument('iterations', type=int, help='Number of iterations')
    args = parser.parse_args()

    try:
        n = int(args.iterations)
        if n <= 0:
            raise ValueError("Number of iterations must be a positive integer.")
    except ValueError as e:
        parser.error(str(e))

    seed = 42
    start_time = time.time()
    pi = monte_carlo_pi(n, seed)
    end_time = time.time()
    if pi is not None:
        print(f"Pi: {pi}, Error: {abs(pi - real_pi)}, Time: {end_time - start_time} seconds")