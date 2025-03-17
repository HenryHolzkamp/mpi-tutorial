#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

const double real_pi = 3.141592653589793;

double monte_carlo_pi(int n, unsigned int* seed) {
    int rank, size, iterations_per_process;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    iterations_per_process = n / size;

    int count = 0;
    *seed *= rank; // Seed for random number generator

    for (int i = 0; i < iterations_per_process; ++i) {
        double x = (double)rand_r(seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(seed) / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y < 1.0) {
            count++;
        }
    }

    int global_count;
    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        return (4.0 * global_count / (n));
    } else {
        return 0.0;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int n;
    try {
        if (argc != 2 || (n = std::stoi(argv[1])) <= 0) {
            throw std::invalid_argument("Invalid number of iterations");
        }
    } catch (const std::exception& e) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <number_of_iterations>" << std::endl;
            std::cerr << "Error: " << e.what() << std::endl;
            std::cerr << "Please provide a positive integer greater than 0." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    unsigned int seed = 42;

    // Start the timer
    double start_time = MPI_Wtime();

    double pi = monte_carlo_pi(n, &seed);

    // Stop the timer
    double end_time = MPI_Wtime();

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "Pi: " << pi << ", Error: " << fabs(pi - real_pi) << std::endl;
        std::cout << "Execution time: " << end_time - start_time << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}