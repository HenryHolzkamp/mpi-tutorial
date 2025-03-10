#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>

std::vector<float> create_random_nums(int num_elements) {
    std::vector<float> rand_nums(num_elements);
    // Seed the random number generator to get different results each time
    srand(time(NULL));

    for (int i = 0; i < num_elements; i++) {
        rand_nums[i] = (rand() / (float) RAND_MAX);
    }
    return rand_nums;
}

float compute_avg(std::vector<float> nums) {
    float sum = 0;
    for (int i = 0; i < nums.size(); i++) {
        sum += nums[i];
    }
    return sum / nums.size();
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Generate a random number array on the root process
    std::vector<float> rand_nums;
    int num_elements_per_proc = 10000;
    if (world_rank == 0) {
        rand_nums = create_random_nums(num_elements_per_proc * world_size);
    }

    // For each process, create a buffer that will hold a subset of the entire array
    std::vector<float> sub_rand_nums(num_elements_per_proc);

    // Scatter the random numbers from the root process to all processes in the MPI world
    MPI_Scatter(rand_nums.data(), num_elements_per_proc, MPI_FLOAT, sub_rand_nums.data(), num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Compute the average of the subset
    float sub_avg = compute_avg(sub_rand_nums);

    // Gather all partial averages down to the root process
    std::vector<float> sub_avgs(world_size);
    if (world_rank == 0) {
        std::vector<float> sub_avgs(world_size);
    }

    MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Compute the total average of all numbers
    if (world_rank == 0) {
        float avg = compute_avg(sub_avgs);
        printf("Avg of all elements is %f\n", avg);
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}