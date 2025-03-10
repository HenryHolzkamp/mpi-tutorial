#include <mpi.h>
#include <iostream>

int main() {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int number_amount;
    if (world_rank == 0) {
        const int MAX_NUMBERS = 100;
        int numbers[MAX_NUMBERS];

        // Pick a random amount of integers to send to process 1
        srand(time(NULL));
        number_amount = (rand() / (float)RAND_MAX) * MAX_NUMBERS;

        // Send the amount of integers to process 1
        MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "0 sent " << number_amount << " numbers to 1\n";
    } else if (world_rank == 1) {
        MPI_Status status;

        // Probe for an incoming message from process 0
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

        // When probe returns, the status object has the size and other
        // attributes of the incoming message. Get the size of the message
        MPI_Get_count(&status, MPI_INT, &number_amount);

        // Allocate a buffer to hold the incoming numbers
        int* numbers = new int[number_amount];

        // Now receive the message with the allocated buffer
        MPI_Recv(numbers, number_amount, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "1 dynamically received " << number_amount << " numbers from 0\n";
        delete[] numbers;
    } 

    MPI_Finalize();
    return 0; 
}