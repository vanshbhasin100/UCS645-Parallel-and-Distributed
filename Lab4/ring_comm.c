#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value;
    int next_rank = (rank + 1) % size;
    int prev_rank = (rank - 1 + size) % size;

    if (rank == 0) {
        // Process 0 starts with 100, adds its rank (0), sends to next
        value = 100;
        value += rank;
        printf("Process %d starting value: %d, sending to Process %d\n",
               rank, value, next_rank);
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        // Wait to receive final value back from last process
        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received final value back: %d\n", value);
    } else {
        // Receive from previous process
        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received value: %d\n", rank, value);
        // Add own rank and send to next
        value += rank;
        printf("Process %d sending value: %d to Process %d\n", rank, value, next_rank);
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
