#include <mpi.h>
#include <stdio.h>

#define SIZE 8

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = SIZE / size;

    int A[SIZE] = {1, 2, 3, 4, 5, 6, 7, 8};
    int B[SIZE] = {8, 7, 6, 5, 4, 3, 2, 1};

    int local_A[SIZE], local_B[SIZE];

    // Distribute both vectors
    MPI_Scatter(A, chunk, MPI_INT, local_A, chunk, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, chunk, MPI_INT, local_B, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes its partial dot product
    int local_dot = 0;
    for (int i = 0; i < chunk; i++)
        local_dot += local_A[i] * local_B[i];

    printf("Process %d partial dot product: %d\n", rank, local_dot);

    // Reduce to get global dot product
    int global_dot = 0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nGlobal dot product: %d\n", global_dot);
        printf("Expected:           120\n");
    }

    MPI_Finalize();
    return 0;
}
