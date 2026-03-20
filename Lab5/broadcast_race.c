#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 10000000  // 10 million doubles (~80 MB)

// Part A: Manual broadcast using a for loop of MPI_Send
void MyBcast(double *buffer, int count, int rank, int size) {
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(buffer, count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(buffer, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *buffer = (double*)malloc(ARRAY_SIZE * sizeof(double));

    // Initialize buffer on rank 0
    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++)
            buffer[i] = (double)i;
    }

    // ---- Part A: MyBcast (manual for loop) ----
    MPI_Barrier(MPI_COMM_WORLD);
    double start_a = MPI_Wtime();
    MyBcast(buffer, ARRAY_SIZE, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_a = MPI_Wtime();

    // Reset buffer on non-root processes
    if (rank != 0)
        for (int i = 0; i < ARRAY_SIZE; i++) buffer[i] = 0.0;

    // Re-initialize on rank 0
    if (rank == 0)
        for (int i = 0; i < ARRAY_SIZE; i++) buffer[i] = (double)i;

    // ---- Part B: MPI_Bcast (built-in) ----
    MPI_Barrier(MPI_COMM_WORLD);
    double start_b = MPI_Wtime();
    MPI_Bcast(buffer, ARRAY_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_b = MPI_Wtime();

    if (rank == 0) {
        printf("=== Broadcast Race (Array: %d doubles, ~%d MB) ===\n",
               ARRAY_SIZE, (int)(ARRAY_SIZE * sizeof(double) / 1024 / 1024));
        printf("Processes  : %d\n", size);
        printf("MyBcast    : %.6f seconds\n", end_a - start_a);
        printf("MPI_Bcast  : %.6f seconds\n", end_b - start_b);
        printf("Speedup    : %.2fx\n", (end_a - start_a) / (end_b - start_b));
    }

    free(buffer);
    MPI_Finalize();
    return 0;
}
