#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_SIZE 1000000LL  // 500 million elements

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Rank 0 sets multiplier, broadcasts to all
    double multiplier = 3.0;
    if (rank == 0) {
        printf("=== Distributed Dot Product (N=500M) ===\n");
        printf("Processes  : %d\n", size);
        printf("Multiplier : %.1f\n", multiplier);
    }
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 2: Each process generates its own local chunk
    long long chunk = TOTAL_SIZE / size;

    double *A = (double*)malloc(chunk * sizeof(double));
    double *B = (double*)malloc(chunk * sizeof(double));
    if (!A || !B) {
    printf("Memory allocation failed on rank %d\n", rank);
    MPI_Finalize();
    return 1;
}

    for (long long i = 0; i < chunk; i++) {
        A[i] = 1.0;
        B[i] = 2.0 * multiplier;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Step 3: Each process computes its local dot product
    double local_dot = 0.0;
    for (long long i = 0; i < chunk; i++) {
        local_dot += A[i] * B[i];
    }

    // Step 4: Reduce all local dot products to rank 0
    double final_result = 0.0;
    MPI_Reduce(&local_dot, &final_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Result     : %.2f\n", final_result);
        printf("Expected   : %.2f\n", (double)TOTAL_SIZE * 1.0 * 2.0 * multiplier);
        printf("Time       : %.6f seconds\n", end - start);
    }

    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}
