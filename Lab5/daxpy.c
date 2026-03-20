#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N (1 << 16)  // 2^16 = 65536

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 2.5;
    int chunk = N / size;

    double *X = (double*)malloc(chunk * sizeof(double));
    double *Y = (double*)malloc(chunk * sizeof(double));

    // Each process initializes its own chunk
    for (int i = 0; i < chunk; i++) {
        X[i] = (double)(rank * chunk + i);
        Y[i] = (double)(rank * chunk + i) * 0.5;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // DAXPY: X[i] = a * X[i] + Y[i]
    for (int i = 0; i < chunk; i++) {
        X[i] = a * X[i] + Y[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("=== DAXPY (N=2^16=%d, a=%.1f) ===\n", N, a);
        printf("Processes : %d\n", size);
        printf("Time      : %.6f seconds\n", end - start);
        printf("Sample X[0] = %.2f (expected: a*0 + 0 = 0.00)\n", X[0]);
        printf("Sample X[1] = %.2f (expected: a*1 + 0.5 = 3.00)\n", X[1]);
    }

    free(X);
    free(Y);
    MPI_Finalize();
    return 0;
}
