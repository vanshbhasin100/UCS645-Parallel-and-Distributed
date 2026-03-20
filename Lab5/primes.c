#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VALUE 1000

int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i <= (int)sqrt((double)n); i += 2)
        if (n % i == 0) return 0;
    return 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Need at least 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        // ---- MASTER ----
        int next_num = 2;
        int received;
        MPI_Status status;
        int primes[MAX_VALUE];
        int prime_count = 0;

        // Send initial numbers to all slaves
        for (int i = 1; i < size && next_num <= MAX_VALUE; i++) {
            MPI_Send(&next_num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            next_num++;
        }

        // Keep receiving and sending until all numbers are tested
        int active_slaves = (size - 1) < (MAX_VALUE - 1) ? (size - 1) : (MAX_VALUE - 1);
        int done_slaves = 0;

        while (done_slaves < active_slaves) {
            // a. Receive from any slave
            MPI_Recv(&received, 1, MPI_INT, MPI_ANY_SOURCE, 0,
                     MPI_COMM_WORLD, &status);

            // b. If positive = prime, if negative = not prime
            if (received > 0) {
                primes[prime_count++] = received;
            }

            // c. Send next number or termination signal (-1)
            if (next_num <= MAX_VALUE) {
                MPI_Send(&next_num, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                next_num++;
            } else {
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                done_slaves++;
            }
        }

        // Print all found primes
        printf("=== Prime Numbers up to %d ===\n", MAX_VALUE);
        printf("Found %d primes:\n", prime_count);
        for (int i = 0; i < prime_count; i++) {
            printf("%d ", primes[i]);
            if ((i + 1) % 15 == 0) printf("\n");
        }
        printf("\n");

    } else {
        // ---- SLAVE ----
        int num;
        // Send initial zero to signal ready
        int zero = 0;
        MPI_Send(&zero, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (1) {
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Termination signal
            if (num < 0) break;

            // Test and return positive if prime, negative if not
            int result = is_prime(num) ? num : -num;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
