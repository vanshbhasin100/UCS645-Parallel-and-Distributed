#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_VALUE 10000

int is_perfect(int n) {
    if (n < 2) return 0;
    int sum = 1;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i)
                sum += n / i;
        }
    }
    return sum == n;
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
        int perfect[100];
        int perfect_count = 0;

        // Send initial numbers to all slaves
        for (int i = 1; i < size && next_num <= MAX_VALUE; i++) {
            MPI_Send(&next_num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            next_num++;
        }

        int active_slaves = (size - 1) < (MAX_VALUE - 1) ? (size - 1) : (MAX_VALUE - 1);
        int done_slaves = 0;

        while (done_slaves < active_slaves) {
            // a. Receive from any slave (zero = just starting)
            MPI_Recv(&received, 1, MPI_INT, MPI_ANY_SOURCE, 0,
                     MPI_COMM_WORLD, &status);

            // b. Positive = perfect, negative = not perfect, zero = starting
            if (received > 0) {
                perfect[perfect_count++] = received;
            }

            // c. Send next number or termination signal
            if (next_num <= MAX_VALUE) {
                MPI_Send(&next_num, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                next_num++;
            } else {
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                done_slaves++;
            }
        }

        printf("=== Perfect Numbers up to %d ===\n", MAX_VALUE);
        printf("Found %d perfect numbers:\n", perfect_count);
        for (int i = 0; i < perfect_count; i++)
            printf("%d ", perfect[i]);
        printf("\n");

    } else {
        // ---- SLAVE ----
        int num;
        // Send zero to signal starting
        int zero = 0;
        MPI_Send(&zero, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        while (1) {
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Termination signal
            if (num < 0) break;

            // Return positive if perfect, negative if not
            int result = is_perfect(num) ? num : -num;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
