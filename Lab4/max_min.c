#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Seed differently per process
    srand(time(NULL) + rank * 100);

    // Each process generates 10 random numbers
    int nums[10];
    printf("Process %d generated: ", rank);
    for (int i = 0; i < 10; i++) {
        nums[i] = rand() % 1001;  // 0 to 1000
        printf("%d ", nums[i]);
    }
    printf("\n");

    // Find local max and min
    int local_max = nums[0], local_min = nums[0];
    for (int i = 1; i < 10; i++) {
        if (nums[i] > local_max) local_max = nums[i];
        if (nums[i] < local_min) local_min = nums[i];
    }
    printf("Process %d -> local max: %d, local min: %d\n", rank, local_max, local_min);

    // Use MPI_MAXLOC and MPI_MINLOC to find global max/min with rank
    struct { int value; int rank; } local_max_loc, local_min_loc;
    struct { int value; int rank; } global_max_loc, global_min_loc;

    local_max_loc.value = local_max;
    local_max_loc.rank  = rank;
    local_min_loc.value = local_min;
    local_min_loc.rank  = rank;

    MPI_Reduce(&local_max_loc, &global_max_loc, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_loc, &global_min_loc, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nGlobal Maximum: %d (from Process %d)\n",
               global_max_loc.value, global_max_loc.rank);
        printf("Global Minimum: %d (from Process %d)\n",
               global_min_loc.value, global_min_loc.rank);
    }

    MPI_Finalize();
    return 0;
}
