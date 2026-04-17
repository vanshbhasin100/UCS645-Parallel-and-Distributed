/*
 * Assignment 7 - Problem 1
 * CUDA program where all threads perform different tasks:
 * a. Thread 0: Iterative sum of first N integers
 * b. Thread 1: Formula-based sum of first N integers
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Kernel: each thread does a different task
__global__ void sumKernel(int *input, long long *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        // Task a: Iterative sum of first N integers
        long long sum = 0;
        for (int i = 0; i < N; i++) {
            sum += input[i];
        }
        output[0] = sum;
    }
    else if (tid == 1) {
        // Task b: Formula-based sum: N*(N+1)/2
        output[1] = (long long)N * (N + 1) / 2;
    }
    // Other threads are idle (different tasks concept)
}

int main() {
    int h_input[N];
    long long h_output[2] = {0, 0};

    // Step 4: Fill the array with first N integers (1 to N)
    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1;
    }

    // Step 3: Allocate memory on device
    int *d_input;
    long long *d_output;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, 2 * sizeof(long long));

    // Step 5: Copy data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, 2 * sizeof(long long));

    // Step 6: Define block and grid sizes
    int blockSize = 32;
    int gridSize = 1;

    // Step 7: Launch kernel
    sumKernel<<<gridSize, blockSize>>>(d_input, d_output);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, 2 * sizeof(long long), cudaMemcpyDeviceToHost);

    printf("=== Problem 1: Sum of First %d Integers ===\n", N);
    printf("a. Iterative Sum  : %lld\n", h_output[0]);
    printf("b. Formula Sum    : %lld\n", h_output[1]);
    printf("Expected (N*(N+1)/2): %lld\n", (long long)N * (N + 1) / 2);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
