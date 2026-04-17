// Assignment 6 - Part B: Sum of Array Elements using CUDA
// Parallel reduction to compute sum of a floating-point array

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 1024      // Must be a power of 2 for simple reduction
#define BLOCK_SIZE 256

// ----------------------------------------------------------------
// CUDA Kernel: Parallel Reduction Sum
// Each block reduces its segment into a single partial sum stored
// in partialSums[blockIdx.x].
// ----------------------------------------------------------------
__global__ void sumReductionKernel(float *input, float *partialSums, int n) {
    // Shared memory for this block
    __shared__ float sharedData[BLOCK_SIZE];

    int tid       = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element into shared memory (0 if out of bounds)
    sharedData[tid] = (globalIdx < n) ? input[globalIdx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory (tree reduction)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the block's partial sum
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

// ----------------------------------------------------------------
// Host helper: CPU reference sum for verification
// ----------------------------------------------------------------
float cpuSum(float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += arr[i];
    return sum;
}

int main() {
    int n = ARRAY_SIZE;
    size_t bytes = n * sizeof(float);

    // ── Step 1: Allocate and initialize host array ──────────────
    float *h_input = (float *)malloc(bytes);
    if (!h_input) { fprintf(stderr, "Host malloc failed\n"); return 1; }

    srand(42);
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;   // values in [0, 10)
    }
    printf("Array size: %d elements\n", n);
    printf("First 5 elements: %.2f %.2f %.2f %.2f %.2f ...\n",
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);

    // ── Step 2: Allocate device memory ──────────────────────────
    float *d_input, *d_partialSums;
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void **)&d_input,       bytes);
    cudaMalloc((void **)&d_partialSums, numBlocks * sizeof(float));

    // ── Step 3: Copy host memory to device ──────────────────────
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // ── Step 4: Initialize thread block and kernel grid dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(numBlocks);
    printf("\nKernel launch: <<<%d, %d>>>\n", numBlocks, BLOCK_SIZE);

    // ── Step 5: Invoke CUDA kernel ───────────────────────────────
    sumReductionKernel<<<gridDim, blockDim>>>(d_input, d_partialSums, n);
    cudaDeviceSynchronize();

    // Collect partial sums on host and add them
    float *h_partialSums = (float *)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_partialSums, d_partialSums,
               numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float gpuSum = 0.0f;
    for (int i = 0; i < numBlocks; i++) gpuSum += h_partialSums[i];

    // ── Step 6: Free device memory ───────────────────────────────
    cudaFree(d_input);
    cudaFree(d_partialSums);

    // ── Step 7: Verify against CPU reference ─────────────────────
    float cpu = cpuSum(h_input, n);
    printf("\nCPU Sum  = %.4f\n", cpu);
    printf("GPU Sum  = %.4f\n", gpuSum);
    printf("Diff     = %.6f\n", fabsf(cpu - gpuSum));
    printf("Result   : %s\n\n", fabsf(cpu - gpuSum) < 1e-2f ? "PASS" : "FAIL");

    free(h_input);
    free(h_partialSums);
    return 0;
}
