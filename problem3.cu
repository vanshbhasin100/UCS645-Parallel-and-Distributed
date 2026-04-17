/*
 * Assignment 7 - Problem 3
 * CUDA Vector Addition Kernel with full profiling
 * 1.1 Static global variables (no cudaMalloc for arrays)
 * 1.2 Kernel execution timing
 * 1.3 Theoretical memory bandwidth using cudaDeviceProp
 * 1.4 Measured memory bandwidth
 *
 * Compile: nvcc -O2 -o problem3 problem3.cu
 * Profile: nvprof ./problem3
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1 << 20   // ~1 million elements (compile-time constant)

// =============================================================
// 1.1: Statically defined global device variables
//      These are device symbols - NOT the same as device pointers.
//      Do NOT pass these directly as kernel arguments!
// =============================================================
__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

// =============================================================
// Vector Addition Kernel
// Reads from d_A and d_B, writes to d_C
// =============================================================
__global__ void vectorAdd() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // Access global device symbols directly inside the kernel
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

int main() {
    // Host arrays
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    // =============================================================
    // 1.3: Query device properties for theoretical bandwidth
    // =============================================================
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // First CUDA device

    // memoryClockRate is in kHz, memoryBusWidth is in bits
    // DDR memory is double-pumped (x2)
    // Formula: BW = 2 * memoryClockRate(kHz) * memoryBusWidth(bits)
    //
    // Convert: kHz -> Hz (* 1000), bits -> bytes (/ 8), bytes -> GB (/ 1e9)
    // BW (GB/s) = 2 * memoryClockRate * 1000 * memoryBusWidth / 8 / 1e9
    //           = 2 * memoryClockRate * memoryBusWidth / 8e6

    double theoreticalBW = 2.0 * prop.memoryClockRate   // kHz
                               * prop.memoryBusWidth     // bits
                               / 8.0                     // bits -> bytes
                               / 1.0e6;                  // kbytes -> GB/s

    printf("=== Problem 3: CUDA Vector Addition with Profiling ===\n\n");
    printf("Device: %s\n", prop.name);
    printf("Memory Clock Rate : %d kHz\n",  prop.memoryClockRate);
    printf("Memory Bus Width  : %d bits\n", prop.memoryBusWidth);
    printf("Theoretical Bandwidth: %.2f GB/s\n\n", theoreticalBW);

    // =============================================================
    // Copy host data to device static symbols using cudaMemcpyToSymbol
    // =============================================================
    cudaMemcpyToSymbol(d_A, h_A, N * sizeof(float));
    cudaMemcpyToSymbol(d_B, h_B, N * sizeof(float));

    // Grid/block configuration
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    // =============================================================
    // 1.2: Timing the kernel using CUDA events
    // =============================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<gridSize, blockSize>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);

    printf("Kernel execution time: %.4f ms\n", elapsedMs);

    // =============================================================
    // 1.4: Measured memory bandwidth
    // RBytes: each thread reads 2 floats (d_A[idx] and d_B[idx])
    // WBytes: each thread writes 1 float (d_C[idx])
    // Total threads launched = gridSize * blockSize
    // =============================================================
    long long totalThreads = (long long)gridSize * blockSize;
    long long RBytes = totalThreads * 2 * sizeof(float);  // reads
    long long WBytes = totalThreads * 1 * sizeof(float);  // writes

    double timeSeconds = elapsedMs / 1000.0;
    double measuredBW  = (double)(RBytes + WBytes) / timeSeconds / 1.0e9;  // GB/s

    printf("Threads launched   : %lld\n",    totalThreads);
    printf("Bytes Read         : %lld bytes\n", RBytes);
    printf("Bytes Written      : %lld bytes\n", WBytes);
    printf("Measured Bandwidth : %.2f GB/s\n\n", measuredBW);

    printf("=== Bandwidth Comparison ===\n");
    printf("Theoretical BW : %.2f GB/s\n", theoreticalBW);
    printf("Measured BW    : %.2f GB/s\n", measuredBW);
    printf("Efficiency     : %.2f%%\n", (measuredBW / theoreticalBW) * 100.0);

    // Copy results back and validate
    cudaMemcpyFromSymbol(h_C, d_C, N * sizeof(float));

    // Spot check
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (abs(h_C[i] - (h_A[i] + h_B[i])) > 1e-4f) {
            correct = false; break;
        }
    }
    printf("\nResult Validation  : %s\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
