// Assignment 6 - Part C: Matrix Addition using CUDA
// Adds two large integer matrices element-wise on the GPU

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ROWS       1024
#define COLS       1024
#define BLOCK_DIM  16       // 16x16 = 256 threads per block

// ----------------------------------------------------------------
// CUDA Kernel: Matrix Addition
//
// Analysis for an N x M matrix:
//   Floating-point operations  : N * M  (one addition per element)
//   Global memory reads        : 2 * N * M  (read A[i][j] and B[i][j])
//   Global memory writes       : N * M  (write C[i][j])
// ----------------------------------------------------------------
__global__ void matAddKernel(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];   // 1 addition (1 FLOP), 2 reads, 1 write
    }
}

// ----------------------------------------------------------------
// Host: initialise matrix with random ints in [0, 100)
// ----------------------------------------------------------------
void initMatrix(int *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = rand() % 100;
}

// ----------------------------------------------------------------
// Host: CPU reference addition for verification
// ----------------------------------------------------------------
void cpuMatAdd(int *A, int *B, int *C, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        C[i] = A[i] + B[i];
}

// ----------------------------------------------------------------
// Host: compare two matrices
// ----------------------------------------------------------------
int compareMatrices(int *ref, int *gpu, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        if (ref[i] != gpu[i]) return 0;
    return 1;
}

int main() {
    int rows = ROWS, cols = COLS;
    size_t bytes = rows * cols * sizeof(int);

    printf("Matrix size: %d x %d (%zu MB each)\n", rows, cols,
           bytes / (1024 * 1024));

    // ── Allocate host memory ─────────────────────────────────────
    int *h_A   = (int *)malloc(bytes);
    int *h_B   = (int *)malloc(bytes);
    int *h_C   = (int *)malloc(bytes);      // GPU result
    int *h_ref = (int *)malloc(bytes);      // CPU reference

    srand(123);
    initMatrix(h_A, rows, cols);
    initMatrix(h_B, rows, cols);

    // ── Allocate device memory ───────────────────────────────────
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // ── Copy host → device ───────────────────────────────────────
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // ── Configure grid and block dimensions ─────────────────────
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);   // 16x16 threads per block
    dim3 gridDim((cols + BLOCK_DIM - 1) / BLOCK_DIM,
                 (rows + BLOCK_DIM - 1) / BLOCK_DIM);

    printf("Block dim : (%d x %d)\n", BLOCK_DIM, BLOCK_DIM);
    printf("Grid  dim : (%d x %d)\n", gridDim.x, gridDim.y);
    printf("Total threads: %d\n\n", gridDim.x * gridDim.y * BLOCK_DIM * BLOCK_DIM);

    // ── Launch kernel ────────────────────────────────────────────
    matAddKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    cudaDeviceSynchronize();

    // ── Copy device → host ───────────────────────────────────────
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // ── Verify ───────────────────────────────────────────────────
    cpuMatAdd(h_A, h_B, h_ref, rows, cols);
    int correct = compareMatrices(h_ref, h_C, rows, cols);
    printf("Verification: %s\n\n", correct ? "PASS" : "FAIL");

    // ── Operation count analysis ─────────────────────────────────
    long long total = (long long)rows * cols;
    printf("=== Kernel Operation Analysis ===\n");
    printf("  Total elements           : %lld\n", total);
    printf("  Floating-point ops (FLOP): %lld  (1 add per element)\n", total);
    printf("  Global memory reads      : %lld  (2 reads per element: A + B)\n", 2 * total);
    printf("  Global memory writes     : %lld  (1 write per element: C)\n", total);
    printf("  Arithmetic intensity     : %.4f FLOP/byte\n",
           (double)total / (3.0 * total * sizeof(int)));

    // ── Free device memory ───────────────────────────────────────
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
