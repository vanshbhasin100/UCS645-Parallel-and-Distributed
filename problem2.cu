/*
 * Assignment 7 - Problem 2
 * Merge Sort Implementation:
 * a. Sequential merge sort with pipelining concept
 * b. Parallel merge sort using CUDA
 * c. Performance comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>

#define ARRAY_SIZE 1000

// =============================================
// Part (a): Sequential Merge Sort (Pipelining)
// =============================================

void merge(int *arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else               arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L); free(R);
}

void mergeSort(int *arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// =============================================
// Part (b): CUDA Parallel Merge Sort
// =============================================

// Each thread sorts a sub-segment using insertion sort (bottom-up approach)
__global__ void insertionSortKernel(int *arr, int n, int segSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid * segSize;
    int end = min(start + segSize, n);

    // Insertion sort within segment
    for (int i = start + 1; i < end; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= start && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Merge two sorted halves on GPU
__global__ void mergeKernel(int *arr, int *temp, int n, int width) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int left = tid * 2 * width;

    if (left >= n) return;

    int mid   = min(left + width, n);
    int right = min(left + 2 * width, n);

    // Merge [left, mid) and [mid, right) into temp
    int i = left, j = mid, k = left;
    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else                   temp[k++] = arr[j++];
    }
    while (i < mid)   temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];
}

void cudaMergeSort(int *h_arr, int n) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr,  n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int segSize = 32;  // Initial segment size for insertion sort
    int threads = 256;
    int blocks  = (n + threads * segSize - 1) / (threads * segSize);

    // Step 1: Sort segments in parallel using insertion sort
    insertionSortKernel<<<blocks, threads>>>(d_arr, n, segSize);
    cudaDeviceSynchronize();

    // Step 2: Bottom-up merge on GPU
    for (int width = segSize; width < n; width *= 2) {
        int mergeBlocks = (n / (2 * width)) + 1;
        mergeKernel<<<mergeBlocks, 1>>>(d_arr, d_temp, n, width);
        cudaDeviceSynchronize();
        // Copy merged result back
        cudaMemcpy(d_arr, d_temp, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
}

// Verify array is sorted
int isSorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++)
        if (arr[i] > arr[i + 1]) return 0;
    return 1;
}

int main() {
    int arr_seq[ARRAY_SIZE], arr_cuda[ARRAY_SIZE];

    srand(42);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr_seq[i]  = rand() % 10000;
        arr_cuda[i] = arr_seq[i];
    }

    printf("=== Problem 2: Merge Sort ===\n");
    printf("Array size: %d\n\n", ARRAY_SIZE);

    // --- Part (a): Sequential merge sort ---
    clock_t start_seq = clock();
    mergeSort(arr_seq, 0, ARRAY_SIZE - 1);
    clock_t end_seq = clock();
    double time_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC * 1000.0;

    printf("a. Sequential Merge Sort (Pipelined):\n");
    printf("   Time: %.4f ms\n", time_seq);
    printf("   Sorted correctly: %s\n\n", isSorted(arr_seq, ARRAY_SIZE) ? "YES" : "NO");

    // --- Part (b): CUDA parallel merge sort ---
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    cudaMergeSort(arr_cuda, ARRAY_SIZE);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float time_cuda = 0;
    cudaEventElapsedTime(&time_cuda, ev_start, ev_stop);

    printf("b. CUDA Parallel Merge Sort:\n");
    printf("   Time: %.4f ms\n", time_cuda);
    printf("   Sorted correctly: %s\n\n", isSorted(arr_cuda, ARRAY_SIZE) ? "YES" : "NO");

    // --- Part (c): Comparison ---
    printf("c. Performance Comparison:\n");
    printf("   Sequential time : %.4f ms\n", time_seq);
    printf("   CUDA time       : %.4f ms\n", time_cuda);
    if (time_cuda < time_seq)
        printf("   Speedup: %.2fx (CUDA is faster)\n", time_seq / time_cuda);
    else
        printf("   Note: For n=1000, CPU overhead may dominate due to small data size.\n");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return 0;
}
