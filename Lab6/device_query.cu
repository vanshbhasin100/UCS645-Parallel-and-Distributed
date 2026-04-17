// Assignment 6 - Part A: Device Query
// Queries GPU properties using CUDA Runtime API

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("===== Device %d: %s =====\n", dev, prop.name);
        printf("\n-- Architecture & Compute Capability --\n");
        printf("  Compute Capability:             %d.%d\n", prop.major, prop.minor);

        printf("\n-- Thread / Block / Grid Dimensions --\n");
        printf("  Max Threads per Block:          %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dimensions:           [%d x %d x %d]\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions:            [%d x %d x %d]\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Warp Size:                      %d\n", prop.warpSize);
        printf("  Max Threads per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Number of SMs:                  %d\n", prop.multiProcessorCount);

        printf("\n-- Memory --\n");
        printf("  Total Global Memory:            %.2f MB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0));
        printf("  Total Constant Memory:          %zu bytes (%zu KB)\n",
               prop.totalConstMem, prop.totalConstMem / 1024);
        printf("  Shared Memory per Block:        %zu bytes (%zu KB)\n",
               prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024);
        printf("  Shared Memory per SM:           %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  L2 Cache Size:                  %d bytes\n", prop.l2CacheSize);
        printf("  Memory Bus Width:               %d bits\n", prop.memoryBusWidth);
        printf("  Memory Clock Rate:              %d kHz\n", prop.memoryClockRate);

        printf("\n-- Double Precision Support --\n");
        // Devices with compute capability >= 1.3 support double precision
        if (prop.major > 1 || (prop.major == 1 && prop.minor >= 3)) {
            printf("  Double Precision:               SUPPORTED\n");
        } else {
            printf("  Double Precision:               NOT SUPPORTED\n");
        }

        printf("\n-- Other Properties --\n");
        printf("  Clock Rate:                     %.2f MHz\n", prop.clockRate / 1000.0);
        printf("  Registers per Block:            %d\n", prop.regsPerBlock);
        printf("  Concurrent Kernels:             %s\n",
               prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC Enabled:                    %s\n",
               prop.ECCEnabled ? "Yes" : "No");
        printf("  Unified Addressing:             %s\n",
               prop.unifiedAddressing ? "Yes" : "No");
        printf("\n");
    }

    // Answer Q3: Max threads for 1D grid with maxGrid=65535, maxBlock=512
    long long maxGrid1D = 65535;
    long long maxBlock1D = 512;
    printf("=== Q3 Calculation ===\n");
    printf("Max 1D Grid Dim = %lld, Max 1D Block Dim = %lld\n", maxGrid1D, maxBlock1D);
    printf("Max Threads = %lld x %lld = %lld\n",
           maxGrid1D, maxBlock1D, maxGrid1D * maxBlock1D);

    return 0;
}
