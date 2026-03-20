#include "correlate.h"
#include <cmath>
#include <vector>
#include <omp.h>
#include <immintrin.h>

// ============================================================
// PART 1: Sequential Baseline
// ============================================================
void correlate_sequential(int ny, int nx, const float* data, float* result) {
    std::vector<double> norm(ny * nx);

    for (int y = 0; y < ny; y++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++) mean += data[x + y * nx];
        mean /= nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double diff = data[x + y * nx] - mean;
            sq_sum += diff * diff;
        }
        double std_dev = std::sqrt(sq_sum);

        for (int x = 0; x < nx; x++) {
            norm[x + y * nx] = (std_dev > 0)
                ? (data[x + y * nx] - mean) / std_dev
                : 0.0;
        }
    }

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int x = 0; x < nx; x++)
                dot += norm[x + i * nx] * norm[x + j * nx];
            result[i + j * ny] = (float)(dot / nx);
        }
    }
}

// ============================================================
// PART 2: OpenMP Parallel Version
// ============================================================
void correlate_openmp(int ny, int nx, const float* data, float* result) {
    std::vector<double> norm(ny * nx);

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < ny; y++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++) mean += data[x + y * nx];
        mean /= nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double diff = data[x + y * nx] - mean;
            sq_sum += diff * diff;
        }
        double std_dev = std::sqrt(sq_sum);

        for (int x = 0; x < nx; x++) {
            norm[x + y * nx] = (std_dev > 0)
                ? (data[x + y * nx] - mean) / std_dev
                : 0.0;
        }
    }

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < ny; j++) {
            if (j > i) continue;
            double dot = 0.0;
            for (int x = 0; x < nx; x++)
                dot += norm[x + i * nx] * norm[x + j * nx];
            result[i + j * ny] = (float)(dot / nx);
        }
    }
}

// ============================================================
// PART 3: Optimized Version (SIMD + Cache-Friendly + OpenMP)
// ============================================================
void correlate_optimized(int ny, int nx, const float* data, float* result) {
    int nx_padded = (nx + 3) & ~3;
    std::vector<double> norm(ny * nx_padded, 0.0);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; y++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++) mean += data[x + y * nx];
        mean /= nx;

        double sq_sum = 0.0;
        for (int x = 0; x < nx; x++) {
            double diff = data[x + y * nx] - mean;
            sq_sum += diff * diff;
        }
        double std_dev = std::sqrt(sq_sum);

        for (int x = 0; x < nx; x++) {
            norm[x + y * nx_padded] = (std_dev > 0)
                ? (data[x + y * nx] - mean) / std_dev
                : 0.0;
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            const double* row_i = &norm[i * nx_padded];
            const double* row_j = &norm[j * nx_padded];

            double acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            int x = 0;
            for (; x + 3 < nx; x += 4) {
                acc0 += row_i[x]   * row_j[x];
                acc1 += row_i[x+1] * row_j[x+1];
                acc2 += row_i[x+2] * row_j[x+2];
                acc3 += row_i[x+3] * row_j[x+3];
            }
            double dot = acc0 + acc1 + acc2 + acc3;
            for (; x < nx; x++) dot += row_i[x] * row_j[x];

            result[i + j * ny] = (float)(dot / nx);
        }
    }
}
