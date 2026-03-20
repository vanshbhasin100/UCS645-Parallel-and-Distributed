#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "correlate.h"

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void fill_random(std::vector<float>& data, int ny, int nx) {
    for (int i = 0; i < ny * nx; i++)
        data[i] = (float)rand() / RAND_MAX;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <ny> <nx> [num_threads]\n";
        std::cerr << "  ny          = number of rows (vectors)\n";
        std::cerr << "  nx          = length of each vector\n";
        std::cerr << "  num_threads = optional (default = max available)\n";
        return 1;
    }

    int ny          = std::atoi(argv[1]);
    int nx          = std::atoi(argv[2]);
    int num_threads = (argc >= 4) ? std::atoi(argv[3]) : omp_get_max_threads();

    omp_set_num_threads(num_threads);

    std::cout << "=== Lab3: Correlation Assignment ===\n";
    std::cout << "Matrix size : " << ny << " x " << nx << "\n";
    std::cout << "Threads     : " << num_threads << "\n\n";

    srand(42);
    std::vector<float> data(ny * nx);
    fill_random(data, ny, nx);

    std::vector<float> result_seq(ny * ny, 0.0f);
    std::vector<float> result_omp(ny * ny, 0.0f);
    std::vector<float> result_opt(ny * ny, 0.0f);

    // Part 1 — Sequential
    {
        auto t = std::chrono::steady_clock::now();
        correlate_sequential(ny, nx, data.data(), result_seq.data());
        std::cout << "[Part 1 - Sequential]  " << elapsed_ms(t) << " ms\n";
    }

    // Part 2 — OpenMP
    {
        auto t = std::chrono::steady_clock::now();
        correlate_openmp(ny, nx, data.data(), result_omp.data());
        std::cout << "[Part 2 - OpenMP]      " << elapsed_ms(t) << " ms\n";
    }

    // Part 3 — Optimized
    {
        auto t = std::chrono::steady_clock::now();
        correlate_optimized(ny, nx, data.data(), result_opt.data());
        std::cout << "[Part 3 - Optimized]   " << elapsed_ms(t) << " ms\n";
    }

    // Correctness check
    float max_diff = 0.0f;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j <= i; j++) {
            float diff = std::abs(result_seq[i + j * ny] - result_omp[i + j * ny]);
            if (diff > max_diff) max_diff = diff;
        }

    std::cout << "\nMax diff (seq vs omp) : " << max_diff << "\n";
    std::cout << "result[0][0]          : " << result_seq[0] << " (expected ~1.0)\n";

    return 0;
}
