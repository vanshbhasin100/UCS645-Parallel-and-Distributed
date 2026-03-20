#pragma once

// Part 1: Sequential baseline
void correlate_sequential(int ny, int nx, const float* data, float* result);

// Part 2: OpenMP parallel
void correlate_openmp(int ny, int nx, const float* data, float* result);

// Part 3: Optimized (SIMD + cache-friendly + OpenMP)
void correlate_optimized(int ny, int nx, const float* data, float* result);
