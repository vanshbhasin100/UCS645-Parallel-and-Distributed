#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <algorithm>

using namespace std;

const int N = 2000;          
const int T_STEPS = 100;     
const double DIFFUSION_C = 0.1; 

void initialize_grid(vector<vector<double>>& grid) {
    for (int i = 0; i < N; ++i) {
        fill(grid[i].begin(), grid[i].end(), 0.0);
    }
    int center = N / 2;
    int radius = N / 10;
    for (int i = center - radius; i < center + radius; ++i) {
        for (int j = center - radius; j < center + radius; ++j) {
            grid[i][j] = 100.0;
        }
    }
}

int main() {
    vector<vector<double>> T_curr(N, vector<double>(N));
    vector<vector<double>> T_next(N, vector<double>(N));

    initialize_grid(T_curr);

    cout << "=================================================================\n";
    cout << "  Question 3: Heat Diffusion Simulation (Finite Difference) \n";
    cout << "=================================================================\n";
    cout << "Grid Size: " << N << " x " << N << "\n";
    cout << "Time Steps: " << T_STEPS << "\n";
    cout << "Threads: " << omp_get_max_threads() << "\n\n";

    double start_time = omp_get_wtime();

    for (int t = 0; t < T_STEPS; ++t) {
        
        double total_heat = 0.0;

        #pragma omp parallel for collapse(2) reduction(+:total_heat)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                double neighbors = T_curr[i+1][j] + T_curr[i-1][j] + 
                                   T_curr[i][j+1] + T_curr[i][j-1];
                
                T_next[i][j] = T_curr[i][j] + DIFFUSION_C * (neighbors - 4.0 * T_curr[i][j]);
                
                total_heat += T_next[i][j];
            }
        }

        T_curr.swap(T_next);

        if (t % 20 == 0) {
        }
    }

    double end_time = omp_get_wtime();
    double exec_time = end_time - start_time;
    
    double total_flops = (double)N * N * T_STEPS * 5.0;
    double gflops = (total_flops / exec_time) / 1e9;

    cout << left << setw(10) << "Threads" 
         << setw(15) << "Time (s)" 
         << setw(15) << "GFLOPS" << endl;
    cout << string(40, '-') << endl;
    
    cout << left << setw(10) << omp_get_max_threads()
         << setw(15) << fixed << setprecision(4) << exec_time 
         << setw(15) << setprecision(2) << gflops << endl;

    return 0;
}
