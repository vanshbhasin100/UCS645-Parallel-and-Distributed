#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <random>

using namespace std;

const double EPSILON = 1.0;
const double SIGMA = 1.0;

struct Vector3 {
    double x, y, z;
};

void reset_forces(vector<Vector3>& forces) {
    fill(forces.begin(), forces.end(), Vector3{0.0, 0.0, 0.0});
}

void initialize_particles(int N, vector<Vector3>& positions) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0); 

    for (int i = 0; i < N; ++i) {
        positions[i] = {dis(gen), dis(gen), dis(gen)};
    }
}

int main() {
    int N = 2000; 
    
    vector<Vector3> positions(N);
    vector<Vector3> forces(N);
    initialize_particles(N, positions);

    vector<int> thread_counts = {1, 2, 4, 8};
    double t_serial = 0.0;

    cout << "====================================================================\n";
    cout << "  Question 1: Molecular Dynamics Force Calculation (Lennard-Jones)  \n";
    cout << "====================================================================\n";
    cout << "Particles (N): " << N << "\n";
    cout << "Scheduling: Dynamic (Chunk Size 10)\n";
    cout << "Race Condition Handling: OpenMP Atomics\n\n";
    
    cout << left << setw(10) << "Threads" 
         << setw(15) << "Energy" 
         << setw(15) << "Time (s)" 
         << setw(12) << "Speedup" 
         << setw(12) << "Efficiency" << endl;
    cout << string(64, '-') << endl;

    for (int num_threads : thread_counts) {
        if (num_threads > omp_get_max_threads()) break;

        omp_set_num_threads(num_threads);
        reset_forces(forces);
        double total_energy = 0.0;

        double start_time = omp_get_wtime();

        #pragma omp parallel for schedule(dynamic, 10) reduction(+:total_energy)
        for (int i = 0; i < N; ++i) {
            double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0; 

            for (int j = i + 1; j < N; ++j) {
                double dx = positions[i].x - positions[j].x;
                double dy = positions[i].y - positions[j].y;
                double dz = positions[i].z - positions[j].z;
                double r2 = dx*dx + dy*dy + dz*dz;

                if (r2 < 1e-4) continue; 
                
                double r2inv = 1.0 / r2;
                double r6inv = r2inv * r2inv * r2inv;
                double r12inv = r6inv * r6inv;

                double force_scalar = (24.0 * EPSILON / r2) * (2.0 * r12inv - r6inv);
                
                total_energy += 4.0 * EPSILON * (r12inv - r6inv);

                double f_x = force_scalar * dx;
                double f_y = force_scalar * dy;
                double f_z = force_scalar * dz;

                fx_i += f_x;
                fy_i += f_y;
                fz_i += f_z;
                forces[j].x -= f_x;

                #pragma omp atomic
                forces[j].y -= f_y;
                #pragma omp atomic
                forces[j].z -= f_z;
            }

            #pragma omp atomic
            forces[i].x += fx_i;
            #pragma omp atomic
            forces[i].y += fy_i;
            #pragma omp atomic
            forces[i].z += fz_i;
        }

        double end_time = omp_get_wtime();
        double exec_time = end_time - start_time;

        if (num_threads == 1) t_serial = exec_time;
        double speedup = t_serial / exec_time;          
        double efficiency = speedup / num_threads;      

        cout << left << setw(10) << num_threads 
             << setw(15) << setprecision(5) << scientific << total_energy 
             << setw(15) << fixed << setprecision(5) << exec_time 
             << setw(12) << setprecision(2) << speedup 
             << setw(12) << setprecision(2) << efficiency << endl;
    }
    cout << string(64, '-') << endl;

    return 0;
}
