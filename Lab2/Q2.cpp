#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <string>
#include <random>
#include <iomanip>

using namespace std;

const int MATCH_SCORE = 3;
const int MISMATCH_SCORE = -3;
const int GAP_PENALTY = -2;

string generate_sequence(int length) {
    const char bases[] = "ACGT";
    string seq(length, ' ');
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 3);
    for (int i = 0; i < length; ++i) {
        seq[i] = bases[dis(gen)];
    }
    return seq;
}

double run_smith_waterman(const string& seqA, const string& seqB, const string& sched_type) {
    int n = seqA.length();
    int m = seqB.length();
    
    vector<int> H((n + 1) * (m + 1), 0);

    double start_time = omp_get_wtime();

    int num_diagonals = n + m;

    for (int k = 2; k <= num_diagonals; ++k) {
        
        int i_start = max(1, k - m);
        int i_end = min(n, k - 1);

        if (sched_type == "static") {
            #pragma omp parallel for schedule(static)
            for (int i = i_start; i <= i_end; ++i) {
                int j = k - i;
                int index = i * (m + 1) + j;
                
                int score_diag = H[(i - 1) * (m + 1) + (j - 1)] + 
                                 (seqA[i - 1] == seqB[j - 1] ? MATCH_SCORE : MISMATCH_SCORE);
                int score_up = H[(i - 1) * (m + 1) + j] + GAP_PENALTY;
                int score_left = H[i * (m + 1) + (j - 1)] + GAP_PENALTY;
                
                H[index] = max({0, score_diag, score_up, score_left});
            }
        } else if (sched_type == "dynamic") {
            #pragma omp parallel for schedule(dynamic, 64)
            for (int i = i_start; i <= i_end; ++i) {
                int j = k - i;
                int index = i * (m + 1) + j;
                int score_diag = H[(i - 1) * (m + 1) + (j - 1)] + 
                                 (seqA[i - 1] == seqB[j - 1] ? MATCH_SCORE : MISMATCH_SCORE);
                int score_up = H[(i - 1) * (m + 1) + j] + GAP_PENALTY;
                int score_left = H[i * (m + 1) + (j - 1)] + GAP_PENALTY;
                H[index] = max({0, score_diag, score_up, score_left});
            }
        } else if (sched_type == "guided") {
            #pragma omp parallel for schedule(guided)
            for (int i = i_start; i <= i_end; ++i) {
                int j = k - i;
                int index = i * (m + 1) + j;
                int score_diag = H[(i - 1) * (m + 1) + (j - 1)] + 
                                 (seqA[i - 1] == seqB[j - 1] ? MATCH_SCORE : MISMATCH_SCORE);
                int score_up = H[(i - 1) * (m + 1) + j] + GAP_PENALTY;
                int score_left = H[i * (m + 1) + (j - 1)] + GAP_PENALTY;
                H[index] = max({0, score_diag, score_up, score_left});
            }
        }
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

int main() {
    int N = 5000; 
    int M = 5000; 
    
    cout << "=================================================================\n";
    cout << "  Question 2: Smith-Waterman Wavefront Parallelization \n";
    cout << "=================================================================\n";
    cout << "Sequence Lengths: " << N << " x " << M << "\n";
    cout << "Total Matrix Cells: " << (long long)N * M << "\n";
    cout << "Threads: " << omp_get_max_threads() << "\n\n";

    string seqA = generate_sequence(N);
    string seqB = generate_sequence(M);

    cout << left << setw(15) << "Schedule" 
         << setw(15) << "Time (s)" 
         << setw(15) << "MCUPS" << endl; 
    cout << string(45, '-') << endl;

    vector<string> schedules = {"static", "dynamic", "guided"};
    
    for (const auto& sched : schedules) {
        double time = run_smith_waterman(seqA, seqB, sched);
        double mcups = ((double)N * M / time) / 1e6;
        
        cout << left << setw(15) << sched 
             << setw(15) << fixed << setprecision(4) << time 
             << setw(15) << setprecision(2) << mcups << endl;
    }
    cout << string(45, '-') << endl;

    return 0;
}
