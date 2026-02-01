#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

#define N 65536 //fixed the size of the vectors

int main() {
omp_set_num_threads(16);
vector<double> X(N), Y(N); //double precision vectors
double a = 2.5;

for(int i = 0; i < N; i++) {
X[i] = i * 1.0;
Y[i] = i * 2.0;
}
double start = omp_get_wtime();
#pragma omp parallel for
for(int i = 0; i < N; i++) {
X[i] = a * X[i] + Y[i];
}

double end = omp_get_wtime();

cout << "Time taken: "
<< end - start << " seconds" << endl;

return 0;
}