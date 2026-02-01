#include <iostream>

#include <omp.h>

using namespace std;

#define N 1000

int main() {
    static int A[N][N], B[N][N], C[N][N];
    double start, end;


    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            A[i][j] = 1;
            B[i][j] = 1;
            }
            }

start = omp_get_wtime();


#pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            C[i][j] = 0;
                for(int k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

end = omp_get_wtime();
cout << "2D Threading Time: "
<< end - start << " seconds" << endl;
return 0;
}


                        
