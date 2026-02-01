#include<iostream>
#include<omp.h>

using namespace std;

static long num_steps=100000000;

int main(){
    double step=1.0/num_steps;
    double sum=0.0;
    double pi;

    #pragma omp parallel for reduction(+:sum)
    for (long i=0;i<num_steps;i++){
        double x=(i+0.5)*step;
        sum+=4.0/(1.0+x*x);
    }
    pi=step*sum;

    cout<<"Approximate Value of Pi="<<pi<<endl;
    return 0;
}