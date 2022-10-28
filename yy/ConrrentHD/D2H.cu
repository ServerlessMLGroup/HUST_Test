#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <thread>
#include <random>
#include <ctime>
#include <time.h>

using namespace std;

enum class Unit{
    Byte, KB, MB, GB, TB, PB, EB
};


double convert(double size, Unit unit)
{
    double result = size;
    switch (unit)
    {
    case Unit::EB:
        result /= 1024;     // flow through
    case Unit::PB:
        result /= 1024;     // flow through
    case Unit::TB:
        result /= 1024;     // flow through
    case Unit::GB:
        result /= 1024;     // flow through
    case Unit::MB:
        result /= 1024;     // flow through
    case Unit::KB:
        result /= 1024;     // flow through
    case Unit::Byte:
        result /= 1;
    default:
        break;
    }
    return result;
}

void getMem() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Free mem = %.4f MB, Total = %.4f MB \n", convert(free, Unit::MB), convert(total, Unit::MB));
}

void getMembycu() {
    size_t free, total;
    int err = cuMemGetInfo(&free, &total);
    if (err) {
        cout<<"getMembycu error:"<<err<<endl;
    }
    else {
        printf("Free mem = %.4f MB, Total = %.4f MB \n", convert(free, Unit::MB), convert(total, Unit::MB)); 
    }
}


__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i]; 
}


int main()
{
    cudaSetDevice(3);
    getMem();
    

    clock_t start,finish;
    double Times;
    
    cout<<"initialize variable at GPU"<<endl;
    int N = 10485760;
    size_t size = N * sizeof(float);

    cout<<"Allocate Host Memory"<<endl;
    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    //Initialize input vectors
    uniform_real_distribution<float> u(0,10);
    default_random_engine e(time(NULL));
    for(int i=0;i < N; ++i){
        *(h_A + i) = u(e);
	*(h_B + i) = u(e);
	*(h_C + i) = u(e);
    }

    cout<<"Allocate Device Memory"<<endl;
    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);
    getMem();
  
    cout<<"MemCpy D2H"<<endl;
    start = clock();
    for(int i=0;i<1000;i++){
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    }
    finish = clock();
    
    Times=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"start time: "<<start<<endl;
    cout<<"finish time: "<<finish<<endl;
    cout<<"CLOCKS_PER_SEC: "<<CLOCKS_PER_SEC<<endl;
    cout<<"Time use(s): "<<Times<<endl;

    //Free Device Memory 
    cudaFree(d_A);
    getMem();
    cudaFree(d_B);
    getMem();
    cudaFree(d_C);

    return 0;

}
