#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <thread>
#include <random>
#include <ctime>
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
    CUcontext pctx;
    CUdevice dev;
    getMem();
    cout<<"after new context:"<<endl;
    int err = cuCtxGetDevice(&dev);
    if(err){
        cout<<"cuCtxGetDevice error:"<<err<<endl;
        return 0;
    }
    err = cuCtxCreate(&pctx, CU_CTX_SCHED_YIELD, dev);
    if(err) {
        cout<<"cuCtxCreate error:"<<err<<endl;
        return 0;
    }

    getMem();
    cout<<"initialize variable at GPU"<<endl;
    int N = 10485760;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    uniform_real_distribution<float> u(0, 10);
    default_random_engine e(time(NULL));

    for(int i = 0; i < N; ++i) {
        *(h_A + i) = u(e);
        *(h_B + i) = u(e);
        *(h_C + i) = u(e);
    }
    cout<<"initialize for three times * size"<<endl;
    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    getMem();
    float* d_B;
    cudaMalloc(&d_B, size);
    getMem();
    float* d_C;
    cudaMalloc(&d_C, size);
    getMem();
  
    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cout<<"after VecAdd"<<endl;
    getMem();


    // create a new context and use variable d_A,test whther it's ok
    cout<<"after new context initialization:"<<endl;
    CUcontext pctxnew;
    err = cuCtxCreate(&pctxnew, CU_CTX_SCHED_YIELD, dev);
    getMem();
    int err2=0;
    if(err){
        cout<<"cuCtx new create error: "<<err<<endl;
    }
    err2 = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    if(err2){
        cout<<"Failed to use variable under another context though belong to th same thread:"<<err2<<endl;
    }
    cout<<"new context transfer data from d_B to h_B"<<endl;
    getMem();

    cout<<"new context vecadd"<<endl;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    getMem();

    cuCtxPopCurrent(&pctxnew);
    cout<<"throw new context"<<endl;
    getMem();

    //destroy the new context
    

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cout<<"Memcpy fron d_c to h_C"<<endl;
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    getMem();
    cudaFree(d_A);
    getMem();
    cudaFree(d_B);
    getMem();
    cudaFree(d_C);

    getMem();
    return 0;

}
