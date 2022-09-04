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
    cudaSetDevice(0);
    CUcontext pctx;
    CUdevice dev;
    // getMem();
    /*
    cout<<"getDeviceCount"<<endl;
    int count;
    int err = cudaGetDeviceCount(&count);
    if(err){
        cout<<"getDeviceCount error:"<<err<<endl;
        return 0;
    }
    cout<<"getDeviceCount Fininshed"<<endl;

    // cout<<"after new context1:"<<endl;
    */
    cout<<"cuCtxGetDevice"<<endl;
    int err = cuCtxGetDevice(&dev);
    if(err){
        cout<<"cuCtxGetDevice error:"<<err<<endl;
        return 0;
    }
    cout<<"basic memory"<<endl;


    cout<<"new context:"<<endl;
    err = cuCtxCreate(&pctx, CU_CTX_SCHED_YIELD, dev);
    if(err) {
        cout<<"cuCtxCreate error:"<<err<<endl;
        return 0;
    }
    getMem();


    cout<<"new context:"<<endl;
    err = cuCtxCreate(&pctx, CU_CTX_SCHED_YIELD, dev);
    if(err) {
        cout<<"cuCtxCreate error:"<<err<<endl;
        return 0;
    }
    getMem();


    cout<<"new context:"<<endl;
    err = cuCtxCreate(&pctx, CU_CTX_SCHED_YIELD, dev);
    if(err) {
        cout<<"cuCtxCreate error:"<<err<<endl;
        return 0;
    }
    getMem();

    return 0;
}
