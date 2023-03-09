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
    export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="2=512MB";
    //putenv("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=1G");
    cudaSetDevice(2);
    CUcontext pctx;
    CUdevice dev;

    int err = cuCtxGetDevice(&dev);
    if(err){
        cout<<"cuCtxGetDevice error:"<<err<<endl;
        return 0;
    }

    getMem();

    //1048576 -> 1M
    size_t storage_size = 1048576*2000;
    float* h_A;
    cudaMallocHost(&h_A, storage_size);

    //nan dao cu driver api bu xing?
    /*
    CUdeviceptr device_ptr;
    int i=cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size);
    */
    float* device_ptr;
    cudaMalloc(&device_ptr,storage_size);



    if(i)
    {
    cout<<"error: "<<i<<endl;
    }
    CUstream firststream;
    cuStreamCreate(&firststream,0);
    i=cuMemcpyHtoDAsync((CUdeviceptr)device_ptr,h_A,storage_size,firststream);
    if(i)
    {
    cout<<"error: "<<i<<endl;
    }
    cudaDeviceSynchronize();

    getMem();

    /*
    getMem();
    cout<<"after first new context:"<<endl;
    err = cuCtxCreate(&pctx, CU_CTX_SCHED_YIELD, dev);
    if(err) {
        cout<<"cuCtxCreate error:"<<err<<endl;
        return 0;
    }
    getMem();

    cout<<"after second new context:"<<endl;
    err = cuCtxCreate(&pctx, CU_CTX_SCHED_YIELD, dev);
    if(err) {
        cout<<"cuCtxCreate error:"<<err<<endl;
        return 0;
    }
    getMem();
    */

    return 0;

}
