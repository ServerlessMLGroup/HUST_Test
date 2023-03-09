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
    int j=cudaMemGetInfo(&free, &total);
    if(j)
    {
       cout<<"get mem error: "<<j<<endl;
    }
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


__global__ void VecAdd(float* h_A, float N)
{
    int tx =threadIdx.x;
    int bx =blockIdx.x;
    int offset=100*bx+tx;
    for(int i=0;i<50000;i++)
    {
    h_A[offset*50000+i]=h_A[offset*50000+i]+N;
    }
}


int main()
{

    //putenv("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=1G");
    cudaSetDevice(2);
    CUcontext pctx;
    CUdevice dev;

    int i=0;
    int err = cuCtxGetDevice(&dev);
    if(err){
        cout<<"cuCtxGetDevice error:"<<err<<endl;
        return 0;
    }

    getMem();

    //1048576 -> 1M
    //size_t storage_size = 1048576*2000;
    size_t storage_size = 2000000000;
    float* h_A;
    i=cudaMallocManaged(&h_A,storage_size);
    if(i)
    {
    cout<<"cuda malloc managed error: "<<i<<endl;
    }
    for(int k=0;k<500000000;k++)
    {
        h_A[k]=1.0;
    }


    /*
    cudaMallocHost(&h_A, storage_size);

    //nan dao cu driver api bu xing?

    CUdeviceptr device_ptr;
    i=cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size);


    float* device_ptr;
    i=cudaMalloc(&device_ptr,storage_size);

    if(i)
    {
    cout<<"cuda malloc error: "<<i<<endl;
    }
    CUstream firststream;
    cuStreamCreate(&firststream,0);
    i=cuMemcpyHtoDAsync((CUdeviceptr)device_ptr,h_A,storage_size,firststream);
    if(i)
    {
    cout<<"memcpy error: "<<i<<endl;
    }
    cudaDeviceSynchronize();
    */

    /*
    i=cudaMemPrefetchAsync(h_A,storage_size,2);
    if(i)
    {
    cout<<"prefetch error: "<<i<<endl;
    }
    */
    VecAdd<<<100,100>>>(h_A,1.0);
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
