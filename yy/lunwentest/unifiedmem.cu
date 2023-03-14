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
    for(int i=0;i<1000;i++)
    {
    h_A[offset*1000+i]=h_A[offset*1000+i]+N;
    }
}


int main()
{

    //putenv("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=1G");
    cudaSetDevice(3);
    CUcontext pctx;
    CUdevice dev;

    int i=0;
    int err = cuCtxGetDevice(&dev);
    if(err){
        cout<<"cuCtxGetDevice error:"<<err<<endl;
        return 0;
    }
    cuCtxGetCurrent(&pctx);

    getMem();

    //1048576 -> 1M
    //size_t storage_size = 1048576*400;
    size_t storage_size = 1000*1048576;

    storage_size *=25;

    cout<<"storage_size: "<<storage_size<<endl;
    cout<<"size of size_t: "<<sizeof(size_t)<<endl;
    cout<<"size of float: "<<sizeof(float)<<endl;
    float* h_A;
    float* h_B;
    float* h_C;

    //cuda malloc
    float* device_ptr;
    i=cudaMalloc(&device_ptr,storage_size);
    if(i)
    {
    cout<<"cuda malloc error: "<<i<<endl;
    }

    //cuda malloc managed
    i=cudaMallocManaged(&h_A,storage_size);
    if(i)
    {
    cout<<"cuda malloc h_A managed error: "<<i<<endl;
    }



    //use mamaged mem
    /*
    for(int k=0;k<1000000;k++)
    {
        h_A[k]=1.0;
    }
    */

    //prefetch h_A
    i=cudaMemPrefetchAsync(h_A,storage_size,3);
    if(i)
    {
    cout<<"prefetch error: "<<i<<endl;
    }

    //cuda malloc after prefetch
    float* device_ptr1;
    storage_size = 1048576*1000;
    i=cudaMalloc(&device_ptr1,storage_size);
    if(i)
    {
    cout<<"cuda malloc error: "<<i<<endl;
    }

    //VecAdd<<<100,100>>>(h_A,1.0);
    cudaDeviceSynchronize();
    for(int k=999990;k<1000000;k++)
    {
       //cout<<"after add "<<h_A[k]<<endl;
    }

    cudaError_t errd = cudaGetLastError();  // add
    if (errd) cout << "CUDA error: " << cudaGetErrorString(errd) << endl; // add

    getMem();
    return 0;

}
