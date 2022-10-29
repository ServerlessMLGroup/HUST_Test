#include <iostream>
#include <thread>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "unistd.h"
#include <thread>
#include <mutex>
#include <random>
#include <ctime>
#include <time.h>
using namespace std;

//Mutex
mutex mtx1,mtx2;

void thread1(CUcontext ctx,float* d_a,float* h_a,size_t size)
{
    clock_t start,finish;
    double singletime=0.0;
    double cotime=0.0;
    int err;
    err=cuCtxPushCurrent(ctx);
    if(err){
    cout<<"Push Context ERR! "<<err<<endl;
    }
    for(int i=0;i < 10;i++)
    {
    mtx2.unlock();
    mtx1.lock();
    start=clock();
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    finish=clock();
    cotime += (double)(finish-start)/CLOCKS_PER_SEC;
    }

    cout<<"device 3 time: "<<cotime<<" s"<<endl;
}

void thread2(CUcontext ctx,float* d_b,float* h_b,size_t size)
{
    clock_t start,finish;
    double singletime=0.0;
    int err;
    err=cuCtxPushCurrent(ctx);
    if(err){
    cout<<"Push Context ERR! "<<err<<endl;
    }
    for(int i=0;i < 10;i++)
    {
    mtx2.lock();
    start=clock();
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    finish=clock();
    singletime += (double)(finish-start)/CLOCKS_PER_SEC;
    mtx1.unlock();
    }
    cout<<"device 2 time: "<<singletime<<" s"<<endl;
}

int main()
{
    cuInit(0);
    cudaSetDevice(3);
    //clock for collection

    //data size
    int N = 10485760;
    size_t size = N * sizeof(float);


    //device3
    cout<<"Create two context and their memory"<<endl;
    int err;
    CUcontext cont1,cont2;
    CUdevice dev;
    cout<<"Device 3: d_A"<<endl;
    err = cuCtxGetDevice(&dev);
    if(err)
    {
        cout<<"Can't get device, err" << err<<endl;
        return 0;
    }
    err = cuCtxCreate(&cont1,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        cout<<"Can't create Context, err" << err << endl;
        return 0;
    }
    float* d_A;
    cudaMalloc(&d_A, size);

    //device2
    cudaSetDevice(2);
    err = cuCtxGetDevice(&dev);
    if(err)
    {
        cout<<"Can't get device, err" << err<<endl;
        return 0;
    }
    err = cuCtxCreate(&cont2,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        cout<<"Can't create Context, err" << err<<endl;
        return 0;
    }
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);


    cout<<"Allocate Host Memory"<<endl;
    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    uniform_real_distribution<float> u(0,10);
    default_random_engine e(time(NULL));
    for(int i=0;i < N; ++i){
    *(h_A + i) = u(e);
	*(h_B + i) = u(e);
	*(h_C + i) = u(e);
    }

    //prepare
    mtx2.lock();
    thread first=thread(thread1,cont1,d_A,h_A,size);
    thread second=thread(thread2,cont2,d_C,h_C,size);
    second.join();
    first.join();
    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
