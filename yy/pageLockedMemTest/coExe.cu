#include <iostream>
#include <pthread.h>
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
#include <stdio.h>
#include<cuda_runtime.h>
using namespace std;
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

mutex workend2;
mutex workend1;
//diy thread

__global__ void kernel(float n1, float n2, float n3, int stop) {
	for (int i = 0; i < stop; i++) {
		n1=cosf(n1);
		n3=n2/n3;
	}
}


void thread1(CUcontext ctx,float* d_a,float* h_a,size_t size,int i)
{
    //set CPU
    /*
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    {
            perror("pthread_setaffinity_np");
    }
    */

    cout<<"number: "<<i<<"   one thread starts: "<<endl;
    int err;
    err=cuCtxPushCurrent(ctx);
    if(err){
    cout<<"Push Context ERR! "<<err<<endl;
    }

    cudaStream_t tempstream;

    cudaError_t cudaStatus;
    cudaStatus = cudaStreamCreate(&tempstream);
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));

    for(int i=1;i < 10;i++)
    {
    //cudaMemcpyAsync(d_a, h_a,size, cudaMemcpyHostToDevice, tempstream);
    kernel<<<1,1,0,tempstream>>>(1.0,2.0,3.0,10000000);
    }

    cuStreamSynchronize(tempstream);
    workend1.unlock();
}

void thread2(CUcontext ctx,int i)
{
    //set CPU
    /*
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    {
            perror("pthread_setaffinity_np");
    }
    */

    clock_t start,finish;
    double time=0.0;

    cout<<"number: "<<i<<"   one thread starts: "<<endl;
    int err;
    err=cuCtxPushCurrent(ctx);
    if(err){
    cout<<"Push Context ERR! "<<err<<endl;
    }

    float* hostdata1;
    float* hostdata2;
    float* hostdata3;
    float* hostdata4;
    float* hostdata5;

    //allocate locked memory
    int N = 262144;
    size_t size;

    size = N * sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata1, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"1"<<" Timeuse: "<<time<<" (s)"<<endl;

    size = N *10 *sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata2, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"2"<<" Timeuse: "<<time<<" (s)"<<endl;

    size = N *20* sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata3, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"3"<<" Timeuse: "<<time<<" (s)"<<endl;

    size = N *100* sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata4, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"4"<<" Timeuse: "<<time<<" (s)"<<endl;

    size = N * 500*sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata5, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"5"<<" Timeuse: "<<time<<" (s)"<<endl;

}

int main()
{

    cuInit(0);
    cudaSetDevice(1);

    //synchronize thread 1
    workend1.lock();

    //set cpu
    /*
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }
    */

    //Context
    cout<<"Create context"<<endl;
    int err;
    CUcontext cont1;
    CUdevice dev;
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

    //400M data for data transfer
    int N = 262144;
    size_t size = N*400*sizeof(float);

    //allocate device variable(data)
    float* d_A;
    cudaMalloc(&d_A, size);

    // Allocate input vectors h_A in host memory
    float* h_A;
    cudaMallocHost(&h_A, size);

    // create thread
    thread first=thread(thread1,cont1,d_A,h_A,size,1);
    thread second=thread(thread2,cont1,2);
    first.join();
    second.join();

    workend1.lock();
    cudaFree(d_A);

    return 0;
}
