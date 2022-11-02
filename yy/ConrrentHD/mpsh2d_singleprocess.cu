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
#include <stdio.h>
using namespace std;

//Mutex
mutex mtx1;
mutex mtx2;

void thread1(CUcontext ctx,float* d_a,float* d_b,float* h_a,float* h_b,size_t size)
{
    //set CPU

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }

    /*
    float* h_A;
    float* h_B;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    */

    cout<<"game start "<<endl;

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
    mtx1.lock();
    start=clock();
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    finish=clock();
    singletime += (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"This time single data transfer: "<<((double)(finish-start)/CLOCKS_PER_SEC)<<"(s)"<<endl;
    cout<<"1-1 timeline: "<<(double)(start)/CLOCKS_PER_SEC<<" to "<<(double)(finish)/CLOCKS_PER_SEC<<endl;
    mtx2.unlock();

    start=clock();
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    finish=clock();
    cotime += (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"This time concurrent data 111 transfer: "<<((double)(finish-start)/CLOCKS_PER_SEC)<<"(s)"<<endl;
    cout<<"1-2 timeline: "<<(double)(start)/CLOCKS_PER_SEC<<" to "<<(double)(finish)/CLOCKS_PER_SEC<<endl;
    }

    cout<<"single time: "<<singletime<<" s"<<endl;
    cout<<"cocurrent time1: "<<cotime<<" s"<<endl;
    /*
    while(1){
    sleep(1);
    cout<<"I'm alive"<<endl;
    }
    */

}

void thread2(CUcontext ctx,float* d_c,float* h_c,size_t size)
{

    cpu_set_t mask;
    CPU_ZERO(&mask);

    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }

    /*
    float* h_C;
    cudaMallocHost(&h_C, size);
    */

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
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);
    finish=clock();
    singletime += (double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"This time concurrent 222 data transfer: "<<((double)(finish-start)/CLOCKS_PER_SEC)<<"(s)"<<endl;
    cout<<"2-1 timeline: "<<(double)(start)/CLOCKS_PER_SEC<<" to "<<(double)(finish)/CLOCKS_PER_SEC<<endl;
    mtx1.unlock();
    }
    cout<<"cocurrent time2: "<<singletime<<" s"<<endl;
    cout<<"game end"<<endl;
   /*
    while(1){
    sleep(1);
    cout<<"I'm alive"<<endl;
    }
    */
}

int main()
{
    cuInit(0);
    cudaSetDevice(1);
    //clock for collection

    //data size
    int N = 209715200;
    size_t size = N * sizeof(float);

    /*
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(15, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }
    */


    //Context and memory
    cout<<"Create two context and their memory"<<endl;
    int err;
    CUcontext cont1,cont2;
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
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);

    //cudaSetDevice(1);
    err = cuCtxCreate(&cont2,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        cout<<"Can't create Context, err" << err<<endl;
        return 0;
    }
    float* d_C;
    cudaMalloc(&d_C, size);


    cout<<"Allocate Host Memory"<<endl;
    // Allocate input vectors h_A and h_B in host memory

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    /*
    float* h_A;
    float* h_B;
    float* h_C;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);
    */

    uniform_real_distribution<float> u(0,10);
    default_random_engine e(time(NULL));
    for(int i=0;i < N; ++i){
    *(h_A + i) = u(e);
	*(h_B + i) = u(e);
	*(h_C + i) = u(e);
    }

    //prepare
    mtx2.lock();
    thread first=thread(thread1,cont1,d_A,d_B,h_A,h_B,size);

    thread second=thread(thread2,cont2,d_C,h_C,size);
    second.join();
    first.join();
    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
