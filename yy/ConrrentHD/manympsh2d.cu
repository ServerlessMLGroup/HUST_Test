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

void thread1(CUcontext ctx,float* d_a,float* h_a,size_t size,int i)
{
    //set CPU
    clock_t start,finish;
    double time=0.0;
    /*
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }
    */
    cout<<"one thread starts: "<<endl;
    int err;
    err=cuCtxPushCurrent(ctx);

    if(err){
    cout<<"Push Context ERR! "<<err<<endl;
    }
    for(int i=0;i < 10;i++)
    {
    start=clock();
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    finish=clock();
    time += (double)(finish-start)/CLOCKS_PER_SEC;
    }
    cout <<i<<" Timeuse: "<<time<<" (s)"<<endl;
}



int main()
{
    /*
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }
    */

    cuInit(0);
    cudaSetDevice(2);
    //clock for collection


    //data size
    int N = 209715200;
    size_t size = N * sizeof(float);

    //Context and memory
    cout<<"Create n contexts and their memory"<<endl;
    int err;
    CUcontext cont1,cont2,cont3,cont4,cont5,cont6,cont7,cont8,cont9,cont10;
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
    float* h_A;
    //cudaMallocHost(&h_A, size);
    h_A = (float*)malloc(size);

    err = cuCtxCreate(&cont2,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        cout<<"Can't create Context, err" << err << endl;
        return 0;
    }
    float* d_B;
    cudaMalloc(&d_B, size);
    float* h_B;
    //cudaMallocHost(&h_B, size);
    h_B = (float*)malloc(size);

    err = cuCtxCreate(&cont3,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        cout<<"Can't create Context, err" << err << endl;
        return 0;
    }
    float* d_C;
    cudaMalloc(&d_C, size);
    float* h_C;
    //cudaMallocHost(&h_C, size);
    h_C = (float*)malloc(size);

    err = cuCtxCreate(&cont4,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        cout<<"Can't create Context, err" << err << endl;
        return 0;
    }
    float* d_D;
    cudaMalloc(&d_D, size);
    float* h_D;
    //cudaMallocHost(&h_D, size);
    h_D = (float*)malloc(size);

    err = cuCtxCreate(&cont5,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        cout<<"Can't create Context, err" << err << endl;
        return 0;
    }
    float* d_E;
    cudaMalloc(&d_E, size);
    float* h_E;
    //cudaMallocHost(&h_E, size);
    h_E = (float*)malloc(size);

    uniform_real_distribution<float> u(0,10);
    default_random_engine e(time(NULL));
    for(int i=0;i < N; ++i){
    *(h_A + i) = u(e);
	*(h_B + i) = u(e);
	*(h_C + i) = u(e);
	*(h_D + i) = u(e);
	*(h_E + i) = u(e);
    }


    thread th1=thread(thread1,cont1,d_A,h_A,size,1);
    thread th2=thread(thread1,cont2,d_B,h_B,size,2);
    thread th3=thread(thread1,cont3,d_C,h_C,size,3);
    thread th4=thread(thread1,cont4,d_D,h_D,size,4);
    thread th5=thread(thread1,cont5,d_E,h_E,size,5);

    th1.join();
    th2.join();
    th3.join();
    th4.join();
    th5.join();

    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);

    return 0;
}
