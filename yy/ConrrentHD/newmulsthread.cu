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

//void *thread1(void *dummy,void* d_A,void *h_A)
//
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
    //set GPU
    //cudaSetDevice(1);


    //yy change:huan yi ge wenjian hai yao gai makefile,wojiu yong zhe ge le
    //wo hui zai wo gaide mei yige difang jia shang zhushi yy
    //yy preparation

    CUevent  start, stop;
    float time;
    cuEventCreate(&start,0);
    cuEventCreate(&stop,0);

    cout<<"one thread starts: "<<endl;
    int err;
    err=cuCtxPushCurrent(ctx);
    if(err){
    cout<<"Push Context ERR! "<<err<<endl;
    }

    cudaStream_t tempstream;

    cudaError_t cudaStatus;
    cudaStatus = cudaStreamCreate(&tempstream);
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));

    /*
    int *flag = (int *)dummy;
    int *d_a = (int *)d_A;
    int *h_a = (int *)h_A;
    */

    if(i==1)
    {
    workend1.unlock();
    workend2.lock();
    }
    else
    {
    workend2.unlock();
    workend1.lock();
    }

    for(int i=1;i < 10;i++)
    {
    //cuEventRecord(start,0);
    cudaMemcpyAsync(d_a, h_a,size, cudaMemcpyHostToDevice, tempstream);
    //cuEventRecord(stop,0);
    //cuEventSynchronize(stop);
    //cuEventElapsedTime(&time, start, stop);
	//std::cout<< i <<" time: "<<1000*time<<" us"<<std::endl;
    }


    cuStreamSynchronize(tempstream);
}

/*
pthread_t ntid1;
pthread_t ntid2;
*/

int main()
{
    //preparation
    workend1.lock();
    workend2.lock();

    cuInit(0);
    cudaSetDevice(1);
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

    //40M data
    int N = 4*52428800/20;
    size_t size = N * sizeof(float);

    //allocate device variable(data)
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);


    // Allocate input vectors h_A and h_B in host memory
    float* h_A;
    float* h_B;
    float* h_C;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);

    uniform_real_distribution<float> u(0,10);
    default_random_engine e(time(NULL));
    for(int i=0;i < N; ++i){
    /*
    *(h_A + i) = u(e);
	*(h_B + i) = u(e);
	*(h_C + i) = u(e);
    */
    *(h_A + i) = 1;
	*(h_B + i) = 1;
	*(h_C + i) = 1;
    }


    CUevent  start, stop;
    float time;
    cuEventCreate(&start,0);
    cuEventCreate(&stop,0);

    for(int i=1;i < 10;i++)
    {
    cuEventRecord(start,0);
    cudaMemcpyAsync(d_A, h_A,size, cudaMemcpyHostToDevice, 0);
    cuEventRecord(stop,0);
    cuEventSynchronize(stop);
    cuEventElapsedTime(&time, start, stop);
	std::cout<< i <<" time: "<<1000*time<<" us"<<std::endl;
    }

    /*
    pthread_create(&ntid1, NULL, thread1, flag1,d_A,h_A);
    pthread_create(&ntid2, NULL, thread1, flag2,d_B,h_B);
    pthread_join(ntid1, NULL);
    pthread_join(ntid2, NULL);
    */

    //thread second=thread(thread1,cont1,d_B,h_B,size,1);
    //thread first=thread(thread1,cont1,d_A,h_A,size,2);
    //second.join();
    //first.join();

    //change,check whether the cudamemcpy works
    for(int i=0;i < N; ++i){
    /*
    *(h_A + i) = u(e);
	*(h_B + i) = u(e);
	*(h_C + i) = u(e);
    */
    *(h_A + i) = 0;
	*(h_B + i) = 0;
	*(h_C + i) = 0;
    }
    cudaMemcpy(h_A, d_A,size, cudaMemcpyDeviceToHost);
    for(int i=0;i < 10; ++i){
    //cout<<"now  data"<<*(h_A + i)<<endl;
    }


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
.
    return 0;
}
