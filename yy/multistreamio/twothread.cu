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
    workend2.unlock();
    workend1.lock();
    }
    else
    {
    workend2.lock();
    workend1.unlock();
    }

    for(int j=0;j<10;j++)
    {
    cudaMemcpyAsync(d_a, h_a,size, cudaMemcpyHostToDevice, tempstream);
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
    cudaSetDevice(2);
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

    //262144 1M
    int N = 262144/50;
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

    thread second=thread(thread1,cont1,d_B,h_B,size,1);
    thread first=thread(thread1,cont1,d_A,h_A,size,2);
    second.join();
    first.join();

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

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
