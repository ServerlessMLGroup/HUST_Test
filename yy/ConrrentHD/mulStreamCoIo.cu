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

int main()
{
    cuInit(0);
    cudaSetDevice(1);
    //clock for collection

    //cpu_set_t mask;
    /*
    CPU_ZERO(&mask);
    CPU_SET(15, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }
    */

    //data size, 209715200 > 800 M
    int N = 209715200/20;
    size_t size = N * sizeof(float);

    //Alloc Device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Allocate input vectors
    float* h_A;
    float* h_B;
    float* h_C;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);

    uniform_real_distribution<float> u(0,10);
    default_random_engine e(time(NULL));
    for(int i=0;i < N; ++i){
    *(h_A + i) = u(e);
	*(h_B + i) = u(e);
	*(h_C + i) = u(e);
    }

    //Create Stream
    cudaStream_t firststream;
    cudaStream_t secondstream;
    cudaStreamCreate(&firststream);
    cudaStreamCreate(&secondstream);

    //test
    /*
    cudaMemcpyAsync(d_C, h_C, size, cudaMemcpyHostToDevice, secondstream);
    cout<<"what?"<<endl;
    /*
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, firststream);
    //*/

    for(int i=0;i < 10;i++)
    {
    cudaMemcpyAsync(d_C, h_C,size, cudaMemcpyHostToDevice, secondstream);
    cudaMemcpyAsync(d_B, h_B,size/2, cudaMemcpyHostToDevice, firststream);
    }

    for(int i=0;i < 10;i++)
    {
    //cudaMemcpyAsync(d_A, h_A,size, cudaMemcpyHostToDevice, secondstream);
    }
    //Should i add some code to exit the thread here?

    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
