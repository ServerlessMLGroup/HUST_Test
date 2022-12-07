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
    cudaSetDevice(2);

    //cpu_set_t mask;
    /*
    CPU_ZERO(&mask);
    CPU_SET(15, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }
    */


    //data size, 262144 > 1 M
    int N = 262144;
    size_t size = N * sizeof(float);
    int datasize = 20;
    float* devicedata[datasize];
    float* hostdata[datasize];

    for(int i=0;i<datasize;i++)
    {
    cudaMalloc(&devicedata[i], size);
    }

    for(int i=0;i<datasize;i++)
    {
    cudaMallocHost(&hostdata[i], size);
    }

    uniform_real_distribution<float> u(0,10);
    default_random_engine e(time(NULL));
    for(int i=0;i < datasize; ++i){
    //*hostdata[i] = u(e);
    }

    //Create Stream
    cudaStream_t firststream;
    cudaStreamCreate(&firststream);

    for(int i=0;i<datasize;i++)
    {
    cudaMemcpyAsync(devicedata[i], hostdata[i], size, cudaMemcpyHostToDevice, firststream);
    }

    cudaDeviceSynchronize();
    //Free memory
    for(int i=0;i<datasize;i++)
    {
    cudaFree(devicedata[i]);
    }

    return 0;
}
