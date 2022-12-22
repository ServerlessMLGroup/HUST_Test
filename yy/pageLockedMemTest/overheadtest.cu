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


int main()
{
    cuInit(0);
    cudaSetDevice(1);

    //set CPU
    clock_t start,finish;
    double time=0.0;
    /*
    cout<<"set cpu"<<endl;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }
    */



    //data size, 262144 > 1 M
    int N = 262144;
    size_t size = N * sizeof(float);
    float* hostdata1;
    float* hostdata2;
    float* hostdata3;
    float* hostdata4;
    float* hostdata5;
    float* hostdata6;
    float* hostdata7;
    float* hostdata8;
    float* hostdata9;
    float* hostdata0;
    float* devicedata;


    cudaMalloc(&devicedata,size)
    cudaMallocHost(&hostdata, size);

    //Create Stream
    cudaStream_t firststream;
    cudaStreamCreate(&firststream);

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


    //
    cout <<"Add computation between malloc host"<<endl;
    int temp=0;
    for(int i=;i<1000;i++)
    {
    temp++;
    }
    size = N * sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata6, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"6"<<" Timeuse: "<<time<<" (s)"<<endl;

    for(int i=;i<1000;i++)
    {
    temp++;
    }
    size = N *10 *sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata7, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"7"<<" Timeuse: "<<time<<" (s)"<<endl;

    for(int i=;i<1000;i++)
    {
    temp++;
    }
    size = N *20* sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata8, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"8"<<" Timeuse: "<<time<<" (s)"<<endl;

    for(int i=;i<1000;i++)
    {
    temp++;
    }
    size = N *100* sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata9, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"9"<<" Timeuse: "<<time<<" (s)"<<endl;

    for(int i=;i<1000;i++)
    {
    temp++;
    }
    size = N * 500*sizeof(float);
    start=clock();
    cudaMallocHost(&hostdata0, size);
    finish=clock();
    time = (double)(finish-start)/CLOCKS_PER_SEC;
    cout <<"10"<<" Timeuse: "<<time<<" (s)"<<endl;


    //Free memory
    cudaFree(devicedata);

    return 0;
}







