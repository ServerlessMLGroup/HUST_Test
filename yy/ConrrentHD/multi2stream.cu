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
mutex mtx1_1;
mutex mtx1_2;
mutex mtx2_1;
mutex workend1;
mutex workend2;
//mutex test;
//clock_t
clock_t start1,finish1;
clock_t start1_2,finish1_2;
clock_t start2,finish2;
double singletime = 0.0;
double cotime1=0.0;
double cotime2=0.0;

void CUDART_CB thread1_1callback(void *data) {
    //mtx1_1.lock();
    start1=clock();
    //test.unlock();
}


void CUDART_CB thread1_2callback(void *data) {
    finish1=clock();
    singletime += (double)(finish1 - start1)/CLOCKS_PER_SEC;
    cout<<"This time single data transfer: "<<((double)(finish1-start1)/CLOCKS_PER_SEC)<<"(s)"<<endl;
    cout<<"1-1 timeline: "<<(double)(start1)/CLOCKS_PER_SEC<<" to "<<(double)(finish1)/CLOCKS_PER_SEC<<endl;
}

void CUDART_CB thread1_3callback(void *data) {
    //mtx2_1.unlock();
    //mtx1_2.lock();
    start1_2=clock();
}

void CUDART_CB thread1_4callback(void *data) {
    finish1_2=clock();
    cotime1 += (double)(finish1_2-start1_2)/CLOCKS_PER_SEC;
    cout<<"This time cocurrent data transfer 1111: "<<((double)(finish1_2-start1_2)/CLOCKS_PER_SEC)<<"(s)"<<endl;
    cout<<"1-2 timeline: "<<(double)(start1_2)/CLOCKS_PER_SEC<<" to "<<(double)(finish1_2)/CLOCKS_PER_SEC<<endl;
    //mtx1_1.unlock();
}
void CUDART_CB thread1_5callback(void *data) {
    cout<<"single time: "<<singletime<<" s"<<endl;
    cout<<"cocurrent time1111: "<<cotime1<<" s"<<endl;
    workend1.unlock();
}

void CUDART_CB thread2_1callback(void *data) {
    //mtx2_1.lock();
    start2=clock();
}

void CUDART_CB thread2_2callback(void *data) {
    finish2=clock();
    cotime2 += (double)(finish2-start2)/CLOCKS_PER_SEC;
    cout<<"This time cocurrent data transfer 2222: "<<((double)(finish2-start2)/CLOCKS_PER_SEC)<<"(s)"<<endl;
    cout<<"2-1 timeline: "<<(double)(start2)/CLOCKS_PER_SEC<<" to "<<(double)(finish2)/CLOCKS_PER_SEC<<endl;
    //mtx1_2.unlock();
}

void CUDART_CB thread2_3callback(void *data) {
    cout<<"cocurrent time2222: "<<cotime2<<" s"<<endl;
    workend2.unlock();
}


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

    //data size
    //int N = 4*52428800;
    int N = 52428800/4;
    size_t size = N * sizeof(float);

    //
    double testtime;

    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);

    //cudaSetDevice(0);
    float* d_C;
    cudaMalloc(&d_C, size);

    //test.lock();
    cout<<"Allocate Host Memory"<<endl;
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
    *(h_A + i) = u(e);
	*(h_B + i) = u(e);
	*(h_C + i) = u(e);
    }

    //Create Stream
    cudaStream_t firststream;
    cudaStream_t secondstream;
    cudaStreamCreate(&firststream);
    cudaStreamCreate(&secondstream);

    //prepare
    mtx2_1.lock();
    workend1.lock();
    workend2.lock();

    //divide the formal funtion here
    cudaHostFn_t fn1 = thread1_1callback;
    cudaHostFn_t fn2 = thread1_2callback;
    cudaHostFn_t fn3 = thread1_3callback;
    cudaHostFn_t fn4 = thread1_4callback;
    cudaHostFn_t fn5 = thread1_5callback;
    cudaHostFn_t fn6 = thread2_1callback;
    cudaHostFn_t fn7 = thread2_2callback;
    cudaHostFn_t fn8 = thread2_3callback;


    for(int i=0;i < 10;i++)
    {
    //cudaLaunchHostFunc(secondstream, fn6, 0);
    //cudaMemcpyAsync(d_C, h_C,size, cudaMemcpyHostToDevice, secondstream);
    //cudaLaunchHostFunc(secondstream, fn7, 0);
    /*
    cudaLaunchHostFunc(firststream, fn3, 0);
    cudaMemcpyAsync(d_B, h_B,size, cudaMemcpyHostToDevice, firststream);
    cudaLaunchHostFunc(firststream, fn4, 0);
    */
    //cudaLaunchHostFunc(secondstream, fn3, 0);
    cudaMemcpyAsync(d_A, h_A, size/2, cudaMemcpyHostToDevice, secondstream);
    //cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, firststream);
    //cudaLaunchHostFunc(secondstream, fn4, 0);

    }

    for(int i=0;i < 10;i++)
    {
    //cudaLaunchHostFunc(firststream, fn1, 0);
    //test.lock();
    //cout<<"Pass the test"<<endl;
    cudaMemcpyAsync(d_B, h_B,size, cudaMemcpyHostToDevice, firststream);
    //cudaLaunchHostFunc(firststream, fn2, 0);
    //cudaLaunchHostFunc(firststream, fn3, 0);
    //cudaMemcpyAsync(d_B, h_B,size, cudaMemcpyHostToDevice, firststream);
    //cudaLaunchHostFunc(firststream, fn4, 0);
    }
    //Should i add some code to exit the thread here?

    cudaLaunchHostFunc(firststream, fn5, 0);


    cudaLaunchHostFunc(secondstream, fn8, 0);

    workend1.lock();
    workend2.lock();
    cout<<"It can't be like this"<<endl;
    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
