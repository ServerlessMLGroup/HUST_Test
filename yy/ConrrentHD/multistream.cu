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
//clock_t
clock_t start1,finish1;
clock_t start1_2,finish1_2;
clock_t start2,finish2;
double singletime = 0.0;
double cotime1=0.0;
double cotime2=0.0;

void CUDART_CB thread1_1callback(void *data) {
    mtx1_1.lock();
    start1=clock();
}

void CUDART_CB thread1_2callback(void *data) {
    finish1=clock();
    singletime += (double)(finish1 - start1)/CLOCKS_PER_SEC;
    cout<<"This time single data transfer: "<<((double)(finish1-start1)/CLOCKS_PER_SEC)<<"(s)"<<endl;
}

void CUDART_CB thread1_3callback(void *data) {
    mtx2_1.unlock();
    mtx1_2.lock();
    start1_2=clock();
}

void CUDART_CB thread1_4callback(void *data) {
    finish1=clock();
    cotime1 += (double)(finish1_2-start1_2)/CLOCKS_PER_SEC;
    cout<<"This time cocurrent data transfer 1111: "<<((double)(finish1-start1)/CLOCKS_PER_SEC)<<"(s)"<<endl;
    mtx1_1.unlock();
}
void CUDART_CB thread1_5callback(void *data) {
    cout<<"single time: "<<singletime<<" s"<<endl;
    cout<<"cocurrent time1111: "<<cotime1<<" s"<<endl;
}

void CUDART_CB thread2_1callback(void *data) {
    mtx2_1.lock();
    start2=clock();
}

void CUDART_CB thread2_2callback(void *data) {
    finish2=clock();
    cotime2 += (double)(finish2-start2)/CLOCKS_PER_SEC;
    cout<<"This time cocurrent data transfer 2222: "<<((double)(finish2-start2)/CLOCKS_PER_SEC)<<"(s)"<<endl;
    mtx1_2.unlock();
}

void CUDART_CB thread2_3callback(void *data) {
    cout<<"cocurrent time2222: "<<cotime2<<" s"<<endl;
}

void thread1(cudaStream_t stream,float* d_a,float* d_b,float* h_a,float* h_b,size_t size)
{
    cudaHostFn_t fn1 = thread1_1callback;
    cudaHostFn_t fn2 = thread1_2callback;
    cudaHostFn_t fn3 = thread1_3callback;
    cudaHostFn_t fn4 = thread1_4callback;
    cudaHostFn_t fn5 = thread1_5callback;
    for(int i=0;i < 10;i++)
    {
    cudaLaunchHostFunc(stream, fn1, 0);
    cudaMemcpyAsync(d_a, h_a,size, cudaMemcpyHostToDevice, stream);
    cudaLaunchHostFunc(stream, fn2, 0);
    cudaLaunchHostFunc(stream, fn3, 0);
    cudaMemcpyAsync(d_b, h_b,size, cudaMemcpyHostToDevice, stream);
    cudaLaunchHostFunc(stream, fn4, 0);
    }
    //Should i add some code to exit the thread here?
    cudaLaunchHostFunc(stream, fn5, 0);
}

void thread2(cudaStream_t stream,float* d_c,float* h_c,size_t size)
{
    cudaHostFn_t fn1 = thread2_1callback;
    cudaHostFn_t fn2 = thread2_2callback;
    cudaHostFn_t fn3 = thread2_3callback;
    for(int i=0;i < 10;i++)
    {
    cudaLaunchHostFunc(stream, fn1, 0);
    cudaMemcpyAsync(d_c, h_c,size, cudaMemcpyHostToDevice, stream);
    cudaLaunchHostFunc(stream, fn2, 0);
    }
    cudaLaunchHostFunc(stream, fn3, 0);
}

int main()
{
    cuInit(0);
    cudaSetDevice(2);
    //clock for collection

    //data size
    int N = 10485760;
    size_t size = N * sizeof(float);
    int err;

    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);


    cout<<"Allocate Host Memory"<<endl;
    // Allocate input vectors h_A and h_B in host memory
    float* h_A,h_B,h_C;
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
    cudaStream_t* firststream,secondstream;
    cudaStreamCreate(firststream);
    cudaStreamCreate(secondstream);

    //prepare
    mtx2_1.lock();
    thread first=thread(thread1,firststream,d_A,d_B,h_A,h_B,size);
    thread second=thread(thread2,secondstream,d_C,h_C,size);
    second.join();
    first.join();
    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
