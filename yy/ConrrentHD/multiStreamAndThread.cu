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
#include<cuda_runtime.h>
using namespace std;
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

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

__global__ void kernel(float n1, float n2, float n3, int stop) {

	for (int i = 0; i < stop; i++) {
		n1=sinf(n1);
		n2=n3/n2;
	}

}


__global__ void kernel_timer(long long unsigned *times,int *flag) {
		unsigned long long mclk2;
		int i=0;
		while(i<11)
		{
		    
		    while(flag[0] != 1) {
		        __nanosleep(5000); // 500us
		        //__syncthreads();
              }

		    if (threadIdx.x == 0){
		    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk2));
		    times[i] = mclk2/ 1000000;
		    }
		    i++;
		}
}


__global__ void kernel_flager(int i,int *flag) {
		flag[i] = 1;
}

void CUDART_CB thread1_5callback(void *data) {
    workend1.unlock();
}

void CUDART_CB thread2_3callback(void *data) {
    workend2.unlock();
}

void thread1(cudaStream_t stream,float* d_a,float* h_a,size_t size,long long unsigned *timeline,int number,int *flag)
{
    //set CPU
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    {
            perror("pthread_setaffinity_np");
    }
    //kernel<<<1,1,0,stream>>>(1.0,2.0,3.0,100);
    kernel_flager<<<1,1,0,stream>>>(0,flag);

    for(int i=1;i < 11;i++)
    {
    //kernel<<<1,1,0,stream>>>(1.0,2.0,3.0,100);
    cudaMemcpyAsync(d_a, h_a,size, cudaMemcpyHostToDevice, stream);
    kernel_flager<<<1,1,0,stream>>>(i,flag);
    //kernel<<<1,1,0,stream>>>(1.0,2.0,3.0,100);
    }

}

int main()
{
    cuInit(0);
    cudaSetDevice(2);

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0) {
            perror("pthread_setaffinity_np");
    }

    int N = 4*52428800/20;
    size_t size = N * sizeof(float);

    double testtime;
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);

    float* d_C;
    cudaMalloc(&d_C, size);

    //create timeline and flag
    long long unsigned *timeline1;
	long long unsigned *timeline2;
    int *flag1;
    int *flag2;
    int *flag1h;
    int *flag2h;
    flag1h = (int*) malloc(11 * sizeof(int));
    flag2h = (int*) malloc(11 * sizeof(int));

    //use 111 instaead to check
	size_t size2 = 111 * sizeof(long long unsigned);
	cudaMalloc(&timeline1, size2);
    cudaMalloc(&timeline2, size2);

    size_t size3 = 11*sizeof(int);
    cudaMalloc(&flag1, size3);
    cudaMalloc(&flag2, size3);

    for(int i=0;i<11;i++)
    {
    flag1h[i]=0;
    flag2h[i]=0;
    }

    cudaMemcpy(flag1, flag1h, sizeof(int) * 11, cudaMemcpyHostToDevice);
    cudaMemcpy(flag2, flag2h, sizeof(int) * 11, cudaMemcpyHostToDevice);

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
    cudaStream_t flagonestream;
    cudaStream_t flagtwostream;
    cudaStreamCreate(&firststream);
    cudaStreamCreate(&secondstream);
    cudaStreamCreate(&flagonestream);
    cudaStreamCreate(&flagtwostream);

    kernel_timer<<<1,1,0,flagonestream>>>(timeline1,flag1);
    kernel_timer<<<1,1,0,flagtwostream>>>(timeline2,flag2);



    //cudaMemcpyAsync(d_A, h_A,size/2, cudaMemcpyHostToDevice, firststream);
    //cudaMemcpyAsync(d_B, h_B,size, cudaMemcpyHostToDevice, secondstream);
    //prepare

    //mtx2_1.lock();
    workend1.lock();
    workend2.lock();

    //divide the formal funtion here
    cudaHostFn_t fn5 = thread1_5callback;
    cudaHostFn_t fn8 = thread2_3callback;
    //cudaLaunchHostFunc(flagonestream, fn5, 0);
    //cudaLaunchHostFunc(flagtwostream, fn8, 0);

    thread first=thread(thread1,firststream,d_A,d_A,size,timeline1,1,flag1);
    thread second=thread(thread1,secondstream,d_B,d_B,size,timeline2,2,flag2);
    second.join();
    first.join();


    cudaLaunchHostFunc(firststream, fn5, 0);
    cudaLaunchHostFunc(secondstream, fn8, 0);


    cout<<"reach here"<<endl;
    workend1.lock();
    workend2.lock();


    long long unsigned* timelineh1;
    long long unsigned* timelineh2;
    timelineh1 =(long long unsigned*)malloc(size2);

    timelineh2 =(long long unsigned*)malloc(size2);


    cudaMemcpy(timelineh1, timeline1, size2, cudaMemcpyDeviceToHost);
    cudaMemcpy(timelineh2, timeline2, size2, cudaMemcpyDeviceToHost);

    for(int k=0;k< 11;k++)
    {
    printf("Timeline0-%d %llu (s)\n",k, timelineh1[k]);
    }
    for(int k=0;k< 11;k++)
    {
    printf("Timeline1-%d %llu (s)\n",k, timelineh2[k]);
    }

    cout<<"It can't be like this"<<endl;
    //Free memory

    cudaFree(timeline1);
    cudaFree(timeline2);
    cudaFree(flag1);
    cudaFree(flag2);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
