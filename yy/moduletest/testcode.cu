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
float* host1;
float* device1;
float* host2;
float* device2;
size_t newsize=600*1024*1024;
//Mutex
mutex mtx1_1;
mutex mtx1_2;


__global__ void testkernel(float n1,float n2) {
    float n3;
    for(int i=0;i<100000;i++)
    {
    n3 =n1/n2;
    }
}


//yy add
//void thread1(CUcontext ctx)
void thread1(CUcontext ctx)
{
   //1.cpu bundle
   /*
   cpu_set_t mask;
   CPU_ZERO(&mask);
   CPU_SET(16, &mask); //指定该线程使用的CPU
   if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
   {
           perror("pthread_setaffinity_np");
   }
   */

   CUcontext* pctx;
   int err;

   //2.create new context?
   /*
   cudaSetDevice(3);
   //here,maybe just cudaSetDevice can make change
   CUcontext tempcont;
   CUdevice dev;
   err = cuCtxGetDevice(&dev);
   if(err)
   {
       std::cout<<"Can't get device, err" << err<<std::endl;
   }
   err = cuCtxCreate(&tempcont,CU_CTX_SCHED_YIELD,dev);
   if(err)
   {
       std::cout<<"Can't create Context, err" << err << std::endl;
   }
   err=cuCtxGetCurrent(pctx);
   if(err)
   {
       std::cout<<"Get current context, err" << err<<std::endl;
   }
   std::cout<<"new context"<<*pctx<<std::endl;
    */

   //3.push old context and load
   err=cuCtxPushCurrent(ctx);
   if(err){
   std::cout<<"Push Context ERR! "<<err<<std::endl;
   }

   err=cuCtxGetCurrent(pctx);
   if(err)
   {
       std::cout<<"Get current context, err" << err<<std::endl;
   }
   std::cout<<"set context"<<*pctx<<std::endl;

   //2.5 create stream
   CUstream onestream;
   cuStreamCreate(&onestream,0);
   cudaMemcpyAsync(device1, host1,newsize, cudaMemcpyHostToDevice, onestream);
   //testkernel<<<20, 128>>>(1.0,2.0);
   cuStreamSynchronize(onestream);

   size_t now=0;
   size_t total=0;
   cudaMemGetInfo(&now,&total);
   //std::cout<<"1 Size before"<<now<<std::endl;
   CUmodule mod1,mod2,mod3,mod4,mod5,mod6;
   //mtx1_1.lock();
   //sleep(2);
   cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/yy/moduletest/temp1.ptx");
   /*
   cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/yy/moduletest/temp2.ptx");
   cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/yy/moduletest/temp3.ptx");
   cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/yy/moduletest/temp4.ptx");
   cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/yy/moduletest/temp5.ptx");
   cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/yy/moduletest/temp6.ptx");
   cudaMemGetInfo(&now,&total);
   */
   //std::cout<<"1 Size after"<<now<<std::endl;
}

void thread2(CUcontext ctx)
{
   //1.cpu bundle
   /*
   cpu_set_t mask;
   CPU_ZERO(&mask);
   CPU_SET(16, &mask); //指定该线程使用的CPU
   if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
   {
           perror("pthread_setaffinity_np");
   }
   */

   CUcontext* pctx;
   int err;

   //2.create new context?
   /*
   cudaSetDevice(3);
   //here,maybe just cudaSetDevice can make change
   CUcontext tempcont;
   CUdevice dev;
   err = cuCtxGetDevice(&dev);
   if(err)
   {
       std::cout<<"Can't get device, err" << err<<std::endl;
   }
   err = cuCtxCreate(&tempcont,CU_CTX_SCHED_YIELD,dev);
   if(err)
   {
       std::cout<<"Can't create Context, err" << err << std::endl;
   }
   err=cuCtxGetCurrent(pctx);
   if(err)
   {
       std::cout<<"Get current context, err" << err<<std::endl;
   }
   std::cout<<"new context"<<*pctx<<std::endl;
    */

   //3.push old context and load
   err=cuCtxPushCurrent(ctx);
   if(err){
   std::cout<<"Push Context ERR! "<<err<<std::endl;
   }

   err=cuCtxGetCurrent(pctx);
   if(err)
   {
       std::cout<<"Get current context, err" << err<<std::endl;
   }
   std::cout<<"set context"<<*pctx<<std::endl;

   //2.5 create stream
   CUstream onestream;
   cuStreamCreate(&onestream,0);
   cudaMemcpyAsync(device2, host2,newsize, cudaMemcpyHostToDevice, onestream);
   //testkernel<<<20, 128>>>(1.0,2.0);
   cuStreamSynchronize(onestream);
   size_t now=0;
   size_t total=0;
   cudaMemGetInfo(&now,&total);

   //std::cout<<"1 Size before"<<now<<std::endl;
   CUmodule mod1,mod2,mod3,mod4,mod5,mod6;

   //sleep(2);
   //mtx1_1.unlock();
   cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/yy/moduletest/temp2.ptx");
   /*
   cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/yy/moduletest/temp2.ptx");
   cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/yy/moduletest/temp3.ptx");
   cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/yy/moduletest/temp4.ptx");
   cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/yy/moduletest/temp5.ptx");
   cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/yy/moduletest/temp6.ptx");
   cudaMemGetInfo(&now,&total);
   */
   //std::cout<<"1 Size after"<<now<<std::endl;
}

int main()
{
    cuInit(0);
    cudaSetDevice(3);

    //1.create context
    CUcontext cont1;
    CUcontext cont2;
    CUdevice dev;
    int err;
    err = cuCtxGetDevice(&dev);
    if(err)
    {
        std::cout<<"Can't get device, err" << err<<std::endl;
        return 0;
    }
    err = cuCtxCreate(&cont1,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        std::cout<<"Can't create Context, err" << err << std::endl;
        return 0;
    }
    /*
    err = cuCtxCreate(&cont2,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        std::cout<<"Can't create Context, err" << err << std::endl;
        return 0;
    }
    */

   /*
   err=cuCtxPushCurrent(cont1);
   CUcontext* mainpctx;
   err=cuCtxGetCurrent(mainpctx);
   if(err)
   {
       std::cout<<"Get current context, err" << err<<std::endl;
   }

   */
    std::cout<<"main context"<<cont1<<std::endl;


    //testkernel<<<20, 128>>>(1.0,2.0);
    /*
    //1.1 kernel?
    testkernel<<<20, 128>>>(1.0,2.0);

    //1.2 data transfer?

    CUstream firststream;
    cuStreamCreate(&firststream,0);


    float* cpudata;
    CUdeviceptr gpudata;
    size_t size = 5*1024*1024;
    cuMemAllocHost((void**)(&cpudata),size);
    cuMemAlloc((CUdeviceptr*)(&gpudata), size);


    for(int i=0;i<(5*1024*1024/4);i++)
    {
        //cpudata[i]=1.0;
    }


    cuMemcpyHtoDAsync((CUdeviceptr)(gpudata),cpudata,size,firststream);
    cuStreamSynchronize(firststream);
    */

    //2.test in the mom thread
    /*
    size_t now=0;
    size_t total=0;
    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now before module load "<<now<<std::endl;

    CUmodule mod1,mod2,mod3,mod4,mod5,mod6;
    CUmodule mod7,mod8,mod9,mod10,mod11,mod12;
    cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/yy/moduletest/temp1.ptx");
    cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/yy/moduletest/temp2.ptx");
    cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/yy/moduletest/temp3.ptx");
    cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/yy/moduletest/temp4.ptx");
    cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/yy/moduletest/temp5.ptx");
    cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/yy/moduletest/temp6.ptx");
    cuModuleLoad(&mod7, "/home/wuhao/HUST_Test/yy/moduletest/temp7.ptx");
    cuModuleLoad(&mod8, "/home/wuhao/HUST_Test/yy/moduletest/temp8.ptx");

    cuModuleLoad(&mod9, "/home/wuhao/HUST_Test/yy/moduletest/temp9.ptx");
    cuModuleLoad(&mod10, "/home/wuhao/HUST_Test/yy/moduletest/temp10.ptx");
    cuModuleLoad(&mod11, "/home/wuhao/HUST_Test/yy/moduletest/temp11.ptx");
    cuModuleLoad(&mod12, "/home/wuhao/HUST_Test/yy/moduletest/temp12.ptx");


    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now after module load "<<now<<std::endl;
    */

    // 3. load cuda kernels
    /*
    CUfunction kernel;
    int result=cuModuleGetFunction(&kernel, mod1, "fused_add_10_kernel0");
    std::cout<<"result "<<result<<std::endl;

    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now after function get "<<now<<std::endl;


    //yy add stream
    CUstream firststream;
    cuStreamCreate(&firststream,0);

    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now after stream create "<<now<<std::endl;

    size_t storage_size1 = 602112;
    CUdeviceptr device_ptr1;
    std::vector<char> temp1;
    temp1.resize(storage_size1, 0);
    cuMemAlloc((CUdeviceptr*)&device_ptr1, storage_size1);
    cuMemcpyHtoD(device_ptr1, temp1.data(), storage_size1);

    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now after one malloc "<<now<<std::endl;

    size_t storage_size2 = 602112;
    CUdeviceptr device_ptr2;
    std::vector<char> temp2;
    temp2.resize(storage_size2, 0);
    cuMemAlloc((CUdeviceptr*)&device_ptr2, storage_size2);
    cuMemcpyHtoD(device_ptr2, temp2.data(), storage_size2);

    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now after two malloc "<<now<<std::endl;

    size_t storage_size3 = 12;
    CUdeviceptr device_ptr3;
    std::vector<char> temp3;
    temp3.resize(storage_size3, 0);
    cuMemAlloc((CUdeviceptr*)&device_ptr3, storage_size3);
    cuMemcpyHtoD(device_ptr3, temp3.data(), storage_size3);

    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now after three malloc "<<now<<std::endl;

    size_t storage_size4 = 5242880;
    CUdeviceptr device_ptr4;
    std::vector<char> temp4;
    temp3.resize(storage_size4, 0);
    cuMemAlloc((CUdeviceptr*)&device_ptr4, storage_size4);
    cuMemcpyHtoD(device_ptr4, temp4.data(), storage_size4);

    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now after four malloc "<<now<<std::endl;

    std::vector<CUdeviceptr*> kernel_arg;
    kernel_arg.push_back(&device_ptr1);
    kernel_arg.push_back(&device_ptr2);
    kernel_arg.push_back(&device_ptr3);

    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now before kernel launch "<<now<<std::endl;

    cuLaunchKernel(kernel,
        147, 1, 1,
        1024, 1, 1,
        0, firststream, (void **)kernel_arg.data(), 0 // raw_args是json中指示的storage的下标
    );

    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now after kernel launch "<<now<<std::endl;
    */

    mtx1_1.lock();
    //4.test in two child thread
    cudaMalloc(&device1,newsize);
    cuMemAllocHost((void**)(&host1), newsize);
    cudaMalloc(&device2,newsize);
    cuMemAllocHost((void**)(&host2), newsize);

    //thread first=thread(thread1,cont1,host,device,newsize);
    thread first=thread(thread1,cont1);
    //thread second=thread(thread2,cont1);
    first.join();
    //second.join();

    return 0;
}
