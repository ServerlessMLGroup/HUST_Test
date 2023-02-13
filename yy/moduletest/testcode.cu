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

__global__ void testkernel(float n1,float n2) {
    float n3;
    for(int i=0;i<100000;i++)
    {
    n3 =n1/n2;
    }
}



//yy add
//yy add
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

   //2.create new context?
   /*
   cudaSetDevice(2);
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
   cuCtxGetCurrent(pctx);
   std::cout<<"new context"<<*pctx<<std::endl;
   */

   //3.push old context and load
   int err;
   CUcontext* pctx;
   err=cuCtxPushCurrent(ctx);
   if(err){
   std::cout<<"Push Context ERR! "<<err<<std::endl;
   }
   cuCtxGetCurrent(pctx);
   std::cout<<"set context"<<*pctx<<std::endl;

   size_t now=0;
   size_t total=0;
   cudaMemGetInfo(&now,&total);
   std::cout<<"1 Size before"<<now<<std::endl;
   CUmodule mod1,mod2,mod3,mod4,mod5,mod6;
   cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/yy/moduletest/temp1.ptx");
   cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/yy/moduletest/temp2.ptx");
   cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/yy/moduletest/temp3.ptx");
   cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/yy/moduletest/temp4.ptx");
   cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/yy/moduletest/temp5.ptx");
   cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/yy/moduletest/temp6.ptx");
   cudaMemGetInfo(&now,&total);
   std::cout<<"1 Size after"<<now<<std::endl;
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

   //2.create new context?
   /*
   cudaSetDevice(2);
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
   cuCtxGetCurrent(pctx);
   std::cout<<"new context"<<*pctx<<std::endl;
   */


   int err;
   CUcontext* pctx;
   err=cuCtxPushCurrent(ctx);
   if(err){
   std::cout<<"Push Context ERR! "<<err<<std::endl;
   }
   cuCtxGetCurrent(pctx);
   std::cout<<"set context"<<*pctx<<std::endl;

   size_t now=0;
   size_t total=0;
   cudaMemGetInfo(&now,&total);
   std::cout<<"2 Size before"<<now<<std::endl;
   CUmodule mod1,mod2,mod3,mod4,mod5,mod6;
   cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/yy/moduletest/temp7.ptx");
   cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/yy/moduletest/temp8.ptx");
   cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/yy/moduletest/temm9.ptx");
   cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/yy/moduletest/temp10.ptx");
   cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/yy/moduletest/temp11.ptx");
   cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/yy/moduletest/temp12.ptx");
   cudaMemGetInfo(&now,&total);
   std::cout<<"2 Size after"<<now<<std::endl;
}

int main()
{
    cuInit(0);
    cudaSetDevice(2);


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
    err = cuCtxCreate(&cont2,CU_CTX_SCHED_YIELD,dev);
    if(err)
    {
        std::cout<<"Can't create Context, err" << err << std::endl;
        return 0;
    }

    //1.1 kernel?
    testkernel<<<20, 128>>>(1.0,2.0);

    //1.2 data transfer?
    CUstream firststream;
    cuStreamCreate(&firststream,0);


    float* cpudata;
    CUdeviceptr* gpudata;
    size_t size = 5*1024*1024;
    cuMemAllocHost((void**)(&cpudata),size);
    //cuMemAlloc(gpudata, size);
    /*
    for(int i=0;i<5*1024*1024;i++)
    {
        cpudata[i]=1.0;
    }
    */
    //cuMemcpyHtoDAsync((CUdeviceptr)(*gpudata),cpudata,size,firststream);
    cuStreamSynchronize(firststream);


    //2.test in the mom thread
    size_t now=0;
    size_t total=0;
    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now"<<now<<std::endl;

    CUmodule mod1,mod2,mod3,mod4,mod5,mod6;
    cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/yy/moduletest/temp1.ptx");
    cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/yy/moduletest/temp2.ptx");
    cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/yy/moduletest/temp3.ptx");
    cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/yy/moduletest/temp4.ptx");
    cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/yy/moduletest/temp5.ptx");
    cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/yy/moduletest/temp6.ptx");

    cudaMemGetInfo(&now,&total);
    std::cout<<"Size now"<<now<<std::endl;

    //3.test in two child thread
    /*
    thread first=thread(thread1,cont1);
    thread second=thread(thread2,cont1);
    first.join();
    second.join();
    */

    return 0;
}
