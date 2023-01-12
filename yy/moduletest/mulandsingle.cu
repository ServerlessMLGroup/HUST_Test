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

//yy add
//yy add
void thread1(CUcontext ctx)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(16, &mask); //指定该线程使用的CPU
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    {
            perror("pthread_setaffinity_np");
    }
   int err;
   CUcontext* pctx;

   err=cuCtxPushCurrent(ctx);
   if(err){
   std::cout<<"Push Context ERR! "<<err<<std::endl;
   }

   cuCtxGetCurrent(pctx);
   std::cout<<"set context"<<*pctx<<std::endl;

   size_t now;
   size_t total;
   cudaMemGetInfo(&now,&total);
   std::cout<<"Size now"<<now<<std::endl;
   CUmodule mod1,mod2,mod3,mod4,mod5,mod6;
   cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/yy/moduletest/temp1.ptx");
   cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/yy/moduletest/temp2.ptx");
   cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/yy/moduletest/temp3.ptx");
   cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/yy/moduletest/temp4.ptx");
   cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/yy/moduletest/temp7.ptx");
   cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/yy/moduletest/temp6.ptx");
   cudaMemGetInfo(&now,&total);
   std::cout<<"Size now"<<now<<std::endl;
}

void thread2(CUcontext ctx)
{
   int err;
   CUcontext* pctx;

   err=cuCtxPushCurrent(ctx);
   if(err){
   std::cout<<"Push Context ERR! "<<err<<std::endl;
   }
   cuCtxGetCurrent(pctx);
   std::cout<<"set context"<<*pctx<<std::endl;

   CUmodule mod1,mod2,mod3,mod4,mod5,mod6;
   cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/yy/moduletest/temp7.ptx");
   cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/yy/moduletest/temp8.ptx");
   cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/yy/moduletest/temp9.ptx");
   cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/yy/moduletest/temp10.ptx");
   cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/yy/moduletest/temp11.ptx");
   cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/yy/moduletest/temp12.ptx");
}

void thread3(CUcontext ctx)
{
   int err;
   CUcontext* pctx;

   err=cuCtxPushCurrent(ctx);
   if(err){
   std::cout<<"Push Context ERR! "<<err<<std::endl;
   }
   cuCtxGetCurrent(pctx);
   std::cout<<"set context"<<*pctx<<std::endl;

   CUmodule mod1,mod2,mod3,mod4,mod5,mod6;
   cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/yy/moduletest/temp7.ptx");
   cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/yy/moduletest/temp8.ptx");
   cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/yy/moduletest/temp9.ptx");
   cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/yy/moduletest/temp10.ptx");
   cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/yy/moduletest/temp11.ptx");
   cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/yy/moduletest/temp12.ptx");
}



int main()
{
    cuInit(0);
    cudaSetDevice(2);
    //clock for collection

    //yy change
    CUcontext cont1;
    CUdevice dev;
    int err;
    int temp;
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

    thread first=thread(thread1,cont1);
    thread second=thread(thread2,cont1);
    first.join();
    second.join();

    thread third=thread(thread3,cont1);
    third.join();

    return 0;
}
