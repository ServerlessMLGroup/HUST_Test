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
   /*
   cudaSetDevice(2);
   CUcontext tempcont;
   CUdevice dev;
   int err;
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
   */

   int err;
   CUmodule mod;
   cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx");
   sleep(1);


   err=cuCtxPushCurrent(ctx);
   if(err){
   std::cout<<"Push Context ERR! "<<err<<std::endl;
   }

   CUmodule mod1,mod2,mod3,mod4,mod5,mod6;
   cuModuleLoad(&mod1, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx");
   cuModuleLoad(&mod2, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx");
   cuModuleLoad(&mod3, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx");
   cuModuleLoad(&mod4, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx");
   cuModuleLoad(&mod5, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx");
   cuModuleLoad(&mod6, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx");
}
// add fininshed


int main()
{
    cuInit(0);
    cudaSetDevice(2);
    //clock for collection

    //yy change
    CUcontext cont1;
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

    thread first=thread(thread1,cont1);
    //thread second=thread(thread1,cont1);
    first.join();
    //second.join();

    return 0;
}
