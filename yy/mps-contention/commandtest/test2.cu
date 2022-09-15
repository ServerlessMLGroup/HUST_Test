#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <thread>
#include <random>
#include <ctime>
#include<cstdlib>
#include<string>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<stdio.h>
#include<unistd.h>
#include<sys/types.h>

using namespace std;

const int N = 300;

void Command0(void){

    std::string command = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40";
    putenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40");

    cout<<"Parent set sm 40%: "<<endl;
    int err=cudaSetDevice(0);
    int result = 0;
    if(err){
       cout<<"cudaSetDevice error:"<<err<<endl;
       return;
    }
    CUcontext pctx;
    CUdevice dev;
    err=cuCtxGetDevice(&dev);
    if(err){
       cout<<"cudaGetDevice error:"<<err<<endl;
       return;
    }
    err = cuCtxCreate(&pctx,CU_CTX_SCHED_YIELD,dev);
    if(err){
       cout<<"cudaGetDevice error:"<<err<<endl;
       return;
    }
    err = cudaDeviceGetAttribute(&result,cudaDevAttrMultiProcessorCount,0);
    if(err){
       cout<<"cudaDeviceGetAttribute error:"<<err<<endl;
       return;
    }
    cout<< "Parent : cudaDevAttrMultiProcessorCount is: "<<result<<endl;
}

void Command1(void){

    putenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20");

    cout<<"set sm 20%: "<<endl;
    int err=cudaSetDevice(0);
    int result = 0;
    if(err){
       cout<<"cudaSetDevice error:"<<err<<endl;
       return;
    }
    CUcontext pctx;
    CUdevice dev;
    err=cuCtxGetDevice(&dev);
    if(err){
       cout<<"cudaGetDevice error:"<<err<<endl;
       return;
    }
    err = cuCtxCreate(&pctx,CU_CTX_SCHED_YIELD,dev);
    if(err){
       cout<<"cudaGetDevice error:"<<err<<endl;
       return;
    }
    err = cudaDeviceGetAttribute(&result,cudaDevAttrMultiProcessorCount,0);
    if(err){
       cout<<"cudaDeviceGetAttribute error:"<<err<<endl;
       return;
    }
    cout<< "Child: cudaDevAttrMultiProcessorCount is: "<<result<<endl;
}

enum class Unit{
    Byte, KB, MB, GB, TB, PB, EB
};


double convert(double size, Unit unit)
{
    double result = size;
    switch (unit)
    {
    case Unit::EB:
        result /= 1024;     // flow through
    case Unit::PB:
        result /= 1024;     // flow through
    case Unit::TB:
        result /= 1024;     // flow through
    case Unit::GB:
        result /= 1024;     // flow through
    case Unit::MB:
        result /= 1024;     // flow through
    case Unit::KB:
        result /= 1024;     // flow through
    case Unit::Byte:
        result /= 1;
    default:
        break;
    }
    return result;
}

void getMem() {
    size_t free, total;
    int err=cudaMemGetInfo(&free, &total);
    if(err){
       cout<<"cudaMemGetInfo error:"<<err<<endl;
       return;
    }
    printf("Free mem = %.4f MB, Total = %.4f MB \n", convert(free, Unit::MB), convert(total, Unit::MB));
}

int main(void) {
    int num=3;
    pid_t pid = 0;
    pid = fork();           //创建一个子进程,fork()函数没有参数。
    printf("pid is %d\n",getpid());     //获取进程的pid
    if (0 < pid)        //父进程得到的pid大于0,这段代码是父进程中执行的
    {
        Command0();
        num++;
        printf("I am parent!,num is %d\n",num);
    }
    else if(0 == pid)   //子进程得到的返回值是0，这段代码在子进程中执行
    {
        Command1();
        num--;
        printf("I am son!,num is %d\n",num);
    }
   else                 //创建进程失败
   {
       //有两种情况会失败：
       //1.进程数目达到OS的最大值
       //2.进程创建时内存不够了。
       printf("fork error!\n");
       exit(-1);
   }
   return 0;
}
