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
    putenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40");

    cout<<"set sm 40%: "<<endl;
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
       cout<<"First:cudaDeviceGetAttribute error:"<<err<<endl;
       return;
    }
    cout<< ": cudaDevAttrMultiProcessorCount is: "<<result<<endl;

    cout<<"set sm 20%: "<<endl;
    putenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20");
    err = cuCtxCreate(&pctx,CU_CTX_SCHED_YIELD,dev);
    if(err){
       cout<<"cudaGetDevice error:"<<err<<endl;
       return;
    }
    err = cudaDeviceGetAttribute(&result,cudaDevAttrMultiProcessorCount,0);
    if(err){
       cout<<"First:cudaDeviceGetAttribute error:"<<err<<endl;
       return;
    }
    cout<< ": cudaDevAttrMultiProcessorCount is: "<<result<<endl;

    return 0;
}
