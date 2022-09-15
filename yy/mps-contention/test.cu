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
using namespace std;

const int N = 300;

void Command0(void){
    char line[N];
    FILE *fp;
    string cmd = "echo set_active_thread_percentage 359 10 | nvidia-cuda-mps-control";
    引号内是你的linux指令
    // 系统调用
    const char *sysCommand = cmd.data();
    if ((fp = popen(sysCommand, "r")) == NULL) {
        cout << "error" << endl;
        return;
    }
    while (fgets(line, sizeof(line)-1, fp) != NULL){
        cout << line ;
    }
    pclose(fp);
}

void Command1(void){
    char line[N];
    FILE *fp;
    string cmd = "echo set_active_thread_percentage 359 20 | nvidia-cuda-mps-control";
    引号内是你的linux指令
    // 系统调用
    const char *sysCommand = cmd.data();
    if ((fp = popen(sysCommand, "r")) == NULL) {
        cout << "error" << endl;
        return;
    }
    while (fgets(line, sizeof(line)-1, fp) != NULL){
        cout << line ;
    }
    pclose(fp);
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
    Command0();
    cout<<"set sm 10%: "<<endl;
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
    cout<< "cudaDevAttrMultiProcessorCount is: "<<result<<endl;

    Command1();
    cout<<"set sm 20%: "<<endl;
    CUcontext pctx2;
    err = cuCtxCreate(&pctx2,CU_CTX_SCHED_YIELD,dev);
    if(err){
       cout<<"cudaGetDevice error:"<<err<<endl;
       return;
    }
    err = cudaDeviceGetAttribute(&result,cudaDevAttrMultiProcessorCount,0);
    if(err){
       cout<<"cudaDeviceGetAttribute error:"<<err<<endl;
       return;
    }
    cout<< "cudaDevAttrMultiProcessorCount is: "<<result<<endl;

}
