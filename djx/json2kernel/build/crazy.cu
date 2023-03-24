#include "log.h"
#include <bits/unique_ptr.h>
#include <cuda.h>
#include "cuda_runtime.h"
//yy add
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <math.h>
#include "unistd.h"
#include <thread>
#include <sys/time.h>
#include <unistd.h>

// #include <glog/logging.h>
//Notice
// To make some experiments, i(yy) make some changes here. Before changing, i copied all the code
// Just read the code at copymain.cpp. If some bad change were made, we can fix it by the copy
enum Status {
    Succ,
    Fail,
    NotFound,
    OutOfRange,
    Full
};
#define GPU_RETURN_STATUS(cmd) \
{ \
    CUresult result = cmd; \
    if (result != CUDA_SUCCESS) { \
        std::cout << #cmd " error, return code:" << result << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}
#define RETURN_STATUS(cmd) \
{\
    Status s = cmd;\
    if (s != Status::Succ) {\
        std::cout << #cmd " error, " << __FILE__ << ":" << __LINE__ << std::endl; \
        return s;\
    }\
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
    }
    int gpu_no = atoi(argv[1]);


    CUmodule mod;
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx"));
    printf("load cuda kernels!\n");
    //yy add stream
    CUstream iofirststream;
    cuStreamCreate(&iofirststream,0);
    CUstream iosecondstream;
    cuStreamCreate(&iosecondstream,0);
    CUstream kefirststream;
    cuStreamCreate(&kefirststream,0);
    CUstream kesecondstream;
    cuStreamCreate(&kesecondstream,0);

    CUfunction kernel;
    GPU_RETURN_STATUS(
            cuModuleGetFunction(&kernel, mod, "fused_nn_conv2d_add_multiply_add_nn_relu_kernel0")
        );

    CUdeviceptr deviceptr0;
    CUdeviceptr deviceptr1;
    CUdeviceptr deviceptr2;
    CUdeviceptr deviceptr3;
    CUdeviceptr deviceptr4;
    CUdeviceptr deviceptr5;
    //check answer
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&deviceptr0, sizeof(float)*802816));
    float *placeholder0 = new float[802816];
    for(int i=0;i<802816;i++)
    {
    placeholder0[i]=1;
    }
    GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)deviceptr0,placeholder0, sizeof(float)*802816,iofirststream));

    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&deviceptr1, sizeof(float)*2359296));
    float *placeholder1 = new float[2359296];
    for(int i=0;i<2359296;i++)
    {
    placeholder1[i]=2;
    }
    GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)deviceptr1,placeholder1, sizeof(float)*2359296,iofirststream));

    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&deviceptr2, sizeof(float)*802816));
    float *placeholder2 = new float[802816];
    for(int i=0;i<802816;i++)
    {
    placeholder2[i]=3;
    }
    GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)deviceptr2,placeholder2, sizeof(float)*802816,iofirststream));

    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&deviceptr3, sizeof(float)*802816));
    float *placeholder3 = new float[802816];
    for(int i=0;i<802816;i++)
    {
    placeholder3[i]=4;
    }
    GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)deviceptr3,placeholder3, sizeof(float)*802816,iofirststream));

    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&deviceptr4, sizeof(float)*512));
    float *placeholder4 = new float[512];
    for(int i=0;i<512;i++)
    {
    placeholder4[i]=5;
    }
    GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)deviceptr4,placeholder4, sizeof(float)*512,iofirststream));

    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&deviceptr5, sizeof(float)*512));
    float *placeholder5 = new float[512];
    for(int i=0;i<512;i++)
    {
    placeholder5[i]=6;
    }
    GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)deviceptr5,placeholder5, sizeof(float)*512,iofirststream));

    std::vector<CUdeviceptr*> extrarg;
    extrarg.push_back(&deviceptr0);
    extrarg.push_back(&deviceptr1);
    extrarg.push_back(&deviceptr2);
    extrarg.push_back(&deviceptr3);
    extrarg.push_back(&deviceptr4);
    extrarg.push_back(&deviceptr5);

    GPU_RETURN_STATUS(cuLaunchKernel(kernel,
        1, 1, 512,
        7, 1, 4,
        0, kefirststream, (void **)extrarg.data(), 0 // raw_args1是json中指示的storage的下标
    ));

    cuStreamSynchronize(kefirststream);

    for(int j=0;j<784;j++)
    {
    if(j%10==0)
    {
    std::cout<<std::endl;
    }
    std::cout<<placeholder2[1024*j+j]<<" ";
    }

    printf("reset model!\n");
    model.reset();
    return 0;
}