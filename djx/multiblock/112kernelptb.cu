#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>

#define LAUNCH_THREADX 56
#define LAUNCH_THREADY 1
#define LAUNCH_THREADZ 2

#define LAUNCH_BLOCKX 1
#define ORI_BLOCKX 1
#define LAUNCH_BLOCKY 1
#define ORI_BLOCKY 28
#define LAUNCH_BLOCKZ 512 * 5 // 5是额外部分，满足多层覆盖
#define ORI_BLOCKZ 32

#define SM_NUM 32
#define WORKER_NUM_PERSM 4

#define BLOCK_NUM LAUNCH_BLOCKZ * LAUNCH_BLOCKY * LAUNCH_BLOCKX
#define FLAG_LENGTH 65535
#define FLAG_BLOCK_BASE 0
#define FLAG_SM_BASE (FLAG_BLOCK_BASE + 1)
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
// nvcc -arch=native main.cu -o main

#define GPU_RETURN_STATUS(cmd) \
{ \
    CUresult result = cmd; \
    if (result != CUDA_SUCCESS) { \
        std::cout << #cmd " error, return code:" << result << " | " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    errorStr = cudaGetErrorString(err);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}

__device__ uint get_smid(void) {

    uint ret;

    asm("mov.u32 %0, %smid;" : "=r"(ret) );

    return ret;

}

// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
// #define __shfl_sync(mask, var, lane, width) \
//         __shfl((var), (lane), (width))

// #define __shfl_down_sync(mask, var, offset, width) \
//         __shfl_down((var), (offset), (width))

// #define __shfl_up_sync(mask, var, offset, width) \
//         __shfl_up((var), (offset), (width))
// #endif

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_6_kernel0(int *worker,int number,int *flag,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
    int* sm_flag = flag;
    __shared__ int basicoffset;
    int offset;
    int smid;
    //judge whether to continue work,which work to fetch
    if(threadIdx.x+threadIdx.y+threadIdx.z == 0)
    {
       basicoffset=-1;
       smid = get_smid();

       //judge whther sm id is right
       if((smid < number*SM_NUM)&&(smid >= (number-1)*SM_NUM))
       {
            //judge whether worker is enough
            //get the basic offset for the block
            int blocknumber=atomicAdd(sm_flag + smid, 1);
            if(blocknumber< WORKER_NUM_PERSM)
            {
                basicoffset = WORKER_NUM_PERSM*(smid-(number-1)*SM_NUM) + blocknumber;
                atomicAdd(worker + smid, 1);
                //printf("smid %d\n", smid);
            }
       }
    }
    __syncthreads();
    if (basicoffset < 0) return ;
    //every thread has its own offset
    offset = basicoffset;
    // if ((threadIdx.x + threadIdx.y + threadIdx.z) == 0 && (number == 1)) {
    //     printf("smid %d\n", smid);
    // }

    while(offset < (ORI_BLOCKX * ORI_BLOCKY * ORI_BLOCKZ)) {
        int vx = (offset)/(ORI_BLOCKY * ORI_BLOCKZ);
        int vy = (offset - (vx * ORI_BLOCKY * ORI_BLOCKZ)) / ORI_BLOCKZ;
        int vz = offset - (vx * ORI_BLOCKY * ORI_BLOCKZ) - vy * ORI_BLOCKZ;
        offset += SM_NUM * WORKER_NUM_PERSM;
    }
}

int main(int argc, char *argv[]) {
    // init device
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
        exit(1);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));

    // allocate stream
    int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }


    // allocate flag
    int *flag = new int[FLAG_LENGTH];
    int *g_flag;
    for (int i = 0; i < FLAG_LENGTH; ++i) {
        flag[i] = 0;
    }
    checkCudaErrors(cudaMalloc((void **)&g_flag, sizeof(int) * FLAG_LENGTH));
    checkCudaErrors(cudaMemcpy(g_flag, flag, sizeof(int) * FLAG_LENGTH, cudaMemcpyHostToDevice));

    int *flag_ = new int[FLAG_LENGTH];
    int *g_flag_;
    for (int i = 0; i < FLAG_LENGTH; ++i) {
        flag_[i] = 0;
    }
    checkCudaErrors(cudaMalloc((void **)&g_flag_, sizeof(int) * FLAG_LENGTH));
    checkCudaErrors(cudaMemcpy(g_flag_, flag_, sizeof(int) * FLAG_LENGTH, cudaMemcpyHostToDevice));

    //prepare parm for kernel 1
    int *workers = new int[80];
    for(int i=0;i<80;i++)
    {
    workers[i]=0;
    }
    int *g_worker;
    checkCudaErrors(cudaMalloc((void **)&g_worker, sizeof(int) * 80));
    checkCudaErrors(cudaMemcpy( g_worker,workers, sizeof(int) * 80, cudaMemcpyHostToDevice));


    float *placeholder0 = new float[6422528];
    for(int i=0;i<6422528;i++)
    {
    placeholder0[i]=1;
    }
    float *g_ph0;
    checkCudaErrors(cudaMalloc((void **)&g_ph0, sizeof(float) * 6422528));
    checkCudaErrors(cudaMemcpy(g_ph0, placeholder0, sizeof(float) * 6422528, cudaMemcpyHostToDevice));

    float *placeholder1 = new float[36864];
    for(int i=0;i<36864;i++)
    {
    placeholder1[i]=0;
    }
    float *g_ph1;
    checkCudaErrors(cudaMalloc((void **)&g_ph1, sizeof(float) * 36864));
    checkCudaErrors(cudaMemcpy(g_ph1, placeholder1, sizeof(float) * 36864, cudaMemcpyHostToDevice));

    float *placeholder2 = new float[6422528];
    for(int i=0;i<6422528;i++)
    {
    placeholder2[i]=3;
    }
    float *g_ph2;
    checkCudaErrors(cudaMalloc((void **)&g_ph2, sizeof(float) * 6422528));
    checkCudaErrors(cudaMemcpy(g_ph2, placeholder2, sizeof(float) * 6422528, cudaMemcpyHostToDevice));


    float *placeholder3 = new float[64];
    for(int i=0;i<64;i++)
    {
    placeholder3[i]=4;
    }
    float *g_ph3;
    cudaMalloc((void **)&g_ph3, sizeof(float) * 64);
    cudaMemcpy(g_ph3, placeholder3, sizeof(float) * 64, cudaMemcpyHostToDevice);


    dim3 Dim_block = dim3(LAUNCH_BLOCKX, LAUNCH_BLOCKY, LAUNCH_BLOCKZ);
    dim3 Dim_thread = dim3(LAUNCH_THREADX, LAUNCH_THREADY, LAUNCH_THREADZ);

    printf("hello?");
    // launch kernel
    fused_nn_conv2d_add_nn_relu_6_kernel0<<<Dim_block, Dim_thread, 0, streams[0]>>>(g_worker,1, g_flag, g_ph0, g_ph1, g_ph2, g_ph3);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(placeholder2, g_ph2,sizeof(float) * 6422528, cudaMemcpyDeviceToHost));

    printf("hello3?\n");
    for(int j=0;j<784;j++)
    {
    if(j%10==0)
    {
    printf("\n");
    }
    printf("%f  ",placeholder2[1024*j+j]);
    }

    printf("\n");
    checkCudaErrors(cudaMemcpy(workers,g_worker,sizeof(int) * 80, cudaMemcpyDeviceToHost));
    for(int j=0;j<80;j++)
    {
    if(j%10==0&&j!=0)
    {
    printf("\n");
    }
    printf("%d  ",workers[j]);
    }
    printf("\n");
}