#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#define LAUNCH_THREADX 7
#define LAUNCH_THREADY 1
#define LAUNCH_THREADZ 4
#define LAUNCH_BLOCKX 1
#define ORI_BLOCKX 1
#define LAUNCH_BLOCKY 1
#define ORI_BLOCKY 1
#define ORI_BLOCKZ 512
#define LAUNCH_BLOCKZ ORI_BLOCKZ * 5 // 5是额外部分，满足多层覆盖
#define SM_NUM 32
#define WORKER_NUM_PERSM 24
#define BLOCK_NUM LAUNCH_BLOCKZ * LAUNCH_BLOCKY * LAUNCH_BLOCKX
#define FLAG_LENGTH 65535
#define FLAG_BLOCK_BASE 0
#define FLAG_SM_BASE (FLAG_BLOCK_BASE + BLOCK_NUM)
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
// nvcc -arch=native smtest.cu -o smtest

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

__global__ void workload(int *flag) {
    int n1 = 15.6, n2 = 64.9, n3 = 134.7;
    int smid = get_smid();
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0) atomicAdd(flag + smid, 1);
    for (int i = 0; i < 50000; i++) {
        n1=sinf(n1);
        n2=n3/n2;
    }
    __syncthreads();
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


    dim3 Dim_block = dim3(LAUNCH_BLOCKX, LAUNCH_BLOCKY, LAUNCH_BLOCKZ);
    dim3 Dim_thread = dim3(LAUNCH_THREADX, LAUNCH_THREADY, LAUNCH_THREADZ);
    // launch kernel
    workload<<<Dim_block, Dim_thread, 0, streams[0]>>>(g_flag);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(flag, g_flag, sizeof(int) * FLAG_LENGTH, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 100; ++i) {
        printf("smid-num %d-%d\n", i, flag[i]);
    }

    
}