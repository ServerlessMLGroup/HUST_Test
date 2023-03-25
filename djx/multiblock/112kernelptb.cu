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
#define WORKER_NUM_PERSM 1

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
        float compute[64];
  __shared__ float pad_temp_shared[448];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(28)] = 0.000000e+00f;
  compute[(30)] = 0.000000e+00f;
  compute[(32)] = 0.000000e+00f;
  compute[(34)] = 0.000000e+00f;
  compute[(36)] = 0.000000e+00f;
  compute[(38)] = 0.000000e+00f;
  compute[(40)] = 0.000000e+00f;
  compute[(42)] = 0.000000e+00f;
  compute[(44)] = 0.000000e+00f;
  compute[(46)] = 0.000000e+00f;
  compute[(48)] = 0.000000e+00f;
  compute[(50)] = 0.000000e+00f;
  compute[(52)] = 0.000000e+00f;
  compute[(54)] = 0.000000e+00f;
  compute[(56)] = 0.000000e+00f;
  compute[(58)] = 0.000000e+00f;
  compute[(60)] = 0.000000e+00f;
  compute[(62)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  compute[(29)] = 0.000000e+00f;
  compute[(31)] = 0.000000e+00f;
  compute[(33)] = 0.000000e+00f;
  compute[(35)] = 0.000000e+00f;
  compute[(37)] = 0.000000e+00f;
  compute[(39)] = 0.000000e+00f;
  compute[(41)] = 0.000000e+00f;
  compute[(43)] = 0.000000e+00f;
  compute[(45)] = 0.000000e+00f;
  compute[(47)] = 0.000000e+00f;
  compute[(49)] = 0.000000e+00f;
  compute[(51)] = 0.000000e+00f;
  compute[(53)] = 0.000000e+00f;
  compute[(55)] = 0.000000e+00f;
  compute[(57)] = 0.000000e+00f;
  compute[(59)] = 0.000000e+00f;
  compute[(61)] = 0.000000e+00f;
  compute[(63)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)))] = ((((1 <= (((((int)vy) * 2) + ((((int)threadIdx.x) % 28) / 14)) + ry_outer)) && ((((((int)vy) * 2) + ((((int)threadIdx.x) % 28) / 14)) + ry_outer) < 57)) && (1 <= (((int)threadIdx.x) % 14))) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((int)threadIdx.x) / 28) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + ((((int)threadIdx.x) % 28) * 4)) - 57))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)) + 1))] = ((((1 <= (((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 1) % 112) / 56)) + ry_outer)) && ((((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 1) % 112) / 56)) + ry_outer) < 57)) && (1 <= (((((int)threadIdx.x) * 4) + 1) % 56))) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + 1) / 112) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + (((((int)threadIdx.x) * 4) + 1) % 112)) - 57))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)) + 2))] = ((((1 <= (((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 2) % 112) / 56)) + ry_outer)) && ((((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 2) % 112) / 56)) + ry_outer) < 57)) && (1 <= (((((int)threadIdx.x) * 4) + 2) % 56))) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + 2) / 112) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + (((((int)threadIdx.x) * 4) + 2) % 112)) - 57))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)) + 3))] = ((((1 <= (((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 3) % 112) / 56)) + ry_outer)) && ((((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 3) % 112) / 56)) + ry_outer) < 57)) && (1 <= (((((int)threadIdx.x) * 4) + 3) % 56))) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + 3) / 112) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + (((((int)threadIdx.x) * 4) + 3) % 112)) - 57))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 32) + ((((int)threadIdx.x) * 3) >> 2)) < 64) {
        if (((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) < 256) {
          if (((int)threadIdx.x) < 43) {
            placeholder_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)))] = placeholder1[((((((((int)threadIdx.z) * 18432) + (((((int)threadIdx.x) * 3) >> 2) * 576)) + (rc_outer * 36)) + (((((int)threadIdx.x) * 3) & 3) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 3) + 1) >> 2)) < 64) {
        if (((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) < 255) {
          if (((int)threadIdx.x) < 43) {
            placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[((((((((int)threadIdx.z) * 18432) + ((((((int)threadIdx.x) * 3) + 1) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx.x) * 3) + 1) & 3) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 3) + 2) >> 2)) < 64) {
        if (((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) < 254) {
          if (((int)threadIdx.x) < 42) {
            placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[((((((((int)threadIdx.z) * 18432) + ((((((int)threadIdx.x) * 3) + 2) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx.x) * 3) + 2) & 3) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 8))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 16))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 24))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 32))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 40))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 48))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 56))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 64))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 72))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 80))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 88))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 96))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 104))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 112))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 120))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 136))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 144))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 152))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 160))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 168))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 176))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 184))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 192))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 200))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 208))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 216))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 224))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 232))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 240))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 248))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 8))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 16))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 24))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 32))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 40))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 48))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 56))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 64))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 72))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 80))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 88))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 96))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 104))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 112))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 120))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 136))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 144))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 152))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 160))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 168))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 176))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 184))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 192))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 200))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 208))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 216))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 224))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 232))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 240))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 248))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 9))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 17))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 25))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 33))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 41))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 49))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 57))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 65))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 73))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 81))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 89))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 97))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 105))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 113))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 121))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 137))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 145))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 153))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 161))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 169))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 177))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 185))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 193))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 201))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 209))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 217))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 225))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 233))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 241))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 249))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 9))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 17))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 25))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 33))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 41))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 49))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 57))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 65))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 73))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 81))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 89))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 97))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 105))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 113))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 121))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 137))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 145))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 153))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 161))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 169))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 177))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 185))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 193))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 201))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 209))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 217))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 225))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 233))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 241))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 249))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 10))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 18))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 26))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 34))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 42))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 50))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 58))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 66))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 74))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 82))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 90))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 98))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 106))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 114))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 122))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 138))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 146))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 154))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 162))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 170))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 178))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 186))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 194))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 202))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 210))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 218))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 226))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 234))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 242))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 250))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 10))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 18))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 26))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 34))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 42))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 50))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 58))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 66))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 74))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 82))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 90))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 98))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 106))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 114))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 122))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 138))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 146))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 154))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 162))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 170))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 178))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 186))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 194))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 202))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 210))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 218))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 226))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 234))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 242))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 250))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 11))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 19))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 27))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 35))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 43))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 51))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 59))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 67))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 75))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 83))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 91))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 99))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 107))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 115))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 123))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 139))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 147))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 155))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 163))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 171))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 179))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 187))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 195))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 203))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 211))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 219))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 227))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 235))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 243))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 251))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 11))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 19))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 27))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 35))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 43))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 51))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 59))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 67))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 75))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 83))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 91))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 99))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 107))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 115))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 123))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 139))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 147))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 155))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 163))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 171))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 179))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 187))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 195))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 203))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 211))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 219))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 227))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 235))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 243))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 251))]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)))] = (((1 <= (((((int)vy) * 2) + ((((int)threadIdx.x) % 28) / 14)) + ry_outer)) && ((((((int)vy) * 2) + ((((int)threadIdx.x) % 28) / 14)) + ry_outer) < 57)) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((int)threadIdx.x) / 28) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + ((((int)threadIdx.x) % 28) * 4)) - 56))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)) + 1))] = (((1 <= (((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 1) % 112) / 56)) + ry_outer)) && ((((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 1) % 112) / 56)) + ry_outer) < 57)) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + 1) / 112) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + (((((int)threadIdx.x) * 4) + 1) % 112)) - 56))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)) + 2))] = (((1 <= (((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 2) % 112) / 56)) + ry_outer)) && ((((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 2) % 112) / 56)) + ry_outer) < 57)) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + 2) / 112) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + (((((int)threadIdx.x) * 4) + 2) % 112)) - 56))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)) + 3))] = (((1 <= (((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 3) % 112) / 56)) + ry_outer)) && ((((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 3) % 112) / 56)) + ry_outer) < 57)) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + 3) / 112) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + (((((int)threadIdx.x) * 4) + 3) % 112)) - 56))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 32) + ((((int)threadIdx.x) * 3) >> 2)) < 64) {
        if (((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) < 256) {
          if (((int)threadIdx.x) < 43) {
            placeholder_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)))] = placeholder1[(((((((((int)threadIdx.z) * 18432) + (((((int)threadIdx.x) * 3) >> 2) * 576)) + (rc_outer * 36)) + (((((int)threadIdx.x) * 3) & 3) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 3) + 1) >> 2)) < 64) {
        if (((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) < 255) {
          if (((int)threadIdx.x) < 43) {
            placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[(((((((((int)threadIdx.z) * 18432) + ((((((int)threadIdx.x) * 3) + 1) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx.x) * 3) + 1) & 3) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 3) + 2) >> 2)) < 64) {
        if (((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) < 254) {
          if (((int)threadIdx.x) < 42) {
            placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[(((((((((int)threadIdx.z) * 18432) + ((((((int)threadIdx.x) * 3) + 2) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx.x) * 3) + 2) & 3) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 8))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 16))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 24))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 32))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 40))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 48))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 56))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 64))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 72))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 80))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 88))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 96))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 104))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 112))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 120))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 136))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 144))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 152))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 160))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 168))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 176))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 184))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 192))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 200))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 208))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 216))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 224))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 232))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 240))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 248))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 8))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 16))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 24))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 32))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 40))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 48))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 56))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 64))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 72))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 80))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 88))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 96))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 104))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 112))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 120))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 136))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 144))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 152))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 160))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 168))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 176))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 184))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 192))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 200))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 208))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 216))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 224))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 232))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 240))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 248))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 9))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 17))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 25))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 33))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 41))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 49))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 57))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 65))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 73))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 81))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 89))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 97))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 105))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 113))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 121))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 137))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 145))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 153))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 161))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 169))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 177))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 185))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 193))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 201))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 209))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 217))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 225))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 233))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 241))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 249))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 9))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 17))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 25))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 33))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 41))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 49))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 57))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 65))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 73))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 81))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 89))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 97))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 105))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 113))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 121))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 137))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 145))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 153))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 161))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 169))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 177))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 185))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 193))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 201))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 209))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 217))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 225))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 233))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 241))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 249))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 10))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 18))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 26))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 34))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 42))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 50))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 58))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 66))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 74))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 82))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 90))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 98))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 106))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 114))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 122))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 138))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 146))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 154))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 162))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 170))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 178))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 186))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 194))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 202))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 210))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 218))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 226))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 234))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 242))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 250))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 10))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 18))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 26))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 34))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 42))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 50))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 58))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 66))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 74))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 82))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 90))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 98))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 106))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 114))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 122))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 138))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 146))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 154))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 162))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 170))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 178))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 186))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 194))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 202))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 210))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 218))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 226))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 234))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 242))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 250))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 11))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 19))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 27))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 35))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 43))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 51))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 59))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 67))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 75))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 83))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 91))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 99))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 107))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 115))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 123))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 139))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 147))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 155))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 163))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 171))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 179))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 187))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 195))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 203))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 211))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 219))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 227))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 235))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 243))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 251))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 11))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 19))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 27))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 35))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 43))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 51))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 59))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 67))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 75))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 83))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 91))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 99))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 107))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 115))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 123))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 139))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 147))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 155))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 163))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 171))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 179))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 187))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 195))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 203))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 211))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 219))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 227))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 235))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 243))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 251))]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)))] = (((1 <= (((((int)vy) * 2) + ((((int)threadIdx.x) % 28) / 14)) + ry_outer)) && ((((((int)vy) * 2) + ((((int)threadIdx.x) % 28) / 14)) + ry_outer) < 57)) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((int)threadIdx.x) / 28) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + ((((int)threadIdx.x) % 28) * 4)) - 55))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)) + 1))] = ((((1 <= (((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 1) % 112) / 56)) + ry_outer)) && ((((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 1) % 112) / 56)) + ry_outer) < 57)) && ((((((int)threadIdx.x) * 4) + 1) % 56) < 55)) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + 1) / 112) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + (((((int)threadIdx.x) * 4) + 1) % 112)) - 55))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)) + 2))] = ((((1 <= (((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 2) % 112) / 56)) + ry_outer)) && ((((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 2) % 112) / 56)) + ry_outer) < 57)) && ((((((int)threadIdx.x) * 4) + 2) % 56) < 55)) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + 2) / 112) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + (((((int)threadIdx.x) * 4) + 2) % 112)) - 55))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 224) + (((int)threadIdx.x) * 4)) + 3))] = ((((1 <= (((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 3) % 112) / 56)) + ry_outer)) && ((((((int)vy) * 2) + ((((((int)threadIdx.x) * 4) + 3) % 112) / 56)) + ry_outer) < 57)) && ((((((int)threadIdx.x) * 4) + 3) % 56) < 55)) ? placeholder[(((((((((((int)vz) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 4) + 3) / 112) * 3136)) + (((int)vy) * 112)) + (ry_outer * 56)) + (((((int)threadIdx.x) * 4) + 3) % 112)) - 55))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 32) + ((((int)threadIdx.x) * 3) >> 2)) < 64) {
        if (((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) < 256) {
          if (((int)threadIdx.x) < 43) {
            placeholder_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)))] = placeholder1[(((((((((int)threadIdx.z) * 18432) + (((((int)threadIdx.x) * 3) >> 2) * 576)) + (rc_outer * 36)) + (((((int)threadIdx.x) * 3) & 3) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 3) + 1) >> 2)) < 64) {
        if (((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) < 255) {
          if (((int)threadIdx.x) < 43) {
            placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) + 1))] = placeholder1[(((((((((int)threadIdx.z) * 18432) + ((((((int)threadIdx.x) * 3) + 1) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx.x) * 3) + 1) & 3) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 3) + 2) >> 2)) < 64) {
        if (((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) < 254) {
          if (((int)threadIdx.x) < 42) {
            placeholder_shared[((((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 3)) + 2))] = placeholder1[(((((((((int)threadIdx.z) * 18432) + ((((((int)threadIdx.x) * 3) + 2) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx.x) * 3) + 2) & 3) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 8))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 16))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 24))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 32))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 40))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 48))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 56))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 64))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 72))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 80))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 88))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 96))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 104))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 112))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 120))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 136))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 144))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 152))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 160))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 168))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 176))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 184))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 192))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 200))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 208))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 216))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 224))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 232))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 240))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 248))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 4))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 8))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 16))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 24))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 32))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 40))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 48))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 56))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 64))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 72))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 80))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 88))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 96))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 104))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 112))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 120))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 128))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 136))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 144))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 152))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 160))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 168))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 176))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 184))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 192))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 200))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 208))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 216))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 224))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 232))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 240))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 248))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 9))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 17))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 25))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 33))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 41))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 49))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 57))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 65))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 73))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 81))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 89))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 97))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 105))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 113))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 121))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 137))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 145))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 153))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 161))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 169))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 177))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 185))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 193))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 201))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 209))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 217))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 225))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 233))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 241))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 249))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 1))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 9))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 17))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 25))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 33))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 41))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 49))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 57))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 65))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 73))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 81))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 89))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 97))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 105))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 113))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 121))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 129))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 137))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 145))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 153))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 161))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 169))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 177))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 185))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 193))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 201))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 209))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 217))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 225))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 233))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 241))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 249))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 10))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 18))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 26))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 34))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 42))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 50))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 58))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 66))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 74))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 82))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 90))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 98))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 106))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 114))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 122))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 138))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 146))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 154))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 162))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 170))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 178))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 186))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 194))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 202))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 210))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 218))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 226))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 234))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 242))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 250))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 2))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 10))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 18))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 26))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 34))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 42))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 50))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 58))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 66))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 74))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 82))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 90))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 98))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 106))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 114))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 122))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 130))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 138))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 146))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 154))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 162))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 170))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 178))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 186))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 194))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 202))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 210))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 218))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 226))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 234))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 242))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 250))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 11))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 19))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 27))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 35))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 43))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 51))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 59))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 67))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 75))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 83))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 91))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 99))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 107))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 115))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 123))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 139))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 147))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 155))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 163))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 171))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 179))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 187))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 195))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 203))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 211))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 219))]));
      compute[(56)] = (compute[(56)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 227))]));
      compute[(58)] = (compute[(58)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 235))]));
      compute[(60)] = (compute[(60)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 243))]));
      compute[(62)] = (compute[(62)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 251))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 3))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 11))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 19))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 27))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 35))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 43))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 51))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 59))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 67))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 75))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 83))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 91))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 99))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 107))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 115))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 123))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 131))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 139))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 147))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 155))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 163))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 171))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 179))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 187))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 195))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 203))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 211))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 219))]));
      compute[(57)] = (compute[(57)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 227))]));
      compute[(59)] = (compute[(59)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 235))]));
      compute[(61)] = (compute[(61)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 243))]));
      compute[(63)] = (compute[(63)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 4) + 251))]));
    }
  }
  T_relu[(((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 6272))] = max((compute[(2)] + placeholder2[((((int)threadIdx.z) + 2))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 12544))] = max((compute[(4)] + placeholder2[((((int)threadIdx.z) + 4))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 18816))] = max((compute[(6)] + placeholder2[((((int)threadIdx.z) + 6))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 25088))] = max((compute[(8)] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 31360))] = max((compute[(10)] + placeholder2[((((int)threadIdx.z) + 10))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 37632))] = max((compute[(12)] + placeholder2[((((int)threadIdx.z) + 12))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 43904))] = max((compute[(14)] + placeholder2[((((int)threadIdx.z) + 14))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 50176))] = max((compute[(16)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 56448))] = max((compute[(18)] + placeholder2[((((int)threadIdx.z) + 18))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 62720))] = max((compute[(20)] + placeholder2[((((int)threadIdx.z) + 20))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 68992))] = max((compute[(22)] + placeholder2[((((int)threadIdx.z) + 22))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 75264))] = max((compute[(24)] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 81536))] = max((compute[(26)] + placeholder2[((((int)threadIdx.z) + 26))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 87808))] = max((compute[(28)] + placeholder2[((((int)threadIdx.z) + 28))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 94080))] = max((compute[(30)] + placeholder2[((((int)threadIdx.z) + 30))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 100352))] = max((compute[(32)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 106624))] = max((compute[(34)] + placeholder2[((((int)threadIdx.z) + 34))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 112896))] = max((compute[(36)] + placeholder2[((((int)threadIdx.z) + 36))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 119168))] = max((compute[(38)] + placeholder2[((((int)threadIdx.z) + 38))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 125440))] = max((compute[(40)] + placeholder2[((((int)threadIdx.z) + 40))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 131712))] = max((compute[(42)] + placeholder2[((((int)threadIdx.z) + 42))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 137984))] = max((compute[(44)] + placeholder2[((((int)threadIdx.z) + 44))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 144256))] = max((compute[(46)] + placeholder2[((((int)threadIdx.z) + 46))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 150528))] = max((compute[(48)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 156800))] = max((compute[(50)] + placeholder2[((((int)threadIdx.z) + 50))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 163072))] = max((compute[(52)] + placeholder2[((((int)threadIdx.z) + 52))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 169344))] = max((compute[(54)] + placeholder2[((((int)threadIdx.z) + 54))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 175616))] = max((compute[(56)] + placeholder2[((((int)threadIdx.z) + 56))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 181888))] = max((compute[(58)] + placeholder2[((((int)threadIdx.z) + 58))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 188160))] = max((compute[(60)] + placeholder2[((((int)threadIdx.z) + 60))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 194432))] = max((compute[(62)] + placeholder2[((((int)threadIdx.z) + 62))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 56))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 6328))] = max((compute[(3)] + placeholder2[((((int)threadIdx.z) + 2))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 12600))] = max((compute[(5)] + placeholder2[((((int)threadIdx.z) + 4))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 18872))] = max((compute[(7)] + placeholder2[((((int)threadIdx.z) + 6))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 25144))] = max((compute[(9)] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 31416))] = max((compute[(11)] + placeholder2[((((int)threadIdx.z) + 10))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 37688))] = max((compute[(13)] + placeholder2[((((int)threadIdx.z) + 12))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 43960))] = max((compute[(15)] + placeholder2[((((int)threadIdx.z) + 14))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 50232))] = max((compute[(17)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 56504))] = max((compute[(19)] + placeholder2[((((int)threadIdx.z) + 18))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 62776))] = max((compute[(21)] + placeholder2[((((int)threadIdx.z) + 20))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 69048))] = max((compute[(23)] + placeholder2[((((int)threadIdx.z) + 22))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 75320))] = max((compute[(25)] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 81592))] = max((compute[(27)] + placeholder2[((((int)threadIdx.z) + 26))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 87864))] = max((compute[(29)] + placeholder2[((((int)threadIdx.z) + 28))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 94136))] = max((compute[(31)] + placeholder2[((((int)threadIdx.z) + 30))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 100408))] = max((compute[(33)] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 106680))] = max((compute[(35)] + placeholder2[((((int)threadIdx.z) + 34))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 112952))] = max((compute[(37)] + placeholder2[((((int)threadIdx.z) + 36))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 119224))] = max((compute[(39)] + placeholder2[((((int)threadIdx.z) + 38))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 125496))] = max((compute[(41)] + placeholder2[((((int)threadIdx.z) + 40))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 131768))] = max((compute[(43)] + placeholder2[((((int)threadIdx.z) + 42))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 138040))] = max((compute[(45)] + placeholder2[((((int)threadIdx.z) + 44))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 144312))] = max((compute[(47)] + placeholder2[((((int)threadIdx.z) + 46))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 150584))] = max((compute[(49)] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 156856))] = max((compute[(51)] + placeholder2[((((int)threadIdx.z) + 50))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 163128))] = max((compute[(53)] + placeholder2[((((int)threadIdx.z) + 52))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 169400))] = max((compute[(55)] + placeholder2[((((int)threadIdx.z) + 54))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 175672))] = max((compute[(57)] + placeholder2[((((int)threadIdx.z) + 56))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 181944))] = max((compute[(59)] + placeholder2[((((int)threadIdx.z) + 58))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 188216))] = max((compute[(61)] + placeholder2[((((int)threadIdx.z) + 60))]), 0.000000e+00f);
  T_relu[((((((((int)vz) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)vy) * 112)) + ((int)threadIdx.x)) + 194488))] = max((compute[(63)] + placeholder2[((((int)threadIdx.z) + 62))]), 0.000000e+00f);
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