#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>


#define LAUNCH_THREADX 7
#define LAUNCH_THREADY 1
#define LAUNCH_THREADZ 8

#define LAUNCH_BLOCKX 1
#define ORI_BLOCKX 1
#define LAUNCH_BLOCKY 1
#define ORI_BLOCKY 1
#define LAUNCH_BLOCKZ 512 * 5 // 5是额外部分，满足多层覆盖
#define ORI_BLOCKZ 256

#define SM_NUM 32
#define WORKER_NUM_PERSM 8

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

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_1_kernel0(int *worker,int number,int *flag, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2){
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
        float compute[56];
  __shared__ float pad_temp_shared[1352];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(28)] = 0.000000e+00f;
  compute[(42)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(30)] = 0.000000e+00f;
  compute[(44)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(32)] = 0.000000e+00f;
  compute[(46)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(34)] = 0.000000e+00f;
  compute[(48)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(36)] = 0.000000e+00f;
  compute[(50)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(38)] = 0.000000e+00f;
  compute[(52)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(40)] = 0.000000e+00f;
  compute[(54)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(29)] = 0.000000e+00f;
  compute[(43)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(31)] = 0.000000e+00f;
  compute[(45)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(33)] = 0.000000e+00f;
  compute[(47)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(35)] = 0.000000e+00f;
  compute[(49)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(37)] = 0.000000e+00f;
  compute[(51)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(39)] = 0.000000e+00f;
  compute[(53)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  compute[(41)] = 0.000000e+00f;
  compute[(55)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)))] = (((1 <= (((((int)threadIdx.x) * 25) / 13) + ry_outer)) && (1 <= ((((int)threadIdx.x) * 25) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((((int)threadIdx.x) * 25) / 13) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 25) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 1))] = (((1 <= ((((((int)threadIdx.x) * 25) + 1) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 1) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 1) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 1) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 2))] = (((1 <= ((((((int)threadIdx.x) * 25) + 2) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 2) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 2) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 2) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 3))] = (((1 <= ((((((int)threadIdx.x) * 25) + 3) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 3) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 3) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 3) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 4))] = (((1 <= ((((((int)threadIdx.x) * 25) + 4) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 4) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 4) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 4) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 5))] = (((1 <= ((((((int)threadIdx.x) * 25) + 5) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 5) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 5) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 5) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 6))] = (((1 <= ((((((int)threadIdx.x) * 25) + 6) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 6) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 6) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 6) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 7))] = (((1 <= ((((((int)threadIdx.x) * 25) + 7) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 7) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 7) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 7) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 8))] = (((1 <= ((((((int)threadIdx.x) * 25) + 8) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 8) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 8) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 8) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 9))] = (((1 <= ((((((int)threadIdx.x) * 25) + 9) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 9) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 9) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 9) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 10))] = (((1 <= ((((((int)threadIdx.x) * 25) + 10) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 10) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 10) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 10) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 11))] = (((1 <= ((((((int)threadIdx.x) * 25) + 11) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 11) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 11) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 11) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 12))] = (((1 <= ((((((int)threadIdx.x) * 25) + 12) / 13) + ry_outer)) && (1 <= (((((int)threadIdx.x) * 25) + 12) % 13))) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 12) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 12) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 13))] = ((1 <= ((((int)threadIdx.x) * 25) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((((int)threadIdx.x) * 25) / 13) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 25) % 13)) - 1))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 14))] = ((1 <= (((((int)threadIdx.x) * 25) + 1) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 14) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 1) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 15))] = ((1 <= (((((int)threadIdx.x) * 25) + 2) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 15) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 2) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 16))] = ((1 <= (((((int)threadIdx.x) * 25) + 3) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 16) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 3) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 17))] = ((1 <= (((((int)threadIdx.x) * 25) + 4) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 17) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 4) % 13)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 18))] = ((1 <= (((((int)threadIdx.x) * 25) + 5) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 18) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 5) % 13)) - 15))] : 0.000000e+00f);
      if (((((((int)threadIdx.x) * 25) + 19) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 19) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1333) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 19))] = ((1 <= (((((int)threadIdx.x) * 25) + 6) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 19) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 6) % 13)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 20) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 20) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1332) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 20))] = ((1 <= (((((int)threadIdx.x) * 25) + 7) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 20) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 7) % 13)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 21) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 21) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1331) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 21))] = ((1 <= (((((int)threadIdx.x) * 25) + 8) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 21) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 8) % 13)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 22) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 22) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1330) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 22))] = ((1 <= (((((int)threadIdx.x) * 25) + 9) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 22) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 9) % 13)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 23) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 23) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1329) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 23))] = ((1 <= (((((int)threadIdx.x) * 25) + 10) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 23) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 10) % 13)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 24) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 24) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1328) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 24))] = ((1 <= (((((int)threadIdx.x) * 25) + 11) % 13)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 24) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 11) % 13)) - 15))] : 0.000000e+00f);
            }
          }
        }
      }
      placeholder_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)))] = placeholder1[((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 10) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 10) & 7) * 9)) + (ry_outer * 3)))];
      placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 1))] = placeholder1[((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 1) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 1) & 7) * 9)) + (ry_outer * 3)))];
      placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 2))] = placeholder1[((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 2) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 2) & 7) * 9)) + (ry_outer * 3)))];
      placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 3))] = placeholder1[((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 3) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 3) & 7) * 9)) + (ry_outer * 3)))];
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 4) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 508) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 4))] = placeholder1[((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 4) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 4) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 5) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 507) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 5))] = placeholder1[((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 5) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 5) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 6) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 506) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 6))] = placeholder1[((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 6) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 6) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 7) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 505) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 7))] = placeholder1[((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 7) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 7) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 10) >> 3)) < 63) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 504) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 8))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 10) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 10) & 7) * 9)) + (ry_outer * 3)) + 2304))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 9) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 503) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 9))] = placeholder1[((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 9) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 1) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)))] = ((1 <= (((((int)threadIdx.x) * 25) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((((int)threadIdx.x) * 25) / 13) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 25) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 1))] = ((1 <= ((((((int)threadIdx.x) * 25) + 1) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 1) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 1) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 2))] = ((1 <= ((((((int)threadIdx.x) * 25) + 2) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 2) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 2) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 3))] = ((1 <= ((((((int)threadIdx.x) * 25) + 3) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 3) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 3) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 4))] = ((1 <= ((((((int)threadIdx.x) * 25) + 4) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 4) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 4) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 5))] = ((1 <= ((((((int)threadIdx.x) * 25) + 5) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 5) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 5) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 6))] = ((1 <= ((((((int)threadIdx.x) * 25) + 6) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 6) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 6) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 7))] = ((1 <= ((((((int)threadIdx.x) * 25) + 7) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 7) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 7) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 8))] = ((1 <= ((((((int)threadIdx.x) * 25) + 8) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 8) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 8) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 9))] = ((1 <= ((((((int)threadIdx.x) * 25) + 9) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 9) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 9) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 10))] = ((1 <= ((((((int)threadIdx.x) * 25) + 10) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 10) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 10) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 11))] = ((1 <= ((((((int)threadIdx.x) * 25) + 11) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 11) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 11) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 12))] = ((1 <= ((((((int)threadIdx.x) * 25) + 12) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 12) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 12) % 13)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 13))] = placeholder[((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((((int)threadIdx.x) * 25) / 13) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 25) % 13)))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 14))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 14) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 1) % 13)) - 14))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 15))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 15) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 2) % 13)) - 14))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 16))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 16) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 3) % 13)) - 14))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 17))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 17) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 4) % 13)) - 14))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 18))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 18) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 5) % 13)) - 14))];
      if (((((((int)threadIdx.x) * 25) + 19) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 19) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1333) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 19))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 19) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 6) % 13)) - 14))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 20) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 20) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1332) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 20))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 20) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 7) % 13)) - 14))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 21) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 21) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1331) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 21))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 21) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 8) % 13)) - 14))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 22) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 22) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1330) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 22))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 22) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 9) % 13)) - 14))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 23) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 23) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1329) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 23))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 23) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 10) % 13)) - 14))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 24) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 24) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1328) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 24))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 24) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 11) % 13)) - 14))];
            }
          }
        }
      }
      placeholder_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 10) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 10) & 7) * 9)) + (ry_outer * 3)) + 1))];
      placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 1))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 1) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 1) & 7) * 9)) + (ry_outer * 3)) + 1))];
      placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 2))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 2) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 2) & 7) * 9)) + (ry_outer * 3)) + 1))];
      placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 3))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 3) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 3) & 7) * 9)) + (ry_outer * 3)) + 1))];
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 4) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 508) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 4))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 4) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 4) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 5) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 507) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 5))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 5) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 5) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 6) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 506) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 6))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 6) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 6) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 7) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 505) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 7))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 7) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 7) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 10) >> 3)) < 63) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 504) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 8))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 10) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 10) & 7) * 9)) + (ry_outer * 3)) + 2305))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 9) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 503) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 9))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 9) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 1) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)))] = ((1 <= (((((int)threadIdx.x) * 25) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((((int)threadIdx.x) * 25) / 13) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 25) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 1))] = ((1 <= ((((((int)threadIdx.x) * 25) + 1) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 1) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 1) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 2))] = ((1 <= ((((((int)threadIdx.x) * 25) + 2) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 2) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 2) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 3))] = ((1 <= ((((((int)threadIdx.x) * 25) + 3) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 3) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 3) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 4))] = ((1 <= ((((((int)threadIdx.x) * 25) + 4) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 4) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 4) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 5))] = ((1 <= ((((((int)threadIdx.x) * 25) + 5) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 5) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 5) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 6))] = ((1 <= ((((((int)threadIdx.x) * 25) + 6) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 6) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 6) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 7))] = ((1 <= ((((((int)threadIdx.x) * 25) + 7) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 7) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 7) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 8))] = ((1 <= ((((((int)threadIdx.x) * 25) + 8) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 8) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 8) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 9))] = ((1 <= ((((((int)threadIdx.x) * 25) + 9) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 9) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 9) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 10))] = ((1 <= ((((((int)threadIdx.x) * 25) + 10) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 10) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 10) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 11))] = ((1 <= ((((((int)threadIdx.x) * 25) + 11) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 11) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 11) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 12))] = ((1 <= ((((((int)threadIdx.x) * 25) + 12) / 13) + ry_outer)) ? placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 12) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 12) % 13)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 13))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((((int)threadIdx.x) * 25) / 13) * 14)) + (ry_outer * 14)) + ((((int)threadIdx.x) * 25) % 13)) + 1))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 14))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 14) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 1) % 13)) - 13))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 15))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 15) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 2) % 13)) - 13))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 16))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 16) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 3) % 13)) - 13))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 17))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 17) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 4) % 13)) - 13))];
      pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 18))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 18) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 5) % 13)) - 13))];
      if (((((((int)threadIdx.x) * 25) + 19) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 19) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1333) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 19))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 19) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 6) % 13)) - 13))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 20) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 20) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1332) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 20))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 20) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 7) % 13)) - 13))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 21) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 21) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1331) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 21))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 21) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 8) % 13)) - 13))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 22) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 22) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1330) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 22))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 22) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 9) % 13)) - 13))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 23) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 23) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1329) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 23))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 23) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 10) % 13)) - 13))];
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 25) + 24) / 169) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.z) * 13) + (((((int)threadIdx.x) * 25) + 24) / 13)) < 104) {
          if (((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) < 1328) {
            if (((int)threadIdx.x) < 6) {
              pad_temp_shared[((((((int)threadIdx.z) * 169) + (((int)threadIdx.x) * 25)) + 24))] = placeholder[(((((((((((int)vz) >> 3) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + ((((((int)threadIdx.x) * 25) + 24) / 13) * 14)) + (ry_outer * 14)) + (((((int)threadIdx.x) * 25) + 11) % 13)) - 13))];
            }
          }
        }
      }
      placeholder_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 10) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 10) & 7) * 9)) + (ry_outer * 3)) + 2))];
      placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 1))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 1) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 1) & 7) * 9)) + (ry_outer * 3)) + 2))];
      placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 2))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 2) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 2) & 7) * 9)) + (ry_outer * 3)) + 2))];
      placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 3))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 3) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 3) & 7) * 9)) + (ry_outer * 3)) + 2))];
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 4) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 508) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 4))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 4) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 4) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 5) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 507) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 5))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 5) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 5) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 6) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 506) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 6))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 6) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 6) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 7) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 505) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 7))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 7) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 7) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 10) >> 3)) < 63) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 504) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 8))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 10) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 10) & 7) * 9)) + (ry_outer * 3)) + 2306))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 10) + 9) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) < 503) {
          if (((int)threadIdx.x) < 6) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 10)) + 9))] = placeholder1[(((((((((((int)vz) & 7) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 10) + 9) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 10) + 1) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 128))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 256))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 384))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 26))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 78))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 104))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 130))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 136))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 264))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 156))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 392))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 129))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 257))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 385))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 169))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 195))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 221))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 247))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 273))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 299))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 137))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 265))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 325))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 393))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 130))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 258))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 386))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 338))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 390))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 416))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 442))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 468))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 138))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 266))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 494))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 394))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 131))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 259))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 387))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 507))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 533))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 559))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 585))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 611))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 637))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 139))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 267))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 663))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 395))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 132))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 260))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 388))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 676))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 702))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 754))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 780))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 806))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 140))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 268))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 832))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 396))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 133))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 261))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 389))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 845))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 871))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 897))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 923))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 949))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 975))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 141))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 269))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1001))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 397))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 134))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 262))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 390))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1014))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1040))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1066))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1092))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1118))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1144))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 142))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 270))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1170))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 398))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 135))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 263))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 391))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1183))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1209))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1235))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1261))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1287))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1313))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 143))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 271))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1339))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 399))]));
    }
  }
  T_relu[((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 784))] = max((compute[(14)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 16))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1568))] = max((compute[(28)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 32))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2352))] = max((compute[(42)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 48))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 7))] = max((compute[(2)] + placeholder2[((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 791))] = max((compute[(16)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 16))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1575))] = max((compute[(30)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 32))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2359))] = max((compute[(44)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 48))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 14))] = max((compute[(4)] + placeholder2[((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 798))] = max((compute[(18)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 16))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1582))] = max((compute[(32)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 32))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2366))] = max((compute[(46)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 48))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 21))] = max((compute[(6)] + placeholder2[((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 805))] = max((compute[(20)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 16))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1589))] = max((compute[(34)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 32))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2373))] = max((compute[(48)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 48))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 28))] = max((compute[(8)] + placeholder2[((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 812))] = max((compute[(22)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 16))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1596))] = max((compute[(36)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 32))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2380))] = max((compute[(50)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 48))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 35))] = max((compute[(10)] + placeholder2[((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 819))] = max((compute[(24)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 16))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1603))] = max((compute[(38)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 32))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2387))] = max((compute[(52)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 48))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 42))] = max((compute[(12)] + placeholder2[((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 826))] = max((compute[(26)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 16))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1610))] = max((compute[(40)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 32))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2394))] = max((compute[(54)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 48))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 49))] = max((compute[(1)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 833))] = max((compute[(15)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 17))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1617))] = max((compute[(29)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 33))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2401))] = max((compute[(43)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 49))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 56))] = max((compute[(3)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 840))] = max((compute[(17)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 17))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1624))] = max((compute[(31)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 33))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2408))] = max((compute[(45)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 49))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 63))] = max((compute[(5)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 847))] = max((compute[(19)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 17))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1631))] = max((compute[(33)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 33))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2415))] = max((compute[(47)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 49))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 70))] = max((compute[(7)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 854))] = max((compute[(21)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 17))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1638))] = max((compute[(35)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 33))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2422))] = max((compute[(49)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 49))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 77))] = max((compute[(9)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 861))] = max((compute[(23)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 17))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1645))] = max((compute[(37)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 33))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2429))] = max((compute[(51)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 49))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 84))] = max((compute[(11)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 868))] = max((compute[(25)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 17))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1652))] = max((compute[(39)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 33))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2436))] = max((compute[(53)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 49))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 91))] = max((compute[(13)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 1))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 875))] = max((compute[(27)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 17))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 1659))] = max((compute[(41)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 33))]), 0.000000e+00f);
  T_relu[(((((((int)vz) * 3136) + (((int)threadIdx.z) * 98)) + ((int)threadIdx.x)) + 2443))] = max((compute[(55)] + placeholder2[(((((((int)vz) & 7) * 64) + (((int)threadIdx.z) * 2)) + 49))]), 0.000000e+00f);

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


    float *placeholder0 = new float[1605632];
    for(int i=0;i<1605632;i++)
    {
    placeholder0[i]=1;
    }
    float *g_ph0;
    checkCudaErrors(cudaMalloc((void **)&g_ph0, sizeof(float) * 1605632));
    checkCudaErrors(cudaMemcpy(g_ph0, placeholder0, sizeof(float) * 1605632, cudaMemcpyHostToDevice));

    float *placeholder1 = new float[1179648];
    for(int i=0;i<1179648;i++)
    {
    placeholder1[i]=0;
    }
    float *g_ph1;
    checkCudaErrors(cudaMalloc((void **)&g_ph1, sizeof(float) * 1179648));
    checkCudaErrors(cudaMemcpy(g_ph1, placeholder1, sizeof(float) * 1179648, cudaMemcpyHostToDevice));

    float *placeholder2 = new float[802816];
    for(int i=0;i<802816;i++)
    {
    placeholder2[i]=3;
    }
    float *g_ph2;
    checkCudaErrors(cudaMalloc((void **)&g_ph2, sizeof(float) * 802816));
    checkCudaErrors(cudaMemcpy(g_ph2, placeholder2, sizeof(float) * 802816, cudaMemcpyHostToDevice));


    float *placeholder3 = new float[512];
    for(int i=0;i<512;i++)
    {
    placeholder3[i]=5;
    }
    float *g_ph3;
    cudaMalloc((void **)&g_ph3, sizeof(float) * 512);
    cudaMemcpy(g_ph3, placeholder3, sizeof(float) * 512, cudaMemcpyHostToDevice);


    dim3 Dim_block = dim3(LAUNCH_BLOCKX, LAUNCH_BLOCKY, LAUNCH_BLOCKZ);
    dim3 Dim_thread = dim3(LAUNCH_THREADX, LAUNCH_THREADY, LAUNCH_THREADZ);

    printf("hello?");
    // launch kernel
    fused_nn_conv2d_add_multiply_add_nn_relu_kernel0<<<Dim_block, Dim_thread, 0, streams[0]>>>(g_worker,1, g_flag, g_ph0, g_ph1, g_ph2, g_ph3);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(placeholder2, g_ph2,sizeof(float) * 802816, cudaMemcpyDeviceToHost));
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