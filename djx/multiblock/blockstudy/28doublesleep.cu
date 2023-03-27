#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>


#define LAUNCH_THREADX 7
#define LAUNCH_THREADY 1
#define LAUNCH_THREADZ 4

#define LAUNCH_BLOCKX 1
#define ORI_BLOCKX 1
#define LAUNCH_BLOCKY 1
#define ORI_BLOCKY 1
#define LAUNCH_BLOCKZ 512 * 5 *4 // 5是额外部分，满足多层覆盖
#define ORI_BLOCKZ 512

#define SM_NUM 32
#define WORKER_NUM_PERSM 16

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

extern "C" __global__ void fused_nn_conv2d_add_multiply_add_nn_relu_kernel0(int *worker,int number,int *flag, float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3, float* __restrict__ placeholder4) {
    int* sm_flag = flag;
    __shared__ int basicoffset;
    int offset;
    int smid;

    //judge whether to continue work,which work to fetch
    if(threadIdx.x+threadIdx.y+threadIdx.z == 0)
    {
       basicoffset=-1;
       smid = get_smid();

       int blocknumber=atomicAdd(sm_flag + smid, 1);
       atomicAdd(worker + smid, 1);

       /*
       if(smid>63)
       {
       for(int sleeptime=0;sleeptime<400;sleeptime++)
       {
        __nanosleep(10);
       }
       }
       */

       //judge whther sm id is right
       if((smid < number*SM_NUM)&&(smid >= (number-1)*SM_NUM))
       {
            //judge whether worker is enough
            //get the basic offset for the block
            if(blocknumber< WORKER_NUM_PERSM)
            {

                basicoffset = WORKER_NUM_PERSM*(smid-(number-1)*SM_NUM) + blocknumber;
                //printf("smid %d\n", smid);
            }
       }
       else
       {
       for(int sleeptime=0;sleeptime<50;sleeptime++)
       {
        __nanosleep(10);
       }
       return;

       }

    }
    __syncthreads();
    if (basicoffset < 0)
    {


    return ;
    }

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
        // begin original
        float compute[56];
        __shared__ float pad_temp_shared[196];
        __shared__ float placeholder_shared[128];
        compute[(0)] = 0.000000e+00f;
        compute[(28)] = 0.000000e+00f;
        compute[(4)] = 0.000000e+00f;
        compute[(32)] = 0.000000e+00f;
        compute[(8)] = 0.000000e+00f;
        compute[(36)] = 0.000000e+00f;
        compute[(12)] = 0.000000e+00f;
        compute[(40)] = 0.000000e+00f;
        compute[(16)] = 0.000000e+00f;
        compute[(44)] = 0.000000e+00f;
        compute[(20)] = 0.000000e+00f;
        compute[(48)] = 0.000000e+00f;
        compute[(24)] = 0.000000e+00f;
        compute[(52)] = 0.000000e+00f;
        compute[(1)] = 0.000000e+00f;
        compute[(29)] = 0.000000e+00f;
        compute[(5)] = 0.000000e+00f;
        compute[(33)] = 0.000000e+00f;
        compute[(9)] = 0.000000e+00f;
        compute[(37)] = 0.000000e+00f;
        compute[(13)] = 0.000000e+00f;
        compute[(41)] = 0.000000e+00f;
        compute[(17)] = 0.000000e+00f;
        compute[(45)] = 0.000000e+00f;
        compute[(21)] = 0.000000e+00f;
        compute[(49)] = 0.000000e+00f;
        compute[(25)] = 0.000000e+00f;
        compute[(53)] = 0.000000e+00f;
        compute[(2)] = 0.000000e+00f;
        compute[(30)] = 0.000000e+00f;
        compute[(6)] = 0.000000e+00f;
        compute[(34)] = 0.000000e+00f;
        compute[(10)] = 0.000000e+00f;
        compute[(38)] = 0.000000e+00f;
        compute[(14)] = 0.000000e+00f;
        compute[(42)] = 0.000000e+00f;
        compute[(18)] = 0.000000e+00f;
        compute[(46)] = 0.000000e+00f;
        compute[(22)] = 0.000000e+00f;
        compute[(50)] = 0.000000e+00f;
        compute[(26)] = 0.000000e+00f;
        compute[(54)] = 0.000000e+00f;
        compute[(3)] = 0.000000e+00f;
        compute[(31)] = 0.000000e+00f;
        compute[(7)] = 0.000000e+00f;
        compute[(35)] = 0.000000e+00f;
        compute[(11)] = 0.000000e+00f;
        compute[(39)] = 0.000000e+00f;
        compute[(15)] = 0.000000e+00f;
        compute[(43)] = 0.000000e+00f;
        compute[(19)] = 0.000000e+00f;
        compute[(47)] = 0.000000e+00f;
        compute[(23)] = 0.000000e+00f;
        compute[(51)] = 0.000000e+00f;
        compute[(27)] = 0.000000e+00f;
        compute[(55)] = 0.000000e+00f;
        for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
        for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
            for (int rx_outer = 0; rx_outer < 3; ++rx_outer) {
            __syncthreads();
            pad_temp_shared[(((((int)threadIdx.z) * 49) + (((int)threadIdx.x) * 7)))] = ((((1 <= (((int)threadIdx.x) + ry_outer)) && ((((int)threadIdx.x) + ry_outer) < 8)) && (1 <= rx_outer)) ? placeholder[(((((((((((int)vz) >> 4) * 25088) + (rc_outer * 196)) + (((int)threadIdx.z) * 49)) + (ry_outer * 7)) + (((int)threadIdx.x) * 7)) + rx_outer) - 8))] : 0.000000e+00f);
            pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.x) * 7)) + 1))] = (((1 <= (((int)threadIdx.x) + ry_outer)) && ((((int)threadIdx.x) + ry_outer) < 8)) ? placeholder[(((((((((((int)vz) >> 4) * 25088) + (rc_outer * 196)) + (((int)threadIdx.z) * 49)) + (ry_outer * 7)) + (((int)threadIdx.x) * 7)) + rx_outer) - 7))] : 0.000000e+00f);
            pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.x) * 7)) + 2))] = (((1 <= (((int)threadIdx.x) + ry_outer)) && ((((int)threadIdx.x) + ry_outer) < 8)) ? placeholder[(((((((((((int)vz) >> 4) * 25088) + (rc_outer * 196)) + (((int)threadIdx.z) * 49)) + (ry_outer * 7)) + (((int)threadIdx.x) * 7)) + rx_outer) - 6))] : 0.000000e+00f);
            pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.x) * 7)) + 3))] = (((1 <= (((int)threadIdx.x) + ry_outer)) && ((((int)threadIdx.x) + ry_outer) < 8)) ? placeholder[(((((((((((int)vz) >> 4) * 25088) + (rc_outer * 196)) + (((int)threadIdx.z) * 49)) + (ry_outer * 7)) + (((int)threadIdx.x) * 7)) + rx_outer) - 5))] : 0.000000e+00f);
            pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.x) * 7)) + 4))] = (((1 <= (((int)threadIdx.x) + ry_outer)) && ((((int)threadIdx.x) + ry_outer) < 8)) ? placeholder[(((((((((((int)vz) >> 4) * 25088) + (rc_outer * 196)) + (((int)threadIdx.z) * 49)) + (ry_outer * 7)) + (((int)threadIdx.x) * 7)) + rx_outer) - 4))] : 0.000000e+00f);
            pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.x) * 7)) + 5))] = (((1 <= (((int)threadIdx.x) + ry_outer)) && ((((int)threadIdx.x) + ry_outer) < 8)) ? placeholder[(((((((((((int)vz) >> 4) * 25088) + (rc_outer * 196)) + (((int)threadIdx.z) * 49)) + (ry_outer * 7)) + (((int)threadIdx.x) * 7)) + rx_outer) - 3))] : 0.000000e+00f);
            pad_temp_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.x) * 7)) + 6))] = ((((1 <= (((int)threadIdx.x) + ry_outer)) && ((((int)threadIdx.x) + ry_outer) < 8)) && (rx_outer < 2)) ? placeholder[(((((((((((int)vz) >> 4) * 25088) + (rc_outer * 196)) + (((int)threadIdx.z) * 49)) + (ry_outer * 7)) + (((int)threadIdx.x) * 7)) + rx_outer) - 2))] : 0.000000e+00f);
            placeholder_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)))] = placeholder1[(((((((((((int)vz) & 15) * 147456) + (((int)threadIdx.z) * 36864)) + (((((int)threadIdx.x) * 5) >> 2) * 4608)) + (rc_outer * 36)) + (((((int)threadIdx.x) * 5) & 3) * 9)) + (ry_outer * 3)) + rx_outer))];
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 1))] = placeholder1[(((((((((((int)vz) & 15) * 147456) + (((int)threadIdx.z) * 36864)) + ((((((int)threadIdx.x) * 5) + 1) >> 2) * 4608)) + (rc_outer * 36)) + ((((((int)threadIdx.x) * 5) + 1) & 3) * 9)) + (ry_outer * 3)) + rx_outer))];
            if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 2) >> 2)) < 32) {
                if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) < 126) {
                if (((int)threadIdx.x) < 6) {
                    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 2))] = placeholder1[(((((((((((int)vz) & 15) * 147456) + (((int)threadIdx.z) * 36864)) + ((((((int)threadIdx.x) * 5) + 2) >> 2) * 4608)) + (rc_outer * 36)) + ((((((int)threadIdx.x) * 5) + 2) & 3) * 9)) + (ry_outer * 3)) + rx_outer))];
                }
                }
            }
            if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 3) >> 2)) < 32) {
                if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) < 125) {
                if (((int)threadIdx.x) < 6) {
                    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 3))] = placeholder1[(((((((((((int)vz) & 15) * 147456) + (((int)threadIdx.z) * 36864)) + ((((((int)threadIdx.x) * 5) + 3) >> 2) * 4608)) + (rc_outer * 36)) + ((((((int)threadIdx.x) * 5) + 3) & 3) * 9)) + (ry_outer * 3)) + rx_outer))];
                }
                }
            }
            if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 5) >> 2)) < 31) {
                if (((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) < 124) {
                if (((int)threadIdx.x) < 6) {
                    placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 5)) + 4))] = placeholder1[((((((((((((int)vz) & 15) * 147456) + (((int)threadIdx.z) * 36864)) + (((((int)threadIdx.x) * 5) >> 2) * 4608)) + (rc_outer * 36)) + (((((int)threadIdx.x) * 5) & 3) * 9)) + (ry_outer * 3)) + rx_outer) + 4608))];
                }
                }
            }
            __syncthreads();
            compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
            compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 64))]));
            compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 7))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
            compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 64))]));
            compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
            compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 64))]));
            compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 21))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
            compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 64))]));
            compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
            compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 64))]));
            compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 35))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
            compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 64))]));
            compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[((((int)threadIdx.z) * 16))]));
            compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 64))]));
            compute[(1)] = (compute[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
            compute[(29)] = (compute[(29)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 68))]));
            compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
            compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 68))]));
            compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
            compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 68))]));
            compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
            compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 68))]));
            compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
            compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 68))]));
            compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
            compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 68))]));
            compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 4))]));
            compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 68))]));
            compute[(2)] = (compute[(2)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
            compute[(30)] = (compute[(30)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 72))]));
            compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
            compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 72))]));
            compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
            compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 72))]));
            compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
            compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 72))]));
            compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
            compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 72))]));
            compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
            compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 72))]));
            compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 8))]));
            compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 72))]));
            compute[(3)] = (compute[(3)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
            compute[(31)] = (compute[(31)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 76))]));
            compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
            compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 7))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 76))]));
            compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
            compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 76))]));
            compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
            compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 21))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 76))]));
            compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
            compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 76))]));
            compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
            compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 35))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 76))]));
            compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 12))]));
            compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 76))]));
            compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
            compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 65))]));
            compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
            compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 65))]));
            compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
            compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 65))]));
            compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
            compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 65))]));
            compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
            compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 65))]));
            compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
            compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 65))]));
            compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 1))]));
            compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 65))]));
            compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
            compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 69))]));
            compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
            compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 69))]));
            compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
            compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 69))]));
            compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
            compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 69))]));
            compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
            compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 69))]));
            compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
            compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 69))]));
            compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 5))]));
            compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 69))]));
            compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
            compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 73))]));
            compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
            compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 73))]));
            compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
            compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 73))]));
            compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
            compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 73))]));
            compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
            compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 73))]));
            compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
            compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 73))]));
            compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 9))]));
            compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 73))]));
            compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
            compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 77))]));
            compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
            compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 77))]));
            compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
            compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 63))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 77))]));
            compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
            compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 77))]));
            compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
            compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 77))]));
            compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
            compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 77))]));
            compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 13))]));
            compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 77))]));
            compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
            compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 66))]));
            compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
            compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 66))]));
            compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
            compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 66))]));
            compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 119))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
            compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 119))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 66))]));
            compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
            compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 66))]));
            compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 133))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
            compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 133))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 66))]));
            compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 2))]));
            compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 66))]));
            compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
            compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 70))]));
            compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
            compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 70))]));
            compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
            compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 70))]));
            compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 119))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
            compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 119))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 70))]));
            compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
            compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 70))]));
            compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 133))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
            compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 133))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 70))]));
            compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 6))]));
            compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 70))]));
            compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
            compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 74))]));
            compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
            compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 74))]));
            compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
            compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 74))]));
            compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 119))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
            compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 119))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 74))]));
            compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
            compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 74))]));
            compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 133))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
            compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 133))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 74))]));
            compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 10))]));
            compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 74))]));
            compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
            compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 78))]));
            compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
            compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 78))]));
            compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
            compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 78))]));
            compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 119))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
            compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 119))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 78))]));
            compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
            compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 78))]));
            compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 133))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
            compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 133))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 78))]));
            compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 14))]));
            compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 78))]));
            compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
            compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 67))]));
            compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
            compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 67))]));
            compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 161))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
            compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 161))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 67))]));
            compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
            compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 67))]));
            compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
            compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 67))]));
            compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
            compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 67))]));
            compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 3))]));
            compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 67))]));
            compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
            compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 71))]));
            compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
            compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 71))]));
            compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 161))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
            compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 161))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 71))]));
            compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
            compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 71))]));
            compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
            compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 71))]));
            compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
            compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 71))]));
            compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 7))]));
            compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 71))]));
            compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
            compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 75))]));
            compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
            compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 75))]));
            compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 161))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
            compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 161))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 75))]));
            compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
            compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 75))]));
            compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
            compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 75))]));
            compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
            compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 75))]));
            compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 11))]));
            compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 75))]));
            compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
            compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 147))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 79))]));
            compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
            compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 79))]));
            compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 161))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
            compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 161))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 79))]));
            compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
            compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 79))]));
            compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
            compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 175))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 79))]));
            compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
            compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 79))]));
            compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 15))]));
            compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 189))] * placeholder_shared[(((((int)threadIdx.z) * 16) + 79))]));
            }
        }
        }
        T_relu[((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)))] = max((((compute[(0)] + placeholder2[((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)))]) * placeholder3[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]) + placeholder4[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 784))] = max((((compute[(28)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 784))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 7))] = max((((compute[(4)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 7))]) * placeholder3[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]) + placeholder4[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 791))] = max((((compute[(32)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 791))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 14))] = max((((compute[(8)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 14))]) * placeholder3[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]) + placeholder4[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 798))] = max((((compute[(36)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 798))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 21))] = max((((compute[(12)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 21))]) * placeholder3[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]) + placeholder4[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 805))] = max((((compute[(40)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 805))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 28))] = max((((compute[(16)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 28))]) * placeholder3[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]) + placeholder4[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 812))] = max((((compute[(44)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 812))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 35))] = max((((compute[(20)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 35))]) * placeholder3[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]) + placeholder4[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 819))] = max((((compute[(48)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 819))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 42))] = max((((compute[(24)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 42))]) * placeholder3[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]) + placeholder4[((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 826))] = max((((compute[(52)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 826))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 16))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 49))] = max((((compute[(1)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 49))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 833))] = max((((compute[(29)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 833))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 56))] = max((((compute[(5)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 56))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 840))] = max((((compute[(33)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 840))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 63))] = max((((compute[(9)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 63))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 847))] = max((((compute[(37)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 847))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 70))] = max((((compute[(13)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 70))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 854))] = max((((compute[(41)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 854))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 77))] = max((((compute[(17)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 77))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 861))] = max((((compute[(45)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 861))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 84))] = max((((compute[(21)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 84))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 868))] = max((((compute[(49)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 868))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 91))] = max((((compute[(25)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 91))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 1))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 875))] = max((((compute[(53)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 875))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 17))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 98))] = max((((compute[(2)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 98))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 882))] = max((((compute[(30)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 882))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 105))] = max((((compute[(6)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 105))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 889))] = max((((compute[(34)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 889))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 112))] = max((((compute[(10)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 112))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 896))] = max((((compute[(38)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 896))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 119))] = max((((compute[(14)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 119))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 903))] = max((((compute[(42)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 903))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 126))] = max((((compute[(18)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 126))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 910))] = max((((compute[(46)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 910))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 133))] = max((((compute[(22)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 133))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 917))] = max((((compute[(50)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 917))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 140))] = max((((compute[(26)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 140))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 2))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 924))] = max((((compute[(54)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 924))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 18))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 147))] = max((((compute[(3)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 147))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 931))] = max((((compute[(31)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 931))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 154))] = max((((compute[(7)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 154))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 938))] = max((((compute[(35)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 938))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 161))] = max((((compute[(11)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 161))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 945))] = max((((compute[(39)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 945))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 168))] = max((((compute[(15)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 168))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 952))] = max((((compute[(43)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 952))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 175))] = max((((compute[(19)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 175))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 959))] = max((((compute[(47)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 959))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 182))] = max((((compute[(23)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 182))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 966))] = max((((compute[(51)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 966))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 189))] = max((((compute[(27)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 189))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 3))]), 0.000000e+00f);
        T_relu[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 973))] = max((((compute[(55)] + placeholder2[(((((((int)vz) * 1568) + (((int)threadIdx.z) * 196)) + ((int)threadIdx.x)) + 973))]) * placeholder3[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]) + placeholder4[(((((((int)vz) & 15) * 32) + (((int)threadIdx.z) * 4)) + 19))]), 0.000000e+00f);

        __syncthreads();
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

    int *g_worker2;
    checkCudaErrors(cudaMalloc((void **)&g_worker2, sizeof(int) * 80));
    checkCudaErrors(cudaMemcpy( g_worker2,workers, sizeof(int) * 80, cudaMemcpyHostToDevice));


    float *placeholder0 = new float[802816];
    for(int i=0;i<802816;i++)
    {
    placeholder0[i]=1;
    }
    float *g_ph0;
    checkCudaErrors(cudaMalloc((void **)&g_ph0, sizeof(float) * 802816));
    checkCudaErrors(cudaMemcpy(g_ph0, placeholder0, sizeof(float) * 802816, cudaMemcpyHostToDevice));

    float *placeholder1 = new float[2359296];
    for(int i=0;i<2359296;i++)
    {
    placeholder1[i]=2;
    }
    float *g_ph1;
    checkCudaErrors(cudaMalloc((void **)&g_ph1, sizeof(float) * 2359296));
    //checkCudaErrors(cudaMemcpy(g_ph1, placeholder1, sizeof(float) * 2359296, cudaMemcpyHostToDevice));

    float *placeholder2 = new float[802816];
    for(int i=0;i<802816;i++)
    {
    placeholder2[i]=3;
    }
    float *g_ph2;
    checkCudaErrors(cudaMalloc((void **)&g_ph2, sizeof(float) * 802816));
    checkCudaErrors(cudaMemcpy(g_ph2, placeholder2, sizeof(float) * 802816, cudaMemcpyHostToDevice));

    float *placeholder3 = new float[802816];
    for(int i=0;i<802816;i++)
    {
    placeholder3[i]=4;
    }
    float *g_ph3;
    cudaMalloc((void **)&g_ph3, sizeof(float) * 802816);
    cudaMemcpy(g_ph3, placeholder3, sizeof(float) * 802816, cudaMemcpyHostToDevice);

    float *placeholder4 = new float[512];
    for(int i=0;i<512;i++)
    {
    placeholder4[i]=5;
    }
    float *g_ph4;
    cudaMalloc((void **)&g_ph4, sizeof(float) * 512);
    cudaMemcpy(g_ph4, placeholder4, sizeof(float) * 512, cudaMemcpyHostToDevice);

    float *placeholder5 = new float[802816];
    for(int i=0;i<802816;i++)
    {
    placeholder5[i]=6;
    }
    float *g_ph5;
    cudaMalloc((void **)&g_ph5, sizeof(float) * 802816);
    cudaMemcpy(g_ph5, placeholder5, sizeof(float) * 802816, cudaMemcpyHostToDevice);

    //prepare parm for kernel 2
    float *g_ph0_;
    checkCudaErrors(cudaMalloc((void **)&g_ph0_, sizeof(float) * 802816));
    checkCudaErrors(cudaMemcpy(g_ph0_, placeholder0, sizeof(float) * 802816, cudaMemcpyHostToDevice));


    float *g_ph1_;
    checkCudaErrors(cudaMalloc((void **)&g_ph1_, sizeof(float) * 2359296));
    //checkCudaErrors(cudaMemcpy(g_ph1_, placeholder1, sizeof(float) * 2359296, cudaMemcpyHostToDevice));


    float *g_ph2_;
    checkCudaErrors(cudaMalloc((void **)&g_ph2_, sizeof(float) * 802816));
    checkCudaErrors(cudaMemcpy(g_ph2_, placeholder2, sizeof(float) * 802816, cudaMemcpyHostToDevice));


    float *g_ph3_;
    cudaMalloc((void **)&g_ph3_, sizeof(float) * 802816);
    cudaMemcpy(g_ph3_, placeholder3, sizeof(float) * 802816, cudaMemcpyHostToDevice);


    float *g_ph4_;
    cudaMalloc((void **)&g_ph4_, sizeof(float) * 512);
    cudaMemcpy(g_ph4_, placeholder4, sizeof(float) * 512, cudaMemcpyHostToDevice);


    float *g_ph5_;
    cudaMalloc((void **)&g_ph5_, sizeof(float) * 802816);
    cudaMemcpy(g_ph5_, placeholder5, sizeof(float) * 802816, cudaMemcpyHostToDevice);



    dim3 Dim_block = dim3(LAUNCH_BLOCKX, LAUNCH_BLOCKY, LAUNCH_BLOCKZ);
    dim3 Dim_thread = dim3(LAUNCH_THREADX, LAUNCH_THREADY, LAUNCH_THREADZ);

    printf("hello?");
    // launch kernel

    fused_nn_conv2d_add_multiply_add_nn_relu_kernel0<<<Dim_block, Dim_thread, 0, streams[1]>>>(g_worker2,2, g_flag_, g_ph0_, g_ph1_, g_ph2_, g_ph3_, g_ph4_, g_ph5_);
    fused_nn_conv2d_add_multiply_add_nn_relu_kernel0<<<Dim_block, Dim_thread, 0, streams[0]>>>(g_worker,1, g_flag, g_ph0, g_ph1, g_ph2, g_ph3, g_ph4, g_ph5);

    cudaDeviceSynchronize();
    printf("hello2?");
    checkCudaErrors(cudaMemcpy(placeholder2, g_ph2,sizeof(float) * 802816, cudaMemcpyDeviceToHost));
    printf("hello3?\n");
    /*
    for(int j=0;j<784;j++)
    {
    if(j%10==0)
    {
    printf("\n");
    }
    printf("%f  ",placeholder2[1024*j+j]);
    }
    */

    printf("\n");
    printf("kernel 1 \n");
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
    printf("kernel 2 \n");
    checkCudaErrors(cudaMemcpy(workers,g_worker2,sizeof(int) * 80, cudaMemcpyDeviceToHost));
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