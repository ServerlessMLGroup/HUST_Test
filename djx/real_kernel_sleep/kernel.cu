#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
// nvcc -arch=native ex.cu -o ex_sleep

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

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel1(float* __restrict__ placeholder, float* __restrict__ data_pack, float* __restrict__ bgemm, long long unsigned* times) {
    float bgemm_local[8];
    __shared__ float placeholder_shared[1024];
    __shared__ float data_pack_shared[256];
    for (int co_c_init = 0; co_c_init < 4; ++co_c_init) {
      for (int p_c_init = 0; p_c_init < 2; ++p_c_init) {
        bgemm_local[(((co_c_init * 2) + p_c_init))] = 0.000000e+00f;
      }
    }
    for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
        placeholder_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = placeholder[(((((((((int)blockIdx.z) * 262144) + (ci_outer * 8192)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 1024)) + ((((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) >> 6) * 512)) + (((int)blockIdx.y) * 64)) + (((((int)threadIdx.y) * 8) + ((int)threadIdx.x)) & 63)))];
      }
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1) {
        data_pack_shared[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))] = data_pack[((((((((int)blockIdx.z) * 8192) + (ci_outer * 256)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer1 * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x)))];
      }
      __syncthreads();
      for (int ci_inner = 0; ci_inner < 16; ++ci_inner) {
        for (int co_c = 0; co_c < 4; ++co_c) {
          for (int p_c = 0; p_c < 2; ++p_c) {
            bgemm_local[(((co_c * 2) + p_c))] = (bgemm_local[(((co_c * 2) + p_c))] + (placeholder_shared[((((ci_inner * 64) + (((int)threadIdx.y) * 4)) + co_c))] * data_pack_shared[((((ci_inner * 16) + (((int)threadIdx.x) * 2)) + p_c))]));
          }
        }
      }
    }
    for (int co_inner_inner_inner = 0; co_inner_inner_inner < 4; ++co_inner_inner_inner) {
      for (int p_inner_inner_inner = 0; p_inner_inner_inner < 2; ++p_inner_inner_inner) {
        bgemm[(((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.y) * 64)) + (co_inner_inner_inner * 16)) + (((int)threadIdx.x) * 2)) + p_inner_inner_inner))] = bgemm_local[(((co_inner_inner_inner * 2) + p_inner_inner_inner))];
      }
    }

}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack, long long unsigned* times) {
  float d[16];
    float data_pack_local[16];
    for (int eps = 0; eps < 4; ++eps) {
      for (int nu = 0; nu < 4; ++nu) {
          d[(((eps * 4) + nu))] = (((((1 <= ((((((int)threadIdx.x) & 15) >> 2) * 2) + eps)) && (((((((int)threadIdx.x) & 15) >> 2) * 2) + eps) < 8)) && (1 <= (((((int)threadIdx.x) & 3) * 2) + nu))) && ((((((int)threadIdx.x) & 3) * 2) + nu) < 8)) ? placeholder[((((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (eps * 7)) + ((((int)threadIdx.x) & 3) * 2)) + nu) - 8))] : 0.000000e+00f);
      }
  }
  data_pack_local[(0)] = 0.000000e+00f;
  data_pack_local[(0)] = (data_pack_local[(0)] + d[(0)]);
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(2)] * -1.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + (d[(8)] * -1.000000e+00f));
  data_pack_local[(0)] = (data_pack_local[(0)] + ((d[(10)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(1)] = 0.000000e+00f;
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(1)] * -1.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + d[(2)]);
  data_pack_local[(1)] = (data_pack_local[(1)] + ((d[(9)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(1)] = (data_pack_local[(1)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(2)] = 0.000000e+00f;
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(1)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + d[(2)]);
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(2)] = (data_pack_local[(2)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(3)] = 0.000000e+00f;
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(1)] * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + d[(3)]);
  data_pack_local[(3)] = (data_pack_local[(3)] + ((d[(9)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(3)] = (data_pack_local[(3)] + (d[(11)] * -1.000000e+00f));
  data_pack_local[(4)] = 0.000000e+00f;
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(4)] * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + ((d[(6)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(4)] = (data_pack_local[(4)] + d[(8)]);
  data_pack_local[(4)] = (data_pack_local[(4)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(5)] = 0.000000e+00f;
  data_pack_local[(5)] = (data_pack_local[(5)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(5)] = (data_pack_local[(5)] + d[(10)]);
  data_pack_local[(6)] = 0.000000e+00f;
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(9)]);
  data_pack_local[(6)] = (data_pack_local[(6)] + d[(10)]);
  data_pack_local[(7)] = 0.000000e+00f;
  data_pack_local[(7)] = (data_pack_local[(7)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(7)] = (data_pack_local[(7)] + d[(11)]);
  data_pack_local[(8)] = 0.000000e+00f;
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(4)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(8)] = (data_pack_local[(8)] + d[(8)]);
  data_pack_local[(8)] = (data_pack_local[(8)] + (d[(10)] * -1.000000e+00f));
  data_pack_local[(9)] = 0.000000e+00f;
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(6)]);
  data_pack_local[(9)] = (data_pack_local[(9)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(9)] = (data_pack_local[(9)] + d[(10)]);
  data_pack_local[(10)] = 0.000000e+00f;
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(5)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(6)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(9)]);
  data_pack_local[(10)] = (data_pack_local[(10)] + d[(10)]);
  data_pack_local[(11)] = 0.000000e+00f;
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(7)]);
  data_pack_local[(11)] = (data_pack_local[(11)] + (d[(9)] * -1.000000e+00f));
  data_pack_local[(11)] = (data_pack_local[(11)] + d[(11)]);
  data_pack_local[(12)] = 0.000000e+00f;
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(4)] * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + ((d[(6)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(12)] = (data_pack_local[(12)] + d[(12)]);
  data_pack_local[(12)] = (data_pack_local[(12)] + (d[(14)] * -1.000000e+00f));
  data_pack_local[(13)] = 0.000000e+00f;
  data_pack_local[(13)] = (data_pack_local[(13)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(13)] = (data_pack_local[(13)] + d[(14)]);
  data_pack_local[(14)] = 0.000000e+00f;
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(5)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + (d[(6)] * -1.000000e+00f));
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(13)]);
  data_pack_local[(14)] = (data_pack_local[(14)] + d[(14)]);
  data_pack_local[(15)] = 0.000000e+00f;
  data_pack_local[(15)] = (data_pack_local[(15)] + ((d[(5)] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(7)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + (d[(13)] * -1.000000e+00f));
  data_pack_local[(15)] = (data_pack_local[(15)] + d[(15)]);
  for (int eps1 = 0; eps1 < 4; ++eps1) {
      for (int nu1 = 0; nu1 < 4; ++nu1) {
      data_pack[(((((eps1 * 32768) + (nu1 * 8192)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x)))] = data_pack_local[(((eps1 * 4) + nu1))];
      }
  }
}


void run_kernel() {
	int num_streams = 2;
	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}
	

    long long unsigned *h_sm_ids = new long long unsigned[64 * 2];
    long long unsigned *d_sm_ids;
    cudaMalloc(&d_sm_ids, 64 * sizeof(long long unsigned) * 2);
    
    long long unsigned *h_sm_ids2 = new long long unsigned[128 * 2];
    long long unsigned *d_sm_ids2;
    cudaMalloc(&d_sm_ids2, 128 * sizeof(long long unsigned) * 2);

    float *h_args_55 = new float[25088]; // 55
    float *d_args_55;
    cudaMalloc(&d_args_55, sizeof(float) * 25088);

    float *h_args_56 = new float[4194304]; // 56
    float *d_args_56;
    cudaMalloc(&d_args_56, sizeof(float) * 4194304);

    float *h_args_75 = new float[1806336 / 4 + 1]; // 55
    float *d_args_75;
    cudaMalloc(&d_args_75, sizeof(float) * 1806336 / 4 + 4);

    float *h_args_76 = new float[1806336 / 4 + 1]; // 55
    float *d_args_76;
    cudaMalloc(&d_args_76, sizeof(float) * 1806336 / 4 + 4);


    // allocate flag
    int *flag;
    int *g_flag;
    flag = (int*) malloc(1 * sizeof(int));
    flag[0] = 0;
    cudaMalloc((void **)&g_flag, sizeof(int) * 1);
    cudaMemcpy(g_flag, flag, sizeof(int) * 1, cudaMemcpyHostToDevice);


    // allocate warm flag
    int *flag_warm;
    int *g_flag_warm;
    flag_warm = (int*) malloc(1 * sizeof(int));
    flag_warm[0] = 0;
    cudaMalloc((void **)&g_flag_warm, sizeof(int) * 1);
    cudaMemcpy(g_flag_warm, flag_warm, sizeof(int) * 1, cudaMemcpyHostToDevice);

    // cuda launch kernel
    dim3 D_b_a = dim3(64, 1, 1);
    dim3 D_t_a = dim3(128, 1, 1);
    dim3 D_b_b = dim3(1, 8, 16);
    dim3 D_t_b = dim3(8, 16, 1);
    // warm-up
    for (int i = 0; i < 100; ++i) {
        fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel0 <<<D_b_a, D_t_a, 0, streams[0]>>>(d_args_55, d_args_76, d_sm_ids);
    }
    cudaDeviceSynchronize();
    // test kernel
    fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel0 <<<D_b_a, D_t_a, 0, streams[0]>>>(d_args_55, d_args_76, d_sm_ids);
    cudaDeviceSynchronize();
    // sleep until kernel finish
    fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel1 <<<D_b_b, D_t_b, 0, streams[1]>>>(d_args_56, d_args_76, d_args_75, d_sm_ids2);
    
	
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_sm_ids, d_sm_ids, 64 * sizeof(long long unsigned) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sm_ids2, d_sm_ids2, 128 * sizeof(long long unsigned) * 2, cudaMemcpyDeviceToHost);
    

    // cudaMemcpy(h_sleep_time, d_sleep_time, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_sleep_sm, d_sleep_sm, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);

    long long unsigned maxm = 0, minm = 1768959725180341, max1 = 0, max2=0, min2=1768959725180341;
    long long unsigned maxm_e = 0, minm_e = 1768959725180341;
    printf("---1---\n");
    for (int i = 0; i < 64; i++) {
          // printf("%llu-%llu\n", h_sm_ids[i], h_sm_ids[i + 64]);
          maxm = max(maxm, h_sm_ids[i]);
          minm = min(minm, h_sm_ids[i]);
          maxm_e = max(maxm_e, h_sm_ids[i + 64]);
          minm_e = min(minm_e, h_sm_ids[i + 64]);
          max1 = max(max1, h_sm_ids[i + 64] - h_sm_ids[i]);
    }
    printf("START_TIMING:max-%llu, min-%llu(us)\n", maxm, minm);
    printf("END_TIMING__:max-%llu, min-%llu(us)\n", maxm_e, minm_e);
    printf("DURATION:单block最大执行时间%llu(us)\n", max1);
          
    maxm = 0; minm = 1768959725180341;
    maxm_e = 0; minm_e = 1768959725180341;
    printf("---2---\n");
    for (int i = 0; i < 128; i++) {
      // printf("blcok%d:%llu-%llu   %llu \n",i, h_sm_ids2[i], h_sm_ids2[i + a_blocks] , h_sm_ids2[i + b_blocks]-h_sm_ids2[i]);
          // printf("%llu-%llu\n", h_sm_ids2[i], h_sm_ids2[i + 128]);
          maxm = max(maxm, h_sm_ids2[i]);
          minm = min(minm, h_sm_ids2[i]);
          maxm_e = max(maxm_e, h_sm_ids2[i + 128]);
          minm_e = min(minm_e, h_sm_ids2[i + 128]);
          max2 = max(max2, h_sm_ids2[i + 128]-h_sm_ids2[i]);
          min2 = min(min2, h_sm_ids2[i + 128]-h_sm_ids2[i]);
    }
    printf("START_TIMING:max-%llu, min-%llu(us)\n", maxm, minm);
    printf("END_TIMING__:max-%llu, min-%llu(us)\n", maxm_e, minm_e);
    printf("DURATION:单block最大执行时间%llu(us)  单block最大执行时间与最小的时间差%llu(us)\n", max2, max2 - min2);

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
        exit(1);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));
	run_kernel();

	return 0;
}

