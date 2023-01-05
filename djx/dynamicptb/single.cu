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

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel0_warm(float* __restrict__ placeholder, float* __restrict__ data_pack, int* flag, long long unsigned* times, long long unsigned* sm) {
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

// sm_flag指示i号sm是保留的原先几号block
extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel0(float* __restrict__ placeholder, float* __restrict__ data_pack, int* sm_flag, long long unsigned* worker_num, int* block_flag, long long unsigned* time) {
    unsigned int ns = 5;
    int smid = get_smid();
    if (threadIdx.x == 0 && atomicAdd(sm_flag + smid, 1) == 0) atomicAdd(block_flag + blockIdx.x, 1);
    __syncthreads();
    // while(atomicAdd(flag, 0) == 0) { // 40us版本
    //     __nanosleep(ns); 
    //     if (ns < 1000) {
    //         ns *= 2;
    //     }
    // }

    if (atomicAdd(block_flag + blockIdx.x, 0) == 0) return ;
    __syncthreads();
    
    // unsigned long long mclk;
    // if (threadIdx.x == 1) {
    //     asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
    //     time[get_smid()] = mclk / 1000;
    // }
    // while ((t1 - t0)/(1530000 * 1000.0f / 1000000) < 20) t1 = clock64(); // 20us, 1530000为kilohertz

    // if (threadIdx.x == 0) printf("%d %d\n", smid, blockIdx.x);
    // for (int i = 0; i < 100; ++i) { // 模拟多次主动感知
    ns = 5;
    __shared__ int BlockSyn[128 + 5];
    BlockSyn[threadIdx.x] = 0;
    if (threadIdx.x == 0) {
        while (atomicAdd(worker_num + smid, 0) == 0) {
            //if (threadIdx.x == 0) time[smid] += 1; 
            __nanosleep(10);
            // if (ns < 1000) {
            //     ns *= 2;
            // }
        }
        for(int i = 0; i < 128 + 5; ++i) {
            atomicAdd(BlockSyn + i, 1);
        }
    }
    else {
        while (atomicAdd(BlockSyn + threadIdx.x, 0) == 0) {
            //if (threadIdx.x == 0) time[smid] += 1; 
            __nanosleep(10);
            // if (ns < 1000) {
            //     ns *= 2;
            // }
        }
    }
    // if (threadIdx.x == 0) {
    //     asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk2));
    //     time[get_smid() + 80] = mclk2 / 1000;
    // }
    // if (get_smid() != smid - 1) printf("error in %d-%d\n", smid - 1, get_smid());
    unsigned long long mclk;
    if (threadIdx.x == 1) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
        time[get_smid()] = mclk / 1000;
    }
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

    unsigned long long mclk2; 
    if (threadIdx.x == 1) {
        if (smid == 0) {
            for (int i = 0; i < 80; ++i) {
                atomicExch(worker_num + i, 1);
            }
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk2));
        time[get_smid() + 80] = (mclk2) / 1000;
    }
    
}


void run_kernel() {
	int num_streams = 2;
	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}
	
    // allocate kernel_sleep sm
	long long unsigned *h_sm = new long long unsigned[64];
	long long unsigned *d_sm;
	cudaMalloc(&d_sm, 64 * sizeof(long long unsigned));

    long long unsigned *h_sm_ids = new long long unsigned[64 * 2];
	long long unsigned *d_sm_ids;
	cudaMalloc(&d_sm_ids, 64 * sizeof(long long unsigned) * 2);

    // // allocate resource
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
    int *sm_flag = new int[85];
    int *g_sm_flag;
    for (int i = 0; i < 85; ++i) {
        sm_flag[i] = 0;
    }
    cudaMalloc((void **)&g_sm_flag, sizeof(int) * 85);
    cudaMemcpy(g_sm_flag, sm_flag, sizeof(int) * 85, cudaMemcpyHostToDevice);

    int *block_flag = new int[300];
    int *g_block_flag;
    for (int i = 0; i < 300; ++i) {
        block_flag[i] = 0;
    }
    cudaMalloc((void **)&g_block_flag, sizeof(int) * 300);
    cudaMemcpy(g_block_flag, block_flag, sizeof(int) * 300, cudaMemcpyHostToDevice);

    // allocate kernel_sleep sm
	long long unsigned *worker_num = new long long unsigned[80];
    for (int i = 0; i < 40; ++i) {
        worker_num[i] = 1;
        worker_num[80 - i - 1] = 0;
    }
	long long unsigned *d_worker_num;
	cudaMalloc(&d_worker_num, 80 * sizeof(long long unsigned));
    cudaMemcpy(d_worker_num, worker_num, sizeof(long long unsigned) * 80, cudaMemcpyHostToDevice);

    long long unsigned *time = new long long unsigned[200];
	long long unsigned *d_time;
	cudaMalloc(&d_time, 200 * sizeof(long long unsigned));


    // allocate warm flag
    int *flag_warm;
    int *g_flag_warm;
    flag_warm = (int*) malloc(1 * sizeof(int));
    flag_warm[0] = 0;
    cudaMalloc((void **)&g_flag_warm, sizeof(int) * 9000);
    cudaMemcpy(g_flag_warm, flag_warm, sizeof(int) * 9000, cudaMemcpyHostToDevice);

    // cuda launch kernel
	dim3 D_b_a = dim3(200, 1, 1);
	dim3 D_t_a = dim3(128, 1, 1);
    // warm-up
    for (int i = 0; i < 100; ++i) {
        fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel0_warm <<<D_b_a, D_t_a, 0, streams[0]>>>(d_args_55, d_args_76, g_flag_warm, d_sm_ids, d_sm);
    }
	cudaDeviceSynchronize();
    // test kernel
	fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel0 <<<D_b_a, D_t_a, 0, streams[0]>>>(d_args_55, d_args_76, g_sm_flag, d_worker_num, g_block_flag, d_time);
    // sleep until kernel finish
	// fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel1 <<<D_b_b, D_t_b, 0, streams[1]>>>(d_args_56, d_args_76, d_args_75, g_flag, d_sm_ids2, d_sleep_sm, g_sleep_times);
	// kernel_sleep <<<D_b_b, D_t_b, 0, streams[1]>>>(15.6, 64.9, 134.7, 1000, g_flag);
	cudaDeviceSynchronize();

    // cudaMemcpy(h_sm_ids, d_sm_ids, 64 * sizeof(long long unsigned) * 2, cudaMemcpyDeviceToHost);
	cudaMemcpy(time, d_time, 200 * sizeof(long long unsigned), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 80; ++i) {
        printf("sm-%d---start_time:%llu end_time:%llu\n", i, time[i] , time[i + 80]);
    }
	

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
        exit(1);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));

    // cudaDeviceProp  prop;
    // cudaGetDeviceProperties(&prop, 0); 
    // clock_t clock_rate = prop.clockRate;
    // printf("clock_rate:%d\n", clock_rate); // 1530000
	run_kernel();

	return 0;
}

