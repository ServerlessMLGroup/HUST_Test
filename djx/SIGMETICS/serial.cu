#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

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

__global__ void kernel(float n1, float n2, float n3, long long unsigned *times, int stop, int* flag) {
	// if (threadIdx.x == 0) {
	// 	int sm = get_smid();
	// 	printf("kernel-sm:%d\n", sm);
	// }
	unsigned long long mclk; 
	if (threadIdx.x == 0) {
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
	}

	for (int i = 0; i < stop; i++) {
		n1=sinf(n1);
		n2=n3/n2;
	}
	__syncthreads();
	// flag[0] = 1在此在ms级别无变化
	if (threadIdx.x == 0) {
		unsigned long long mclk2;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk2));
		times[blockIdx.x] = (mclk2 - mclk) / 1000000;
	}
    flag[0] = 1;
}

__global__ void kernel_sleep(float n1, float n2, float n3, long long unsigned *times, int stop, int* flag, long long unsigned * sleep_time, long long unsigned * sleep_sm) {
	if (threadIdx.x == 0) {
		sleep_sm[blockIdx.x] = get_smid();
	}
    #if __CUDA_ARCH__ >= 700
	while(flag[0] != 1) {
		if (threadIdx.x == 0)
			sleep_time[blockIdx.x]++;
		__nanosleep(1000000); // 1ms
	}
	#else
	printf(">>> __CUDA_ARCH__ !\n");
	#endif
	// __syncthreads(); // 如果加入这一行，那么稳定出现20ms计算延迟
    unsigned long long mclk; 
	if (threadIdx.x == 0) {
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
	}
	for (int i = 0; i < stop; i++) {
		n1=sinf(n1);
		n2=n3/n2;
	}
	
	__syncthreads();
	
	if (threadIdx.x == 0) {
		unsigned long long mclk2;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk2));
		times[blockIdx.x] = (mclk2 - mclk) / 1000000;
	}
}

void run_kernel(int a_blocks, int b_blocks, int a_threads, int b_threads) {
	int num_streams = 2;
	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}
	
    cudaEvent_t all_start_event, all_end_event;

    // allocate resource
	long long unsigned *h_sm_ids = new long long unsigned[a_blocks];
	long long unsigned *d_sm_ids;
	cudaMalloc(&d_sm_ids, a_blocks * sizeof(long long unsigned));
	
	long long unsigned *h_sm_ids2 = new long long unsigned[b_blocks];
	long long unsigned *d_sm_ids2;
	cudaMalloc(&d_sm_ids2, b_blocks * sizeof(long long unsigned));

    // allocate flag
    int *flag;
    int *g_flag;
    flag = (int*) malloc(1 * sizeof(int));
    flag[0] = 0;
    cudaMalloc((void **)&g_flag, sizeof(int) * 1);
    cudaMemcpy(g_flag, flag, sizeof(int) * 1, cudaMemcpyHostToDevice);

	// allocate sleep_time
	long long unsigned *h_sleep_time = new long long unsigned[b_blocks];
	long long unsigned *d_sleep_time;
	cudaMalloc(&d_sleep_time, b_blocks * sizeof(long long unsigned));

	// allocate kernel_sleep sm
	long long unsigned *h_sleep_sm = new long long unsigned[b_blocks];
	long long unsigned *d_sleep_sm;
	cudaMalloc(&d_sleep_sm, b_blocks * sizeof(long long unsigned));


    // allocate warm flag
    int *flag_warm;
    int *g_flag_warm;
    flag_warm = (int*) malloc(1 * sizeof(int));
    flag_warm[0] = 0;
    cudaMalloc((void **)&g_flag_warm, sizeof(int) * 1);
    cudaMemcpy(g_flag_warm, flag_warm, sizeof(int) * 1, cudaMemcpyHostToDevice);

    // cuda launch kernel
	dim3 Dba = dim3(a_threads);
	dim3 Dga = dim3(a_blocks,1,1);
	dim3 Dbb = dim3(b_threads);
	dim3 Dgb = dim3(b_blocks,1,1);
    checkCudaErrors(cudaEventCreate(&all_start_event));
    checkCudaErrors(cudaEventCreate(&all_end_event));
    // warm-up
    for (int i = 0; i < 10; ++i) {
        kernel <<<Dga, Dba, 0, streams[0]>>>(15.6, 64.9, 134.7, d_sm_ids, 5000000, g_flag_warm);
    }
	cudaDeviceSynchronize();
    checkCudaErrors(cudaEventRecord(all_start_event, 0));
    // 顺序执行，同一个stream下
	kernel <<<Dga, Dba, 0, streams[0]>>>(15.6, 64.9, 134.7, d_sm_ids, 5000000, g_flag);
	kernel_sleep <<<Dgb, Dbb, 0, streams[0]>>>(15.6, 64.9, 134.7, d_sm_ids2, 5000000, g_flag, d_sleep_time, d_sleep_sm);

    checkCudaErrors(cudaEventRecord(all_end_event, 0));

    checkCudaErrors(cudaEventSynchronize(all_end_event));

    float elapsed;
    checkCudaErrors(cudaEventElapsedTime(&elapsed, all_start_event, all_end_event));
    printf("Total GPU Measured time for sample = %.3fms\n", elapsed); 
	
	cudaMemcpy(h_sm_ids, d_sm_ids, a_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sm_ids2, d_sm_ids2, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_sleep_time, d_sleep_time, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sleep_sm, d_sleep_sm, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);

    printf("---1---\n");
	for (int i = 0; i < a_blocks; i++) {
		printf("%llu\n", h_sm_ids[i]);
	}
	printf("---2---\n");
	for (int i = 0; i < b_blocks; i++) {
		printf("%llu\n", h_sm_ids2[i]);
	}
	printf("---sleep_times---\n");
	for (int i = 0; i < b_blocks; i++) {
		printf("block-%d : %llu\n", i, h_sleep_time[i]);
	}
	printf("---sleep_sm---\n");
	for (int i = 0; i < b_blocks; ++i) {
		printf("block-%d : %llu\n", i, h_sleep_sm[i]);
	}
	
	cudaFree(d_sm_ids);
	cudaFree(d_sm_ids2);

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
        exit(1);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));
	run_kernel(80, 80, 512, 512);

	return 0;
}

