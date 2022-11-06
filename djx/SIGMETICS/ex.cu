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

__global__ void kernel(float n1, float n2, float n3, long long unsigned *times, int stop) {
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

__global__ void kernel_sleep(float n1, float n2, float n3, long long unsigned *times, int stop) {
	#if __CUDA_ARCH__ >= 700
	for (int i = 0; i < 1000; i++)
		__nanosleep(1000000); // 10ms
	#else
	printf(">>> __CUDA_ARCH__ !\n");
	#endif
	unsigned long long mclk; 
	if (threadIdx.x == 0) {
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
	}
	// #if __CUDA_ARCH__ >= 700
	// for (int i = 0; i < 1000; i++)
	// 	__nanosleep(100000000); // 100ms
	// #else
	// printf(">>> __CUDA_ARCH__ !\n");
	// #endif

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
	
	long long unsigned *h_sm_ids = new long long unsigned[a_blocks];
	long long unsigned *d_sm_ids;
	cudaMalloc(&d_sm_ids, a_blocks * sizeof(long long unsigned));
	
	long long unsigned *h_sm_ids2 = new long long unsigned[b_blocks];
	long long unsigned *d_sm_ids2;
	cudaMalloc(&d_sm_ids2, b_blocks * sizeof(long long unsigned));

	dim3 Dba = dim3(a_threads);
	dim3 Dga = dim3(a_blocks,1,1);
	dim3 Dbb = dim3(b_threads);
	dim3 Dgb = dim3(b_blocks,1,1);
	kernel <<<Dga, Dba, 0, streams[0]>>>(15.6, 64.9, 134.7, d_sm_ids, 5000000);
	cudaDeviceSynchronize();
	kernel <<<Dga, Dba, 0, streams[0]>>>(15.6, 64.9, 134.7, d_sm_ids, 5000000);
	kernel_sleep <<<Dgb, Dbb, 0, streams[1]>>>(15.6, 64.9, 134.7, d_sm_ids2, 5000000);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(h_sm_ids, d_sm_ids, a_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sm_ids2, d_sm_ids2, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);

    printf("---1---\n");
	for (int i = 0; i < a_blocks; i++) {
		printf("%llu\n", h_sm_ids[i]);
	}
	printf("---2---\n");
	for (int i = 0; i < b_blocks; i++) {
		printf("%llu\n", h_sm_ids2[i]);
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

