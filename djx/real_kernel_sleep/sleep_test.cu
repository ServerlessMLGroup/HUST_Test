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


__global__ void kernel_sleep_warm() {
    #if __CUDA_ARCH__ >= 700
    for (int i = 0; i < 100; ++i) {
        __nanosleep(1000); // 1us
    }
    #else
    printf(">>> __CUDA_ARCH__ !\n");
    #endif
}

__global__ void kernel_sleep0() {
    #if __CUDA_ARCH__ >= 700
    for (int i = 0; i < 100; ++i) {
        __nanosleep(1000); // 1us
    }
    #else
    printf(">>> __CUDA_ARCH__ !\n");
    #endif
}

__global__ void kernel_sleep1() {
    #if __CUDA_ARCH__ >= 700
    for (int i = 0; i < 100; ++i) {
        __nanosleep(1000); // 1us
    }
    #else
    printf(">>> __CUDA_ARCH__ !\n");
    #endif
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
        exit(1);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));

    int num_streams = 2;
	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}

    dim3 D_b_a = dim3(64, 1, 1);
	dim3 D_t_a = dim3(128, 1, 1);
	dim3 D_b_b = dim3(1, 8, 16);
	dim3 D_t_b = dim3(8, 16, 1);
    // warm-up
    for (int i = 0; i < 100; ++i) {
        kernel_sleep_warm <<<D_b_a, D_t_a, 0, streams[0]>>>();
    }
    cudaDeviceSynchronize();
	kernel_sleep0<<<D_b_a, D_t_a, 0, streams[0]>>>();
    kernel_sleep1<<<D_b_a, D_t_a, 0, streams[1]>>>();
    cudaDeviceSynchronize();
	return 0;
}