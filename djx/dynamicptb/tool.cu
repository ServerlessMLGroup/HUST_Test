#include<cuda.h>
#include<cuda_runtime.h>
#include<stdlib.h>
#include <stdio.h>

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

void initDevice() {
    if (argc < 2) {
      printf("args num error! argc:%d", argc);
      exit(1);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));
}

__device__ uint get_smid(void) {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}
