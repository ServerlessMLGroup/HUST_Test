#include <cuda.h>
#include "cuda_runtime.h"
#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

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

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));
    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    int nstreams = 1;
  
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
  
    if ((deviceProp.concurrentKernels == 0)) {
      printf("> GPU does not support concurrent kernel execution\n");
      printf("  CUDA kernel runs will be serialized\n");
    }
  
    printf("> GPUNo: %d Detected Compute SM %d.%d hardware with %d multi-processors\n",
           cuda_device, deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    
    cudaEvent_t start_event, stop_event;
    CUcontext ctx;
    CUdevice device;
    CUresult result;
    // init CUDA driver API
    GPU_RETURN_STATUS(cuInit(0));
    GPU_RETURN_STATUS(cuDeviceGet(&device, cuda_device));
    GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device));
    CUmodule mod;
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/concurrentKernel/float.ptx"));
    printf("load cuda mod!\n");

    CUfunction kernel;
    GPU_RETURN_STATUS(
        cuModuleGetFunction(&kernel, mod, "matrixMulCpu")
    );
    printf("load cuda kernels!\n");
    
    // allocate host memory
    std::vector<CUdeviceptr*> args;
    size_t storage_size;
    CUdeviceptr device_ptr;
    std::vector<char> temp;

    storage_size = 80 * sizeof(int);
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size)); // sm
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr, temp.data(), storage_size));
    args.push_back(&device_ptr);

    // warm up
    for (int i = 0; i < 10; ++i) {
      GPU_RETURN_STATUS(cuLaunchKernel(kernel,
      1, 1, 80,
      1, 32, 32,
      0, 0 // stream
      , (void **)args.data(), 0 // raw_args是json中指示的storage的下标
      ));
    }
    GPU_RETURN_STATUS(cuCtxSynchronize());

    CUstream stream;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));
    cudaEventRecord(start_event, 0);
    GPU_RETURN_STATUS(cuLaunchKernel(kernel,
      1, 1, 80,
      1, 16, 16,
      0, 0 // stream
      , (void **)args.data(), 0 // raw_args是json中指示的storage的下标
    ));
    float elapsed_time;
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event)); // Waits until the completion of all work currently captured in event
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    printf("Measured time for sample = %.3fms\n", elapsed_time);
    std::vector<int>output(80);
    GPU_RETURN_STATUS(cuMemcpyDtoH(
        output.data(), (CUdeviceptr)*args[0], sizeof(int) * 80
    ));
    for (int i = 0; i < 80; ++i) {
      printf("%d %d\n",i,output[i]);
    }

    std::cout << "End!" << std::endl;
    return 0;


}