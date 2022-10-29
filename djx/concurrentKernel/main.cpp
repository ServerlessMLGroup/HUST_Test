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
        std::cout << #cmd " error, return code:" << result << __FILE__ << ":" << __LINE__ << std::endl; \
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
    int cuda_device = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    int nstreams = 1;
  
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
  
    if ((deviceProp.concurrentKernels == 0)) {
      printf("> GPU does not support concurrent kernel execution\n");
      printf("  CUDA kernel runs will be serialized\n");
    }
  
    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    
    cudaEvent_t start_event, stop_event;
    CUcontext ctx;
    CUdevice device;
    CUresult result;
    // init CUDA driver API
    GPU_RETURN_STATUS(cuInit(0));
    GPU_RETURN_STATUS(cuDeviceGet(&device, 0));
    GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device));
    CUmodule mod;
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx"));
    printf("load cuda mod!\n");

    CUfunction kernel;
    GPU_RETURN_STATUS(
        cuModuleGetFunction(&kernel, mod, "fused_nn_conv2d_add_nn_relu_kernel0")
    );
    printf("load cuda kernels!\n");
    
    // allocate host memory
    std::vector<CUdeviceptr*> args;
    size_t storage_size = 50176 * sizeof(float);
    CUdeviceptr device_ptr;
    std::vector<char> temp;
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size)); // 52
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr, temp.data(), storage_size)); 
    args.push_back(&device_ptr);

    storage_size = 1179648 * sizeof(float);
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size)); // 53
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr, temp.data(), storage_size));
    args.push_back(&device_ptr);

    storage_size = 25088 * sizeof(float);
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size)); // 55, ouput
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr, temp.data(), storage_size));
    args.push_back(&device_ptr);

    storage_size = 512 * sizeof(float);
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size)); // 54
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr, temp.data(), storage_size));
    args.push_back(&device_ptr);

    std::vector<float> input52(50176);
    for (size_t i = 0; i < 50176; i++)
        input52[i] = 10.0;
    std::vector<float> input53(1179648);
    for (size_t i = 0; i < 1179648; i++)
        input53[i] = 10.0;
    std::vector<float> input54(512);
    for (size_t i = 0; i < 512; i++)
        input54[i] = 10.0;
    
    GPU_RETURN_STATUS(cuMemcpyHtoD(
      (CUdeviceptr)*args[0], (void*)input52.data(), 50176 * sizeof(float)
    ))
    GPU_RETURN_STATUS(cuMemcpyHtoD(
      (CUdeviceptr)*args[1], (void*)input53.data(), 1179648 * sizeof(float)
    ))
    GPU_RETURN_STATUS(cuMemcpyHtoD(
      (CUdeviceptr)*args[3], (void*)input54.data(), 512 * sizeof(float)
    ))

    // fused_nn_conv2d_add_nn_relu_kernel0<<<224, 112, 0, 0>>>(args[0], args[1], args[2], args[3]);
    //   {
    //     "name": "fused_nn_conv2d_add_nn_relu_kernel0",
    //     "launch_params": [
    //         1,
    //         7,
    //         32,
    //         7,
    //         1,
    //         16
    //     ],
    //     "args": [
    //         52,
    //         53,
    //         55,
    //         54
    //     ]
    // },
    GPU_RETURN_STATUS(cuLaunchKernel(kernel,
      1, 7, 32,
      7, 1, 16,
      0, 0 // stream
      , (void **)args.data(), 0 // raw_args是json中指示的storage的下标
    ));
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    std::vector<float>output(25088);
    // checkCudaErrors(cudaMemcpyAsync(
    //   output.data(), *args[2], sizeof(float) * 25088, cudaMemcpyDeviceToHost, 0));

    GPU_RETURN_STATUS(cuMemcpyDtoH(
        output.data(), (CUdeviceptr)*args[2], sizeof(float) * 25088
    ));
    
    for (auto i : output) {
      std::cout << i << " ";
    }

    std::cout << std::endl;
    
    // float *arg53 = 0; 
    // checkCudaErrors(cudaMallocHost((void **)&arg53, 1179648 * sizeof(float)));

    // float *arg55 = 0; 
    // checkCudaErrors(cudaMallocHost((void **)&arg55, 25088 * sizeof(float)));

    // float *arg54 = 0; 
    // checkCudaErrors(cudaMallocHost((void **)&arg54, 512 * sizeof(float)));


    // float *d_arg52 = 0; 
    // checkCudaErrors(cudaMalloc((void **)&arg52, 50176 * sizeof(float)));
    
    // float *d_arg53 = 0; 
    // checkCudaErrors(cudaMalloc((void **)&arg53, 1179648 * sizeof(float)));

    // float *d_arg55 = 0; 
    // checkCudaErrors(cudaMalloc((void **)&arg55, 25088 * sizeof(float)));

    // float *d_arg54 = 0; 
    // checkCudaErrors(cudaMalloc((void **)&arg54, 512 * sizeof(float)));

    // allocate and initialize an array of stream handles


    // *****
    // cudaStream_t *streams =
    //  (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    
    // for (int i = 0; i < nstreams; i++) {
    //     checkCudaErrors(cudaStreamCreate(&(streams[i])));
    // }

    // // create CUDA event handles
    // cudaEvent_t start_event, stop_event;
    // checkCudaErrors(cudaEventCreate(&start_event));
    // checkCudaErrors(cudaEventCreate(&stop_event));

    // checkCudaErrors(cudaEventRecord(start_event, streams[0]));

    // // Record the stop event
    // checkCudaErrors(cudaEventRecord(stop_event, streams[0]));

    // // Wait for the stop event to complete
    // checkCudaErrors(cudaEventSynchronize(stop_event));
    return 0;


}