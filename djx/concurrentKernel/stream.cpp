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
    //int nkernels = 2;
    int nstreams = 2;
  
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
  
    if ((deviceProp.concurrentKernels == 0)) {
      printf("> GPU does not support concurrent kernel execution\n");
      printf("  CUDA kernel runs will be serialized\n");
    }
  
    printf("> GPUNo%d Detected Compute SM %d.%d hardware with %d multi-processors\n",
           cuda_device, deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    
    cudaEvent_t start_event[nstreams], stop_event[nstreams], all_start_event, all_end_event;

    CUcontext ctx;
    CUdevice device;
    CUresult result;
    // init CUDA driver API
    GPU_RETURN_STATUS(cuInit(0));
    GPU_RETURN_STATUS(cuDeviceGet(&device, cuda_device));
    GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device));

    int priority_high, priority_low;
    checkCudaErrors(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));

    // allocate and initialize an array of stream handles
    cudaStream_t *streams =
      (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    // 优先级
    checkCudaErrors(cudaStreamCreateWithPriority(&(streams[0]), cudaStreamNonBlocking, priority_high));
    checkCudaErrors(cudaStreamCreateWithPriority(&(streams[1]), cudaStreamNonBlocking, priority_low));
    CUmodule mod;
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx"));
    printf("load cuda mod!\n");
    CUfunction kernel[nstreams];
    std::vector<CUdeviceptr*> args[nstreams];
    CUdeviceptr device_ptr1[nstreams], device_ptr2[nstreams], device_ptr3[nstreams], device_ptr4[nstreams];
    printf("begin alloc storage....\n");
    for (int i = 0; i < nstreams; ++i) {
        GPU_RETURN_STATUS(
            cuModuleGetFunction(&kernel[i], mod, "fused_nn_conv2d_add_nn_relu_kernel0")
        );

        // allocate host memory
        size_t storage_size = 50176 * sizeof(float);
        std::vector<char> temp;
        temp.resize(storage_size, 0);
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr1[i], storage_size)); // 52
        GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr1[i], temp.data(), storage_size)); 
        args[i].push_back(&device_ptr1[i]);

        storage_size = 1179648 * sizeof(float);
        temp.resize(storage_size, 0);
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr2[i], storage_size)); // 53
        GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr2[i], temp.data(), storage_size));
        args[i].push_back(&device_ptr2[i]);

        storage_size = 25088 * sizeof(float);
        temp.resize(storage_size, 0);
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr3[i], storage_size)); // 55, ouput
        GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr3[i], temp.data(), storage_size));
        args[i].push_back(&device_ptr3[i]);

        storage_size = 512 * sizeof(float);
        temp.resize(storage_size, 0);
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr4[i], storage_size)); // 54
        GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr4[i], temp.data(), storage_size));
        args[i].push_back(&device_ptr4[i]);

        // // warm up
        // for (int ii = 0; ii < 10; ++ii) {
        //     GPU_RETURN_STATUS(cuLaunchKernel(kernel[i],
        //         1, 3, 28,
        //         7, 1, 16,
        //         0, 0 // stream
        //         , (void **)args[i].data(), 0 // raw_args是json中指示的storage的下标
        //     ));
        // }
        // GPU_RETURN_STATUS(cuCtxSynchronize());

        std::vector<float> input52(50176);
        for (size_t ii = 0; ii < 50176; ii++)
            input52[ii] = 10.0;
        std::vector<float> input53(1179648);
        for (size_t ii = 0; ii < 1179648; ii++)
            input53[ii] = 10.0;
        std::vector<float> input54(512);
        for (size_t ii = 0; ii < 512; ii++)
            input54[ii] = 10.0;
        
        GPU_RETURN_STATUS(cuMemcpyHtoD(
        (CUdeviceptr)device_ptr1[i], input52.data(), input52.size() * sizeof(float)
        ))
        GPU_RETURN_STATUS(cuMemcpyHtoD(
        (CUdeviceptr)device_ptr2[i], input53.data(), input53.size() * sizeof(float)
        ))
        GPU_RETURN_STATUS(cuMemcpyHtoD(
        (CUdeviceptr)device_ptr4[i], input54.data(), input54.size() * sizeof(float)
        ))
    }
    GPU_RETURN_STATUS(cuCtxSynchronize());
    printf("end alloc storage\n");
    for (int times = 0; times < nstreams; ++times) {
        checkCudaErrors(cudaEventCreate(&start_event[times]));
        checkCudaErrors(cudaEventCreate(&stop_event[times]));
    }
    checkCudaErrors(cudaEventCreate(&all_start_event));
    checkCudaErrors(cudaEventCreate(&all_end_event));
    checkCudaErrors(cudaEventRecord(all_start_event, 0));
    float elapsed_time[nstreams];
    for (int times = 0; times < nstreams; ++times) {
        cudaStream_t  stream  = streams[times];
        cudaEventRecord(start_event[times], stream);
        // printf("launch kernel %d\n", times);
        GPU_RETURN_STATUS(cuLaunchKernel(kernel[times],
            1, 7, 32,
            7, 1, 16,
            0, stream // stream
            , (void **)args[times].data(), 0 // raw_args是json中指示的storage的下标
        ));
        checkCudaErrors(cudaEventRecord(stop_event[times], stream));
    }
    checkCudaErrors(cudaEventRecord(all_end_event, 0));
    for (int i = 0; i < nstreams; ++i) {
        checkCudaErrors(cudaEventSynchronize(stop_event[i])); // Waits until the completion of all work currently captured in event
    }
    checkCudaErrors(cudaEventSynchronize(all_end_event));
    for (int i = 0; i < nstreams; ++i) {
        checkCudaErrors(cudaEventElapsedTime(&elapsed_time[i], start_event[i], stop_event[i]));
        printf("Stream%d Measured time for sample = %.3fms\n", i, elapsed_time[i]);
    }
    float elapsed;
    checkCudaErrors(cudaEventElapsedTime(&elapsed, all_start_event, all_end_event));
    printf("Total GPU Measured time for sample = %.3fms\n", elapsed); 

    std::cout << "End!" << std::endl;
    return 0;
}