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
  
    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
    
    cudaEvent_t start_event[nstreams], stop_event[nstreams], all_start_event, all_end_event;

    CUcontext ctx;
    CUdevice device;
    CUresult result;
    // init CUDA driver API
    GPU_RETURN_STATUS(cuInit(0));
    GPU_RETURN_STATUS(cuDeviceGet(&device, 0));
    GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device));

    // allocate and initialize an array of stream handles
    cudaStream_t *streams =
      (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    for (int i = 0; i < nstreams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }
    CUmodule mod;
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx"));
    printf("load cuda mod!\n");
    CUfunction kernel[nstreams];
    std::vector<CUdeviceptr*> args[nstreams];
    CUdeviceptr device_ptr1[nstreams], device_ptr2[nstreams], device_ptr3[nstreams], device_ptr4[nstreams];
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
    for (int times = 0; times < nstreams; ++times) {
        checkCudaErrors(cudaEventCreate(&start_event[times]));
        checkCudaErrors(cudaEventCreate(&stop_event[times]));
    }
    checkCudaErrors(cudaEventCreate(&all_start_event));
    checkCudaErrors(cudaEventCreate(&all_end_event));
    checkCudaErrors(cudaEventRecord(all_start_event, 0));
    float elapsed_time[nstreams];
    for (int times = 0; times < nkernels; ++times) {
        cudaStream_t  stream  = streams[times];

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
        cudaEventRecord(start_event[times], stream);
        GPU_RETURN_STATUS(cuLaunchKernel(kernel,
            1, 7, 32,
            7, 1, 16,
            0, stream // stream
            , (void **)args[times].data(), 0 // raw_args是json中指示的storage的下标
        ));
        checkCudaErrors(cudaEventRecord(stop_event[times], stream));
    }
    for (int i = 0; i < nstreams; ++i) {
        checkCudaErrors(cudaEventSynchronize(stop_event[i])); // Waits until the completion of all work currently captured in event
    }
    checkCudaErrors(cudaEventRecord(all_end_event, 0));
    for (int i = 0; i < nstreams; ++i) {
        checkCudaErrors(cudaEventElapsedTime(&elapsed_time[i], start_evente[i], stop_evente[i]));
        printf("Stream%d Measured time for sample = %.3fms\n", i, elapsed_time[i]);
    }
    float elapsed;
    checkCudaErrors(cudaEventElapsedTime(&elapsed, all_start_event, all_end_event));
    printf("Total GPU Measured time for sample = %.3fms\n", elapsed); 
    for (int times = 0; times < nstreams; ++times) {
        std::vector<float>output(20);
        GPU_RETURN_STATUS(cuMemcpyDtoH(
            output.data(), (CUdeviceptr)*args[times][2], sizeof(float) * 20
        ));
        std::vector<float> ans = {102410, 153610, 153610, 153610, 153610, 153610, 153610, 153610, 230410, 230410, 230410, 230410, 230410, 230410,
        153610, 230410, 230410, 230410, 230410, 230410};
        for (int i = 0; i < 20; ++i) {
            if (ans[i] != output[i]) std::cout << "ans:" <<ans[i] << " VS "<<"output:" << output[i] <<std::endl;
        }
    }
    std::cout << "End!" << std::endl;
    return 0;
}