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
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18_sm.ptx"));
    printf("load cuda mod!\n");

    CUfunction kernel;
    GPU_RETURN_STATUS(
        cuModuleGetFunction(&kernel, mod, "fused_nn_conv2d_add_nn_relu_kernel0")
    );
    printf("load cuda kernels!\n");
    
    // allocate host memory
    std::vector<CUdeviceptr*> args;
    size_t storage_size = 50176 * sizeof(float);
    CUdeviceptr device_ptr1, device_ptr2, device_ptr3, device_ptr4, device_ptr5;
    std::vector<char> temp;
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr1, storage_size)); // 52
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr1, temp.data(), storage_size)); 
    args.push_back(&device_ptr1);

    storage_size = 1179648 * sizeof(float);
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr2, storage_size)); // 53
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr2, temp.data(), storage_size));
    args.push_back(&device_ptr2);

    storage_size = 25088 * sizeof(float);
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr3, storage_size)); // 55, ouput
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr3, temp.data(), storage_size));
    args.push_back(&device_ptr3);

    storage_size = 512 * sizeof(float);
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr4, storage_size)); // 54
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr4, temp.data(), storage_size));
    args.push_back(&device_ptr4);

    storage_size = 80 * sizeof(int);
    temp.resize(storage_size, 0);
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr5, storage_size)); // sm
    GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr5, temp.data(), storage_size));
    args.push_back(&device_ptr5);

    // warm up
    for (int i = 0; i < 10; ++i) {
      GPU_RETURN_STATUS(cuLaunchKernel(kernel,
      1, 5, 20,
      7, 1, 16,
      0, 0 // stream
      , (void **)args.data(), 0 // raw_args是json中指示的storage的下标
      ));
    }
    GPU_RETURN_STATUS(cuCtxSynchronize());

    std::vector<float> input52(50176); // test -10000 == fail
    for (size_t i = 0; i < 50176; i++)
        input52[i] = 10.0;
    std::vector<float> input53(1179648);
    for (size_t i = 0; i < 1179648; i++)
        input53[i] = 10.0;
    std::vector<float> input54(512);
    for (size_t i = 0; i < 512; i++)
        input54[i] = 10.0;
    
    GPU_RETURN_STATUS(cuMemcpyHtoD(
      (CUdeviceptr)device_ptr1, input52.data(), input52.size() * sizeof(float)
    ))
    GPU_RETURN_STATUS(cuMemcpyHtoD(
      (CUdeviceptr)device_ptr2, input53.data(), input53.size() * sizeof(float)
    ))
    GPU_RETURN_STATUS(cuMemcpyHtoD(
      (CUdeviceptr)device_ptr4, input54.data(), input54.size() * sizeof(float)
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
    CUstream stream;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));
    cudaEventRecord(start_event, 0);
    GPU_RETURN_STATUS(cuLaunchKernel(kernel,
      1, 7, 32,
      7, 1, 16,
      0, 0 // stream
      , (void **)args.data(), 0 // raw_args是json中指示的storage的下标
    ));
    float elapsed_time;
    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event)); // Waits until the completion of all work currently captured in event
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    printf("Measured time for sample = %.3fms\n", elapsed_time);
    std::vector<int>output(80);
    // checkCudaErrors(cudaMemcpyAsync(
    //   output.data(), *args[2], sizeof(float) * 25088, cudaMemcpyDeviceToHost, 0));

    GPU_RETURN_STATUS(cuMemcpyDtoH(
        output.data(), (CUdeviceptr)*args[4], sizeof(int) * 80
    ));
    std::vector<float> ans = {102410, 153610, 153610, 153610, 153610, 153610, 153610, 153610, 230410, 230410, 230410, 230410, 230410, 230410,
     153610, 230410, 230410, 230410, 230410, 230410};
    for (int i = 0; i < 80; ++i) {
      //if (ans[i] != output[i]) std::cout << "ans:" <<ans[i] << " VS "<<"output:" << output[i] <<std::endl;
      printf("%d %d\n",i,output[i]);
    }

    std::cout << "End!" << std::endl;
    
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