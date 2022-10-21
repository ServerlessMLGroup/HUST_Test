#include <cuda.h>
#include "cuda_runtime.h"
#include "kernel.cu"

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    cuGetErrorString(err, &errorStr);
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
    
    // allocate host memory
    float *arg52 = 0; 
    checkCudaErrors(cudaMallocHost((void **)&arg52, 50176 * sizeof(float)));
    
    float *arg53 = 0; 
    checkCudaErrors(cudaMallocHost((void **)&arg53, 1179648 * sizeof(float)));

    float *arg55 = 0; 
    checkCudaErrors(cudaMallocHost((void **)&arg55, 25088 * sizeof(float)));

    float *arg54 = 0; 
    checkCudaErrors(cudaMallocHost((void **)&arg54, 512 * sizeof(float)));


    float *d_arg52 = 0; 
    checkCudaErrors(cudaMalloc((void **)&arg52, 50176 * sizeof(float)));
    
    float *d_arg53 = 0; 
    checkCudaErrors(cudaMalloc((void **)&arg53, 1179648 * sizeof(float)));

    float *d_arg55 = 0; 
    checkCudaErrors(cudaMalloc((void **)&arg55, 25088 * sizeof(float)));

    float *d_arg54 = 0; 
    checkCudaErrors(cudaMalloc((void **)&arg54, 512 * sizeof(float)));

    // allocate and initialize an array of stream handles
    cudaStream_t *streams =
     (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    
    for (int i = 0; i < nstreams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    // create CUDA event handles
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    checkCudaErrors(cudaEventRecord(start_event, streams[0]));

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop_event, streams[0]));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop_event));


}