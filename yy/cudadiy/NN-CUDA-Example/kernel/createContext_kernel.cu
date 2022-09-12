#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
void crContext(int dev) {
	
    cuInit(0);
    // Get handle for device 0
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, dev);

    // Create context
    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);
}
