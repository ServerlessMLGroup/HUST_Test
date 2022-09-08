void crContext(int dev) {
     // Get handle for device 0
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, dev);

    // Create context
    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);
}