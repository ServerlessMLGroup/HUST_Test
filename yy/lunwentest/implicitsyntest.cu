#define GPU_RETURN_STATUS(cmd) \
{ \
    CUresult result = cmd; \
    if (result != CUDA_SUCCESS) { \
        std::cout << #cmd " error, return code:" << result << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__device__ int get_task(int* task_pool) {
    int* mutex = task_pool + POOL_MUTEX_BASE, capacity_ = task_pool[POOL_CAPACITY_BASE], *head_ = task_pool + POOL_HEAD_BASE, *tail_ = task_pool + POOL_TAIL_BASE, *tasks_ = task_pool + POOL_ARRAY_BASE;
    unsigned int ns = 10;
    while (atomicCAS(mutex, 0, 1) != 0) {
      // if (ns < 100) {
      //   ns += 10;
      // }
      __nanosleep(ns);
    };
    if ((*head_) == (*tail_)) {
        atomicExch(mutex, 0);
        return -1;
    }
    int task_id = tasks_[(*tail_)];
    (*tail_) = ((*tail_) + 1) % capacity_;
    atomicExch(mutex, 0);
    return task_id;
}

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_2_kernel2(float* __restrict__ bgemm, float* __restrict__ T_add, float* __restrict__ placeholder) {
  float inverse[4];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = (inverse[(3)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f) * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320))]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_add[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner))] = (inverse[(((ax2_inner * 2) + ax3_inner))] + placeholder[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner))]);
    }
  }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
    }
    int gpu_no = atoi(argv[1]);
    log("preate unique_ptr");
    CUcontext ctx;
    CUdevice device;
    CUresult result;
    // init CUDA driver API
    GPU_RETURN_STATUS(cuInit(0));
    GPU_RETURN_STATUS(cuDeviceGet(&device, gpu_no));
    GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device));

    //yy add stream
    CUstream firststream;
    cuStreamCreate(&firststream,0);
    CUstream secondstream;
    cuStreamCreate(&secondstream,0);
    //add fininshed

    std::vector<CUdeviceptr*> kernel_arg;
    //flag
    size_t storage_size1 = 1*sizeof(float);
    CUdeviceptr device_ptr1;
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr1, storage_size1));
    kernel_arg.push_back(&device_ptr1);

    //int block normal
    size_t storage_size2 = 1*sizeof(int);
    CUdeviceptr device_ptr2;
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr2, storage_size2));
    kernel_arg.push_back(&device_ptr2);

    //75
    size_t storage_size3 = 1806336;
    CUdeviceptr device_ptr3;
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr3, storage_size3));
    kernel_arg.push_back(&device_ptr3);

    //29
    size_t storage_size4 = 100352*sizeof(float);
    CUdeviceptr device_ptr4;
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr4, storage_size4));
    kernel_arg.push_back(&device_ptr4);

    //28
    size_t storage_size5 = 128*sizeof(float);
    CUdeviceptr device_ptr5;
    GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr5, storage_size5));
    kernel_arg.push_back(&device_ptr5);

    CUmodule mod;
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx"));
    printf("load cuda kernels!\n");



    cuStreamSynchronize(firststream);
    std::string& func_name = "fused_nn_contrib_conv2d_winograd_without_weight_transform_add_2_kernel2";
    CUfunction kernel;
    GPU_RETURN_STATUS(
        cuModuleGetFunction(&kernel, mod, kernel_info.name.c_str())
    );


    CUfunction func = kernels[func_name];
    uint32_t *launch_params = kernel_info.launch_params;
    GPU_RETURN_STATUS(cuLaunchKernel(func,
    launch_params[0], launch_params[1], launch_params[2],
    launch_params[3], launch_params[4], launch_params[5],
    0, secondstream, (void **)raw_args[j].data(), 0 // raw_args是json中指示的storage的下标
));


    cuStreamSynchronize(firststream);

    return 0;
}

