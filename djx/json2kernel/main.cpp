#include "model.h"
#include "log.h"
#include <bits/unique_ptr.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define GPU_RETURN_STATUS(cmd) \
{\
    CUresult error = cmd;\
    if (error != CUDA_SUCCESS) {\
        LOG(ERROR) << "cuda error: " << cudaGetErrorString(error) << "at " << __FILE__ << ":" << __LINE__; \
        return Status::Fail;\
    }\
}

int main() {
    std::unique_ptr<Model> model;
    log("preate unique_ptr");
    model.reset(Model::from_json("/home/husterdjx/research/HUST_Test/djx/json2kernel/resnet18-final.json"));
    // 2. load hip kernels
    for (KernelInfo &kernel_info : model->kernels) {
        CUfunction kernel;
        GPU_RETURN_STATUS(
            GPUModuleGetFunction(&kernel, mod, kernel_info.name.c_str())
        );
        kernels.emplace(kernel_info.name, kernel);
    }
    // 3. allocate device storage
    for (StorageInfo &storage_info : model->storage) {
        size_t stype_size = Model::get_stype_size(storage_info.stype);
        size_t storage_size = stype_size * storage_info.size;
        GPUDevicePtr_t device_ptr;
        std::vector<char> temp;
        temp.resize(storage_size, 0);
        GPU_RETURN_STATUS(GPUMalloc((GPUDevicePtr_t*)&device_ptr, storage_size));
        GPU_RETURN_STATUS(GPUMemcpyHtoD(device_ptr, temp.data(), storage_size));
        storage.push_back(device_ptr);
    }
    model.reset();
    return 0;
}