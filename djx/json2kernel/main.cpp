#include "model.h"
#include "log.h"
#include <bits/unique_ptr.h>
#include <cuda.h>
#include "cuda_runtime.h"
// #include <glog/logging.h>

enum Status {
    Succ,
    Fail,
    NotFound,
    OutOfRange,
    Full
};

#define GPU_RETURN_STATUS(cmd) \
{\
    CUresult error = cmd;\
    if (error != CUDA_SUCCESS) {\
        const char** meg_ptr; \
        cuGetErrorString(error, meg_ptr); \
        std::cout << "cuda error: " << meg_ptr << "at " << __FILE__ << ":" << __LINE__; \
        return Status::Fail;\
    }\
}

std::vector<CUdeviceptr> storage;
std::unordered_map<std::string, CUfunction> kernels;
std::vector<std::vector<CUdeviceptr*>> raw_args;
std::unique_ptr<Model> model;
int main() {
    log("preate unique_ptr");
    model.reset(Model::from_json("/home/husterdjx/research/HUST_Test/djx/json2kernel/resource/resnet18-final.json"));
    // CUcontext ctx;
    CUdevice device;
    GPU_RETURN_STATUS(cuDeviceGet(&device, 0));
    // GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device));
    CUmodule mod;
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/husterdjx/research/HUST_Test/djx/json2kernel/resource/resnet18.ptx"));
    // 2. load cuda kernels
    for (KernelInfo &kernel_info : model->kernels) {
        CUfunction kernel;
        GPU_RETURN_STATUS(
            cudaModuleGetFunction(&kernel, mod, kernel_info.name.c_str())
        );
        kernels.emplace(kernel_info.name, kernel);
    }
    // 3. allocate device storage
    for (StorageInfo &storage_info : model->storage) {
        size_t stype_size = Model::get_stype_size(storage_info.stype);
        size_t storage_size = stype_size * storage_info.size;
        CUdeviceptr device_ptr;
        std::vector<char> temp;
        temp.resize(storage_size, 0);
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size));
        GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr, temp.data(), storage_size));
        storage.push_back(device_ptr);
    }

    raw_args.reserve(model->kernels.size());
    for (KernelInfo &kernel_info : model->kernels) {
        std::vector<CUdeviceptr*> kernel_arg;
        for (size_t arg_idx : kernel_info.args) {
            // assert(arg_idx < storage.size());
            kernel_arg.push_back(&storage[arg_idx]);
        }
        raw_args.push_back(kernel_arg);
    }

    std::unique_ptr<ModelParam> params(ModelParamParser::parse_from_file(param_file_path));
    for (size_t i = 0; i < storage.size(); i++) {
        StorageInfo& storage_info = model->storage[i];
        if (params->find(storage_info.name) == params->end()) 
            continue;
        auto &array = params->at(storage_info.name);
        GPU_RETURN_STATUS(cuMemcpyHtoD(
            (CUdeviceptr)storage[i], array.data(), 
            array.size() * sizeof(float))); 
    }

    model.reset();
    return 0;
}

// size_t ExecutorBase::num_kernels() const {
//     return model->kernels.size();
// }


// void ExecutorBase::set_stream(GPUStream_t stream) {
//     s = stream;
// }


// GPUStream_t ExecutorBase::stream() const {
//     return s;
// }

Status execute(CUstream stream, Model* model) {
    execute_to(model->kernels.size(), stream);
    return Status::Succ;
}

Status execute_to(int idx, CUstream stream, Model* model) {
    for (int i = 0; i < idx; i++) {
        GPU_RETURN_STATUS(launch_kernel(i, stream, model));
    }  
    GPU_RETURN_STATUS(cuStreamSynchronize(stream));
    return Status::Succ;
}

// Status execute_kernel(int idx, GPUStream_t stream) {
//     if (idx >= num_kernels()) RETURN_STATUS(Status::OutOfRange);
//     GPU_RETURN_STATUS(launch_kernel(idx, stream));
//     GPU_RETURN_STATUS(GPUStreamSynchronize(stream));
//     return Status::Succ;
// }

Status launch_kernel(int kernel_offset, CUstream stream, Model* model) {
    int i = kernel_offset;
    std::string& func_name = model->kernels[i].name;
    CUfunction func = kernels[func_name];
    uint32_t *launch_params = model->kernels[i].launch_params;
    // std::cout << func_name << std::endl;
    GPU_RETURN_STATUS(cuLaunchKernel(func,
        launch_params[0], launch_params[1], launch_params[2],
        launch_params[3], launch_params[4], launch_params[5],
        0, stream, (void **)raw_args[i].data(), 0
    ));
    return Status::Succ;
}