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
{ \
    CUresult result = cmd; \
    if (result != CUDA_SUCCESS) { \
        std::cout << #cmd " error, return code:" << result << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define RETURN_STATUS(cmd) \
{\
    Status s = cmd;\
    if (s != Status::Succ) {\
        std::cout << #cmd " error, " << __FILE__ << ":" << __LINE__ << std::endl; \
        return s;\
    }\
}

std::vector<CUdeviceptr> storage;
std::unordered_map<std::string, CUfunction> kernels;
std::vector<std::vector<CUdeviceptr*>> raw_args;
std::unique_ptr<Model> model;


Status launch_kernel(int kernel_offset, CUstream stream, Model* model) {
    int i = kernel_offset;
    std::string& func_name = model->kernels[i].name;
    CUfunction func = kernels[func_name];
    uint32_t *launch_params = model->kernels[i].launch_params;
    // std::cout << func_name << std::endl;
    GPU_RETURN_STATUS(cuLaunchKernel(func,
        launch_params[0], launch_params[1], launch_params[2],
        launch_params[3], launch_params[4], launch_params[5],
        0, stream, (void **)raw_args[i].data(), 0 // raw_args是json中指示的storage的下标
    ));
    std::cout << "func_name:" << func_name << " launch_params:" << launch_params[0] << " " << launch_params[1] << " " << launch_params[2] << " " << launch_params[3] << " " << launch_params[4] << " " << launch_params[5] << " raw_args_ptr:" << (void **)raw_args[i].data() << std::endl;
    return Status::Succ;
}

Status execute_to(int idx, CUstream stream, Model* model) {
    for (int i = 0; i < idx; i++) {
        RETURN_STATUS(launch_kernel(i, stream, model));
        GPU_RETURN_STATUS(cuStreamSynchronize(stream));
    }  
    return Status::Succ;
}

Status execute(CUstream stream, Model* model) {
    execute_to(model->kernels.size(), stream, model);
    return Status::Succ;
}


// Status execute_kernel(int idx, GPUStream_t stream) {
//     if (idx >= num_kernels()) RETURN_STATUS(Status::OutOfRange);
//     GPU_RETURN_STATUS(launch_kernel(idx, stream));
//     GPU_RETURN_STATUS(GPUStreamSynchronize(stream));
//     return Status::Succ;
// }

Status find_storage_idx(const std::string& name, size_t& idx) {
    // TODO: O(n) -> O(1)
    for (size_t i = 0; i < storage.size(); i++) {
        StorageInfo& storage_info = model->storage[i];
        if (storage_info.name == name) {
            idx = i;
            return Status::Succ;
        }
    }
    RETURN_STATUS(Status::NotFound);
    return Status::NotFound; // otherwise, the compiler thinks no return value.
}

Status set_input() {
    std::vector<float> input(3 * 224 * 224);
    for (size_t i = 0; i < 3*224*224; i++)
        input[i] = 10.0;
    size_t input_storage_idx;
    if (find_storage_idx("data", input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
    StorageInfo& storage_info = model->storage[input_storage_idx];
    size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
    if (input.size() * sizeof(float) < storage_size) RETURN_STATUS(Status::OutOfRange);
    GPU_RETURN_STATUS(cuMemcpyHtoD(
        (CUdeviceptr)storage[input_storage_idx], (void*)input.data(), 
        storage_size)
    );
    return Status::Succ;
}

Status get_data(int idx, void* out, size_t len) {
    if (idx >= storage.size()) RETURN_STATUS(Status::OutOfRange);
    StorageInfo& storage_info = model->storage[idx];
    size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
    std::cout << "storage_size:" << storage_size << std::endl;
    if (len < storage_size) RETURN_STATUS(Status::Fail);
    GPU_RETURN_STATUS(cuMemcpyDtoH(
        out, (CUdeviceptr)storage[idx], storage_size
    ));
    return Status::Succ;
}

Status get_output(std::vector<float>& out) {
    size_t input_storage_idx;
    if (find_storage_idx("output", input_storage_idx) != Status::Succ) RETURN_STATUS(Status::NotFound);
    StorageInfo& storage_info = model->storage[input_storage_idx];
    if (Model::get_stype_size(storage_info.stype) != sizeof(float)) RETURN_STATUS(Status::Fail);
    out.resize(storage_info.size);
    std::cout << "storage_info.size:" << storage_info.size << std::endl;
    return get_data(input_storage_idx, (void*)out.data(), storage_info.size * sizeof(float));
}

int main() {
    log("preate unique_ptr");
    model.reset(Model::from_json("/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18-final.json"));
    CUcontext ctx;
    CUdevice device;
    CUresult result;
    // init CUDA driver API
    GPU_RETURN_STATUS(cuInit(0));
    GPU_RETURN_STATUS(cuDeviceGet(&device, 1));
    GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device));
    CUmodule mod;
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx"));
    printf("load cuda kernels!\n");
    // 2. load cuda kernels
    for (KernelInfo &kernel_info : model->kernels) {
        CUfunction kernel;
        GPU_RETURN_STATUS(
            cuModuleGetFunction(&kernel, mod, kernel_info.name.c_str())
        );
        kernels.emplace(kernel_info.name, kernel);
    }
    printf("allocate device storage!\n");
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
    printf("map raw args!\n");
    std::cout << "storages.size = " << storage.size() << std::endl;
    raw_args.reserve(model->kernels.size());
    for (KernelInfo &kernel_info : model->kernels) {
        std::vector<CUdeviceptr*> kernel_arg;
        for (size_t arg_idx : kernel_info.args) {
            // assert(arg_idx < storage.size());
            kernel_arg.push_back(&storage[arg_idx]);
        }
        raw_args.push_back(kernel_arg);
    }
    printf("parse params!\n");
    std::unique_ptr<ModelParam> params(ModelParamParser::parse_from_file("/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.params"));
    for (size_t i = 0; i < storage.size(); i++) {
        // std::cout << i << std::endl;
        StorageInfo& storage_info = model->storage[i];
        if (params->find(storage_info.name) == params->end()) 
            continue;
        auto &array = params->at(storage_info.name);
        GPU_RETURN_STATUS(cuMemcpyHtoD(
            (CUdeviceptr)storage[i], array.data(), 
            array.size() * sizeof(float))); 
    }
    std::vector<float> output(1000);
    RETURN_STATUS(set_input());
    RETURN_STATUS(execute(0, model.get()));
    RETURN_STATUS(get_output(output));
    std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
                        0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    for (size_t i = 0; i < ans.size(); i++) {
        // ASSERT_FLOAT_EQ(ans[i], output[i]);
        if (ans[i] != output[i]) {
            std::cout << "not match:" << output[i] << " vs " << ans[i] << std::endl;
            exit(1);
        }
    }
    printf("reset model!\n");
    model.reset();
    return 0;
}