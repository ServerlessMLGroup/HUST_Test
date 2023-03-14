/*
#include "model.h"
#include "log.h"
#include <bits/unique_ptr.h>
#include <cuda.h>
#include "cuda_runtime.h"

//yy add
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <math.h>
#include "unistd.h"
#include <thread>

#define datablocksize 5*1024*1024
#define PARAM_MAGIC "TVM_MODEL_PARAMS"

// #include <glog/logging.h>

// Notice
// Diy main.cpp:data block experiment


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

//old
std::vector<CUdeviceptr*> storage;


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
    // double duration = (double(end - start));
    // std::cout << "func_name:" << func_name << " time:" << duration << std::endl;
    // std::cout << "func_name:" << func_name << " launch_params:" << launch_params[0] << " " << launzch_params[1] << " " << launch_params[2] << " " << launch_params[3] << " " << launch_params[4] << " " << launch_params[5] << " raw_args_ptr:" << (void **)raw_args[i].data() << std::endl;
    return Status::Succ;
}

Status execute_to(int idx, CUstream stream, Model* model) {
    for (int i = 0; i < idx; i++) {
        RETURN_STATUS(launch_kernel(i, stream, model));
        //GPU_RETURN_STATUS(cuStreamSynchronize(stream));
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
    // TODO: printf input_storage_idx
    return Status::Succ;
}

//zhe li dei gai
Status get_data(int idx, void* out, size_t len) {
    if (idx >= storage.size()) RETURN_STATUS(Status::OutOfRange);
    StorageInfo& storage_info = model->storage[idx];
    size_t storage_size = Model::get_stype_size(storage_info.stype) * storage_info.size;
    std::cout << "storage_size:" << storage_size << std::endl;
    if (len < storage_size) RETURN_STATUS(Status::Fail);
    GPU_RETURN_STATUS(cuMemcpyDtoH(
        out, (CUdeviceptr)(*storage[idx]), storage_size
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
    return get_data(input_storage_idx, out.data(), storage_info.size * sizeof(float));
}

bool argexist(int temparg,int* aused,int* top)
{
    for(int i=0;i<*top;i++)
    {
    if(temparg == aused[i])
    {
    return true;
    }
    }
    aused[*top]=temparg;
    *top=*top +1;
    return false;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
    }
    int gpu_no = atoi(argv[1]);
    log("preate unique_ptr");
    model.reset(Model::from_json("/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18-final.json"));
    CUcontext ctx;
    CUdevice device;
    CUresult result;
    // init CUDA driver API
    GPU_RETURN_STATUS(cuInit(0));
    GPU_RETURN_STATUS(cuDeviceGet(&device, gpu_no));
    GPU_RETURN_STATUS(cuCtxCreate(&ctx, 0, device));

    CUmodule mod;
    GPU_RETURN_STATUS(cuModuleLoad(&mod, "/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.ptx"));
    printf("load cuda kernels!\n");


    //yy change:huan yi ge wenjian hai yao gai makefile,wo jiu yong zhe ge le
    //wo hui zai wo gai de mei yige di fang jia shang zhushi yy

    //yy add stream
    CUstream firststream;
    cuStreamCreate(&firststream,0);
    //add fininshed

    // 2. load cuda kernels
    for (KernelInfo &kernel_info : model->kernels) {
        CUfunction kernel;
        GPU_RETURN_STATUS(
            cuModuleGetFunction(&kernel, mod, kernel_info.name.c_str())
        );
        kernels.emplace(kernel_info.name, kernel);
    }
    printf("allocate device storage!\n");


    //liu zai zhe
    for (size_t i = 0; i < storage.size(); i++) {
        // std::cout << i << std::endl;
        StorageInfo& storage_info = model->storage[i];
        if (params->mpdata->find(storage_info.name) == params->mpdata->end())
            continue;

        GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)storage[i],array[i], tempsize[i],firststream));
    }
    //hou mian yong

   //yy add,parameters count
   int paramreused[80];
   size_t paramresize[80];
   int location=0;
   size_t paramloaction[80];
   size_t argloaction[80];
   size_t argsize[80];
   for(i=0;i<80;i++)
   {
   argloaction[i] = 1;
   }

   int offset=0;
   for (KernelInfo &kernel_info : model->kernels) {
        for (size_t arg_idx : kernel_info.args) {
            // assert(arg_idx < storage.size());

            //this pan duan si hu hai bu he li guang != null hao xiang bu gou
            //zhe li ke neng xv yao params
            //hai xv yan zheng
            if((model->storage[arg_idx].name!="null")&&(model->storage[arg_idx].name!="data"))
            {
            if((argloaction[arg_idx] ==1))
            {
            paramreused[offset]=(int)arg_idx;
            size_t stype_size = Model::get_stype_size(model->storage[arg_idx].stype);
            size_t storage_size = stype_size * model->storage[arg_idx].size;
            paramresize[offset]=storage_size;
            paramloaction[offset]=location;
            argloaction[arg_idx]=location;
            argsize[arg_idx]=storage_size;

            location += storage_size;
            offset++;
            }
            }
        }
    }

   int blocknum=0;
   int dataleft=datablocksize;
   int blocksize[40];
   int flags[model->kernels.size()];
   int blockkernel[40];
   int neednew=1;
   int resultchu;
   int resultyu;
   for(int k=0;k<model->kernels.size();k++)
   {
        flags[k]=1;
   }
   for(int k=0;k<40;k++)
   {
        blocksize[k]=0;
        blockkernel=-1;
   }

   for (int j=0;j<model->kernels.size();j++) {
    for(size_t arg_idx : model->kernels[j].args) {
        //qi shi xia mian zhe ge yao pan duan de shi zhe wan yi er zai
        //params li mian,dan shi na yang tai ma fan le,suo yi zhe me yong
        if((model->storage[arg_idx].name!="null")&&(model->storage[arg_idx].name!="data"))
            {
            if(argsize[arg_idx]<dataleft)
            {
            if (neednew==1)
                {
                flags[j]=0;
                neednew=0;
                blockkernel[blocknum]=j;
                blocknum++;
                }
            dataleft -=argsize[arg_idx]
            }
            else if(argsize[arg_idx]==dataleft){
             if (neednew==1)
                {
                flags[j]=0;
                neednew=0;
                blockkernel[blocknum]=j;
                blocknum++;
                }
            blocksize[blocknum-1]=datablocksize;
            dataleft=datablocksize;
            neednew=1;
            }
            else{
                if(argsize[arg_idx]<datablocksize){
                flags[j]=0;
                blockkernel[blocknum]=j;
                blocksize[blocknum-1]=datablocksize-dataleft;
                blocknum++;
                dataleft=datablocksize-argsize[arg_idx];
                }
                else if(argsize[arg_idx]==datablocksize){
                flags[j]=0;
                blockkernel[blocknum]=j;
                blocksize[blocknum-1]=datablocksize-dataleft;
                blocknum++;
                dataleft=datablocksize;
                neednew=1;
                }
                else{
                flags[j]=0;
                //bi mian shang ci dataleft gai wei man
                blocksize[blocknum-1]=datablocksize-(dataleft%datablocksize);
                resultchu=argsize[arg_idx]/datablocksize;
                resultyu=argsize[arg_idx]%datablocksize;
                for(int i=0;i<resultchu){
                blocksize[blocknum+i]=datablocksize;
                }
                blocknum=blocknum+resultchu+1;
                blockkernel[blocknum-1]=j;
                if(resultyu==0)
                {
                neednew=1;
                dataleft=datablocksize;
                }
                }
            }
            }
    }
   }


   //yy cpu,gpu params get
   float* paramdata;
   CUdeviceptr* gpuparam;
   cuMemAllocHost((void**)(&tempdata),location);
   GPU_RETURN_STATUS(cuMemAlloc(gpuparam, location));

   FILE* fp;
   fp = fopen("/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.param", "rb");
   char magic[sizeof(PARAM_MAGIC)];
   // std::cout << "fread" << __FILE__ << __LINE__ << std::endl;
   size_t res = fread(magic, sizeof(char), sizeof(PARAM_MAGIC), fp);
   assert(res == sizeof(PARAM_MAGIC));
   assert(std::string(magic) == PARAM_MAGIC);

   uint64_t params_size;
   // std::cout << "fread" << __FILE__ << __LINE__ << std::endl;
   res = fread(&params_size, sizeof(uint64_t), 1, fp);
   assert(res == 1);
   assert(params_size != 0);
   std::cout << "params_size:" << params_size << std::endl;

   //check whether size same
   int locationbyparams=0;
   int i=0;
   //variable end

   for (uint64_t i = 0; i < params_size; i++) {
        char key_buf[256];
        uint64_t key_len = 0;
        while(true) {
            char c;
            res = fread(&c, sizeof(char), 1, fp);
            assert(res == 1);
            key_buf[key_len] = c;
            key_len++;
            if (c == '\0') break;
        }
        std::string key(key_buf);

        uint64_t array_size;
        res = fread(&array_size, sizeof(uint64_t), 1, fp);
        assert(res == 1);
        assert(array_size != 0);

        std::cout << "params_name:" << key << std::endl;

        i=0;
        while(i<77)
        {
        StorageInfo& storage_info = model->storage[i];
        if(storage_info.name == null)
             {
             i++;
             continue;
             }
        if (storage_info.name == key)
             break;
        i++;
        }
        if(key != null)
        {
        locationbyparams += sizeof(float)*array_size;
        res = fread((paramdata+argloaction[i]), sizeof(float), array_size, fp);
        assert(res == array_size);
        }
    }

    //check zong da xiao
    if(locationbyparams==location)
        std::cout<<" right "<<std::endl;
    else
        std::cout<<" wrong "<<std::endl;


    // 3. allocate device storage
    for (StorageInfo &storage_info : model->storage) {
        size_t stype_size = Model::get_stype_size(storage_info.stype);
        size_t storage_size = stype_size * storage_info.size;
        CUdeviceptr device_ptr;
        std::vector<char> temp;
        temp.resize(storage_size, 0);
        if(argloaction[i]!=1)
        {
        storage[i]=gpuparam+argloaction[i];
        }
        else
        {
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size));
        GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr, temp.data(), storage_size));
        }
        storage.push_back(device_ptr);
    }

    for (int i=0;i<77;i++) {
        size_t stype_size = Model::get_stype_size(model->storage[i].stype);
        size_t storage_size = stype_size * model->storage[i].size;
        CUdeviceptr device_ptr;
        std::vector<char> temp;
        temp.resize(storage_size, 0);
        if(argloaction[i]!=1)
        {
        storage.push_back(gpuparam+argloaction[i]);
        }
        else
        {
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr, storage_size));
        //GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr, temp.data(), storage_size));
        }
        storage.push_back(&device_ptr);
    }

    printf("map raw args!\n");
    std::cout << "storages.size = " << storage.size() << std::endl;
    raw_args.reserve(model->kernels.size());

    //pack the device address into kernel_arg
    for (KernelInfo &kernel_info : model->kernels) {
        std::vector<CUdeviceptr*> kernel_arg;
        for (size_t arg_idx : kernel_info.args) {
            // assert(arg_idx < storage.size());
            kernel_arg.push_back(storage[arg_idx]);
        }
        raw_args.push_back(kernel_arg);
    }

   //divide the data


    std::vector<float> output(1000);
    RETURN_STATUS(set_input());
    RETURN_STATUS(execute(0, model.get()));
    // RETURN_STATUS(get_output(output));
    // std::vector<float> ans = {0.0003392257, 0.0014304413, 0.0004299286, 0.0010349639, 0.0020997059,
    //                     0.0016049921, 0.0010267848, 0.00042607592, 0.0018747754, 0.0024558322};
    // for (size_t i = 0; i < ans.size(); i++) {
    //     // ASSERT_FLOAT_EQ(ans[i], output[i]);
    //     std::cout << output[i] << " vs " << ans[i] << std::endl;
    // }
    printf("reset model!\n");
    model.reset();
    return 0;
}
*/