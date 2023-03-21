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
#include <sys/time.h>
#include <unistd.h>

// #include <glog/logging.h>
//Notice
// To make some experiments, i(yy) make some changes here. Before changing, i copied all the code
// Just read the code at copymain.cpp. If some bad change were made, we can fix it by the copy
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
std::vector<CUdeviceptr> storage;
std::vector<CUdeviceptr> storage1;
std::vector<CUdeviceptr> storage2;
std::unordered_map<std::string, CUfunction> kernels;
std::vector<std::vector<CUdeviceptr*>> raw_args1;
std::vector<std::vector<CUdeviceptr*>> raw_args2;
std::unique_ptr<Model> model;


int main(int argc, char **argv) {
    if (argc < 3) {
        printf("args num error! argc:%d", argc);
    }
    int gpu_no = atoi(argv[1]);
    int blcoknumber = atoi(argv[2]);
    struct timeval t1,t2;
    double timeuse;
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
    //yy add stream
    CUstream iofirststream;
    cuStreamCreate(&iofirststream,0);
    CUstream iosecondstream;
    cuStreamCreate(&iosecondstream,0);
    CUstream kefirststream;
    cuStreamCreate(&kefirststream,0);
    CUstream kesecondstream;
    cuStreamCreate(&kesecondstream,0);
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
    // 3. allocate device storage
    for (StorageInfo &storage_info : model->storage) {
        size_t stype_size = Model::get_stype_size(storage_info.stype);
        size_t storage_size = stype_size * storage_info.size;
        CUdeviceptr device_ptr1;
        CUdeviceptr device_ptr2;
        //std::vector<char> temp;
        //temp.resize(storage_size, 0);
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr1, storage_size));
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)&device_ptr2, storage_size));
        //GPU_RETURN_STATUS(cuMemcpyHtoD(device_ptr, temp.data(), storage_size));
        storage1.push_back(device_ptr1);
        storage2.push_back(device_ptr2);
    }
    printf("map raw args!\n");
    std::cout << "storages.size = " << storage1.size() << std::endl;
    raw_args1.reserve(model->kernels.size());
    raw_args2.reserve(model->kernels.size());

    CUdeviceptr device_ptr11[model->kernels.size()];
    CUdeviceptr device_ptr12[model->kernels.size()];
    CUdeviceptr device_ptr13[model->kernels.size()];
    CUdeviceptr device_ptr21[model->kernels.size()];
    CUdeviceptr device_ptr22[model->kernels.size()];
    CUdeviceptr device_ptr23[model->kernels.size()];
    for (int i=0;i<model->kernels.size();i++) {
        std::vector<CUdeviceptr*> kernel_arg1;
        std::vector<CUdeviceptr*> kernel_arg2;
        //flag
        //CUdeviceptr device_ptr1;
        size_t storage_size = 1 * sizeof(int);
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)(&(device_ptr11[i])), storage_size));
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)(&(device_ptr21[i])), storage_size));
        kernel_arg1.push_back(&(device_ptr11[i]));
        kernel_arg2.push_back(&(device_ptr21[i]));
        //block num
        //CUdeviceptr device_ptr2;
        storage_size = 3 * sizeof(int);
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)(&(device_ptr12[i])), storage_size));
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)(&(device_ptr22[i])), storage_size));
        kernel_arg1.push_back(&(device_ptr12[i]));
        kernel_arg2.push_back(&(device_ptr22[i]));
        //blocksize
        //CUdeviceptr device_ptr3;
        storage_size = 1 * sizeof(int);
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)(&(device_ptr13[i])), storage_size));
        GPU_RETURN_STATUS(cuMemAlloc((CUdeviceptr*)(&(device_ptr23[i])), storage_size));
        kernel_arg1.push_back(&(device_ptr13[i]));
        kernel_arg2.push_back(&(device_ptr23[i]));
        for (size_t arg_idx : model->kernels[i].args) {
            // assert(arg_idx < storage.size());
            kernel_arg1.push_back(&storage1[arg_idx]);
            kernel_arg2.push_back(&storage2[arg_idx]);
        }
        raw_args1.push_back(kernel_arg1);
        raw_args2.push_back(kernel_arg2);
    }

    printf("parse params!\n");
    parseresult* params = ModelParamParser::parse_from_file("/home/wuhao/HUST_Test/djx/json2kernel/resource/resnet18.param");
    std::cout<<" test 2: "<<std::endl;
    int kernel_offset=0;
    float* temp[80];
    size_t evsize[80];
    for (KernelInfo &kernel_info : model->kernels) {
        for (size_t arg_idx : kernel_info.args) {
          //zhe li shao yi ge chuan di
          StorageInfo& storage_info = model->storage[arg_idx];
        if (params->mpdata->find(storage_info.name) == params->mpdata->end())
            continue;
        temp[kernel_offset]=params->mpdata->at(storage_info.name);
        evsize[kernel_offset]= params->mpsize->at(storage_info.name)*sizeof(float);
        kernel_offset++;
        }
    }
    std::cout<<" test 3: "<<std::endl;
    kernel_offset=0;
    int j=0;
    float* temp2;
    //RETURN_STATUS(set_input());
    for (KernelInfo &kernel_info : model->kernels) {
        for (size_t arg_idx : kernel_info.args) {
          //zhe li shao yi ge chuan di
          StorageInfo& storage_info = model->storage[arg_idx];
          if(params->mpdata->find(storage_info.name) == params->mpdata->end())
            continue;
          GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)storage1[arg_idx],temp[kernel_offset], evsize[kernel_offset],iofirststream));
          GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)storage2[arg_idx],temp[kernel_offset], evsize[kernel_offset],iosecondstream));
          kernel_offset++;
        }
        //don't pipe
        /*
        cuStreamSynchronize(firststream);
        std::string& func_name = kernel_info.name;
        CUfunction func = kernels[func_name];
        uint32_t *launch_params = kernel_info.launch_params;
        GPU_RETURN_STATUS(cuLaunchKernel(func,
        launch_params[0], launch_params[1], launch_params[2],
        launch_params[3], launch_params[4], launch_params[5],
        0, secondstream, (void **)raw_args1[j].data(), 0 // raw_args1是json中指示的storage的下标
    ));
        j++;
        */
    }

    //init flag
    int* allflag;
    cuMemAllocHost((void**)(&allflag), 80*sizeof(int));
    for(int i=0;i<80;i++)
    {
    allflag[i]=1;
    }

    //init blocksize
    int* allblocksize;
    cuMemAllocHost((void**)(&allblocksize), 80*sizeof(int));
    for(int i=0;i<80;i++)
    {
    allblocksize[i]=blcoknumber;
    }
    //init blocknum
    int* allblocknum;
    cuMemAllocHost((void**)(&allblocknum), 240*sizeof(int));
    for(int i=0;i<model->kernels.size();i++)
    {
        uint32_t *launch_params = model->kernels[i].launch_params;
        allblocknum[3*i+0]=launch_params[0];
        allblocknum[3*i+1]=launch_params[1];
        allblocknum[3*i+2]=launch_params[2];
        GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)(device_ptr11[i]),(allflag+i),1*sizeof(int),iofirststream));
        GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)(device_ptr12[i]),(allblocknum+3*i),3*sizeof(int),iofirststream));
        GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)(device_ptr13[i]),(allblocksize+i),1*sizeof(int),iofirststream));
        GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)(device_ptr21[i]),(allflag+i),1*sizeof(int),iosecondstream));
        GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)(device_ptr22[i]),(allblocknum+3*i),3*sizeof(int),iosecondstream));
        GPU_RETURN_STATUS(cuMemcpyHtoDAsync((CUdeviceptr)(device_ptr23[i]),(allblocksize+i),1*sizeof(int),iosecondstream));
    }

    cuStreamSynchronize(iofirststream);
    cuStreamSynchronize(iosecondstream);

    gettimeofday(&t1,NULL);
    j=15;
    for (KernelInfo &kernel_info : model->kernels) {
        std::string& func_name = kernel_info.name;
        CUfunction func = kernels[func_name];
        uint32_t *launch_params = kernel_info.launch_params;

        if(launch_params[0]*launch_params[1]*launch_params[2]>blcoknumber)
        {
        GPU_RETURN_STATUS(cuLaunchKernel(func,
        blcoknumber, 1, 1,
        launch_params[3], launch_params[4], launch_params[5],
        0, kefirststream, (void **)raw_args1[j].data(), 0 // raw_args1是json中指示的storage的下标
    ));
        GPU_RETURN_STATUS(cuLaunchKernel(func,
        blcoknumber, 1, 1,
        launch_params[3], launch_params[4], launch_params[5],
        0, kesecondstream, (void **)raw_args2[j].data(), 0 // raw_args1是json中指示的storage的下标
    ));
        }
        else{
        GPU_RETURN_STATUS(cuLaunchKernel(func,
        launch_params[0], launch_params[1], launch_params[2],
        launch_params[3], launch_params[4], launch_params[5],
        0, kefirststream, (void **)raw_args1[j].data(), 0 // raw_args1是json中指示的storage的下标
    ));
        GPU_RETURN_STATUS(cuLaunchKernel(func,
        launch_params[0], launch_params[1], launch_params[2],
        launch_params[3], launch_params[4], launch_params[5],
        0, kesecondstream, (void **)raw_args2[j].data(), 0 // raw_args1是json中指示的storage的下标
    ));
        }

        if(j==28)
        break;
        j++;

    }

    cuStreamSynchronize(kesecondstream);
    cuStreamSynchronize(kefirststream);
    gettimeofday(&t2,NULL);
    timeuse = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    std::cout<<"All "<<" Use Time: "<< timeuse <<std::endl;

    //std::vector<float> output(1000);
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