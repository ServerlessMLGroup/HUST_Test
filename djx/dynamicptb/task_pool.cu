#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <queue>
#define BLOCK_NUM 200
#define THREAD_NUM 128
#define SM_NUM 80
#define TASK_NUM 200
#define RESULT_NUM BLOCK_NUM * TASK_NUM
#define FLAG_LENGTH 65535
#define FLAG_BLOCK_BASE 0
#define FLAG_SM_BASE FLAG_BLOCK_BASE + BLOCK_NUM
#define FLAG_RESULT_BASE FLAG_SM_BASE + SM_NUM
// nvcc -arch=native task_pool.cu tool.cu -o pool

template <typename TASK>
class CudaTaskPool {
 public:
   CudaTaskPool(size_t capacity) : capacity_(capacity) {
    int state = 0;
    cudaMalloc((int*)&&mutex, sizeof(int));
    cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&tasks_, capacity_ * sizeof(TASK));
  }

   ~CudaTaskPool() {
    cudaFree(tasks_);
    cudaFree(&mutex);
  }

  __device__ void push(TASK task) {
    while (atomicCAS(mutex, 0, 1) != 0);
    tasks_[head_] = std::move(task);
    head_ = (head_ + 1) % capacity_;
    atomicExch(mutex, 0);
  }

  __device__ TASK* get() {
    while (atomicCAS(mutex, 0, 1) != 0);
    if (head_ == tail_) {
      return nullptr;
    }
    TASK task = std::move(tasks_[tail_]);
    tail_ = (tail_ + 1) % capacity_;
    atomicExch(mutex, 0);
    return &task;
  }

  __device__ size_t size() const {
    return (head_ + capacity_ - tail_) % capacity_;
  }

 private:
  TASK* tasks_;
  size_t capacity_;
  size_t head_ = 0;
  size_t tail_ = 0;
  int *mutex;
};

extern void initDevice(void);
extern __device__ uint get_smid(void);

struct Task{
    int block_id;
    Task(int data) : block_id(data) {}
};

__global__ void workload() {
  int n1 = 15.6, n2 = 64.9, n3 = 134.7;
    for (int i = 0; i < 500000; i++) {
        n1=sinf(n1);
        n2=n3/n2;
    }
    __syncthreads();
}

__device__ void ElasticKernel(int *flag, CudaTaskPool<Task>& task_pool) {
    int* sm_flag = flag + FLAG_SM_BASE, *block_flag = flag + FLAG_BLOCK_BASE;
    unsigned int ns = 5;
    int smid = get_smid();
    if (threadIdx.x == 0 && atomicAdd(sm_flag + smid, 1) == 0) atomicAdd(block_flag + blockIdx.x, 1);
    __syncthreads();

    if (atomicAdd(block_flag + blockIdx.x, 0) == 0) return ;
    __syncthreads();

    __shared__ int BlockSyn[128 + 5];
    BlockSyn[threadIdx.x] = 0;

    if (threadIdx.x == 0) {
      Task *task;
      while((task = task_pool.get()) != nullptr) {
        printf("block %d get task %d\n", blockIdx.x, task->block_id);
      }
      for (int i = 0; i < THREAD_NUM; ++i) {
        atomicAdd(BlockSyn + i, 1);
      }
    }
    else {
      while (atomicAdd(BlockSyn + threadIdx.x, 0) == 0) {
        __nanosleep(ns);
        if (ns < 1000) {
            ns *= 2;
        }
      }
    }

}

__global__ void LaunchKernel(int *flags, CudaTaskPool<Task>& task_pool) {
  ElasticKernel(flags, task_pool);
}


int main() {
    // init device
    initDevice();

    // allocate stream
    int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
      cudaStreamCreate(&streams[i]);
    }
    // allocate flag
    int *flag = new int[FLAG_LENGTH];
    int *g_flag;
    for (int i = 0; i < FLAG_LENGTH; ++i) {
        flag[i] = 0;
    }
    cudaMalloc((void **)&g_flag, sizeof(int) * FLAG_LENGTH);
    cudaMemcpy(g_flag, flag, sizeof(int) * FLAG_LENGTH, cudaMemcpyHostToDevice);

    // allocate pool
    CudaTaskPool<Task> task_pool(1024);
    // 创建任务队列
    std::vector<Task*> tasks;
    for (int i = 0; i < TASK_NUM; ++i) {
      tasks.push_back(new Task(i));
    }

    // 将任务加入任务池
    for (Task *task : tasks) {
      task_pool.push(std::move(*task));
    }
    CudaTaskPool<Task> *d_task_pool;
    cudaMalloc(&d_task_pool, sizeof(CudaTaskPool<Task>));
    cudaMemcpy(d_task_pool, &task_pool, sizeof(CudaTaskPool<Task>), cudaMemcpyHostToDevice);

    // cuda launch kernel
    dim3 Dim_block = dim3(BLOCK_NUM, 1, 1);
    dim3 Dim_thread = dim3(THREAD_NUM, 1, 1);
    // warm-up
    for (int i = 0; i < 100; ++i) {
        workload <<<Dim_block, Dim_thread, 0, streams[0]>>> ();
    }
    cudaDeviceSynchronize();
    cudaMemcpy(g_flag, flag, sizeof(int) * FLAG_LENGTH, cudaMemcpyHostToDevice);

    

    LaunchKernel <<<Dim_block, Dim_thread, 0, streams[0]>>> (g_flag, *d_task_pool);

    // 释放任务
    for (Task* task : tasks) {
      delete task;
    }

}