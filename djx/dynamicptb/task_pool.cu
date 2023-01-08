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
#define RESULT_NUM SM_NUM
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
    cudaMalloc((void**)&mutex, sizeof(int));
    cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&tasks_, capacity_ * sizeof(TASK));
  }

   ~CudaTaskPool() {
    cudaFree(tasks_);
    cudaFree(mutex);
  }

  __device__ void push(int id) {
    while (atomicCAS(mutex, 0, 1) != 0);
    tasks_[head_].block_id = id;
    head_ = (head_ + 1) % capacity_;
    atomicExch(mutex, 0);
  }

  __device__ int get() {
    while (atomicCAS(mutex, 0, 1) != 0);
    if (head_ == tail_) {
      return -1;  
    }
    TASK task = tasks_[tail_];
    tail_ = (tail_ + 1) % capacity_;
    atomicExch(mutex, 0);
    return task.block_id;
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

extern void initDevice(int argc, char *argv[]);

__device__ uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}


struct Task{
    int block_id;
    __host__ __device__ Task(int data) : block_id(data) {}
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
    int* sm_flag = flag + FLAG_SM_BASE, *block_flag = flag + FLAG_BLOCK_BASE, *result_flag = flag + FLAG_RESULT_BASE;
    unsigned int ns = 5;
    int smid = get_smid();
    if (threadIdx.x == 0 && atomicAdd(sm_flag + smid, 1) == 0) atomicAdd(block_flag + blockIdx.x, 1);
    __syncthreads();

    if (atomicAdd(block_flag + blockIdx.x, 0) == 0) return ;
    __syncthreads();

    __shared__ int BlockSyn[128 + 5];
    BlockSyn[threadIdx.x] = 0;

    if (threadIdx.x == 0) {
      int id;
      while(id = task_pool.get()) {
        if (id == -1) break;
        // printf("block %d get task %d\n", blockIdx.x, id);
        atomicAdd(result_flag + smid, 1);
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
  for (int i = 0; i < TASK_NUM; ++i) {
    task_pool.push(i);
    atomicAdd(flags + FLAG_RESULT_BASE + RESULT_NUM + 1, 1);
  }
  ElasticKernel(flags, task_pool);
}


int main(int argc, char *argv[]) {
    // init device
    initDevice(argc, argv);

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
    CudaTaskPool<Task>* task_pool;
    cudaMallocManaged(&task_pool, sizeof(CudaTaskPool<Task>));

    // 在 CPU 端初始化 task_pool
    new(task_pool) CudaTaskPool<Task>(TASK_NUM);


    // CudaTaskPool<Task> *d_task_pool;
    // cudaMalloc(&d_task_pool, sizeof(CudaTaskPool<Task>));
    // cudaMemcpy(d_task_pool, &task_pool, sizeof(CudaTaskPool<Task>), cudaMemcpyHostToDevice);

    // cuda launch kernel
    dim3 Dim_block = dim3(BLOCK_NUM, 1, 1);
    dim3 Dim_thread = dim3(THREAD_NUM, 1, 1);
    // warm-up
    for (int i = 0; i < 100; ++i) {
        workload <<<Dim_block, Dim_thread, 0, streams[0]>>> ();
    }
    cudaDeviceSynchronize();
    cudaMemcpy(g_flag, flag, sizeof(int) * FLAG_LENGTH, cudaMemcpyHostToDevice);

    cudaMemcpy(flag, g_flag, sizeof(int) * FLAG_LENGTH, cudaMemcpyDeviceToHost);

    int total = 0;
    for (int i = FLAG_RESULT_BASE; i < FLAG_RESULT_BASE + RESULT_NUM; ++i) {
      printf("sm %d get %d task\n", i - FLAG_RESULT_BASE, flag[i]);
      total += flag[i];
    }
    printf("total task:%d\n", total);
    printf("total push:%d\n", flag[FLAG_RESULT_BASE + RESULT_NUM]);

    LaunchKernel <<<Dim_block, Dim_thread, 0, streams[0]>>> (g_flag, *task_pool);
    cudaDeviceSynchronize();


}