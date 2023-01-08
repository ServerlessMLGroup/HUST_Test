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
#define FLAG_SM_BASE (FLAG_BLOCK_BASE + BLOCK_NUM)
#define FLAG_RESULT_BASE (FLAG_SM_BASE + SM_NUM)
#define FLAG_TIME_BASE (FLAG_RESULT_BASE + RESULT_NUM)
#define POOL_CAPACITY 300
#define POOL_LENGTH POOL_CAPACITY + 4
#define POOL_CAPACITY_BASE 0
#define POOL_HEAD_BASE 1
#define POOL_TAIL_BASE 2
#define POOL_MUTEX_BASE 3
#define POOL_ARRAY_BASE 4
// nvcc -arch=native task_pool.cu tool.cu -o pool

__device__ void push_task(int* task_pool, int id) {
    int* mutex = task_pool + POOL_MUTEX_BASE, capacity_ = task_pool[POOL_CAPACITY_BASE], *head_ = task_pool + POOL_HEAD_BASE, *tail_ = task_pool + POOL_TAIL_BASE, *tasks_ = task_pool + POOL_ARRAY_BASE;
    // printf("into push %d\n", id);
    while (atomicCAS(mutex, 0, 1) != 0);
    // printf("into push %d\n", id);
    tasks_[*head_] = id;
    *head_ = (*head_ + 1) % capacity_;
    // printf("push_task %d, head = %d, tail = %d, capacity = %d\n", id, *head_, *tail_, capacity_);
    atomicExch(mutex, 0);
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

extern void initDevice(int argc, char *argv[]);

__device__ uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

__device__ dim3 Dim_block = dim3(BLOCK_NUM, 1, 1);
__device__ dim3 Dim_thread = dim3(THREAD_NUM, 1, 1);


__global__ void workload() {
    int n1 = 15.6, n2 = 64.9, n3 = 134.7;
    for (int i = 0; i < 50000; i++) {
        n1=sinf(n1);
        n2=n3/n2;
    }
    __syncthreads();
}

__global__ void ElasticKernel(int *flag, int* task_pool) {
    int* sm_flag = flag + FLAG_SM_BASE, *block_flag = flag + FLAG_BLOCK_BASE, *result_flag = flag + FLAG_RESULT_BASE, *times = flag + FLAG_TIME_BASE;
    unsigned int ns = 5;
    int smid = get_smid();
    if (threadIdx.x == 0 && atomicAdd(sm_flag + smid, 1) == 0) atomicAdd(block_flag + blockIdx.x, 1);
    __syncthreads();

    if (atomicAdd(block_flag + blockIdx.x, 0) == 0) return ;
    __syncthreads();
    __shared__ int BlockSyn[128 + 5];
    BlockSyn[threadIdx.x] = 0;

    // if (threadIdx.x == 0 )printf("%d\n", smid); 已验证仍然均匀分布

    if (threadIdx.x == 0) {
      int id = 0, task_num = 0, i = 0;
      unsigned long long mclk[2];
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk[(i++) % 2]));
      while((id = get_task(task_pool)) != -1) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk[(i++) % 2]));
        times[smid * TASK_NUM + task_num] = (mclk[(i + 1) % 2] - mclk[i % 2]) / 1000;
        ++task_num;
        atomicAdd(result_flag + smid, 1);

        // workload 目前与blockIdx无关
        int n1 = 15.6, n2 = 64.9, n3 = 134.7;
        for (int ii = 0; ii < 500; ii++) {
            n1=sinf(n1);
            n2=n3/n2;
        }
      }
      times[smid * TASK_NUM + task_num] = -1;
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

__device__ void init_pool(int *flags, int *task_pool) {
  int* mutex = task_pool + POOL_MUTEX_BASE, *capacity_ = task_pool + POOL_CAPACITY_BASE, *head_ = task_pool + POOL_HEAD_BASE, *tail_ = task_pool + POOL_TAIL_BASE, *tasks_ = task_pool + POOL_ARRAY_BASE;
  atomicExch(mutex, 0);
  atomicExch(capacity_, POOL_CAPACITY);
  atomicExch(head_, 0);
  atomicExch(tail_, 0);
  for (int i = 0; i < TASK_NUM; ++i) {
    push_task(task_pool, i);
    //atomicAdd(flags + FLAG_RESULT_BASE + RESULT_NUM, 1);
  }
}

__global__ void LaunchKernel(int *flags, int *task_pool) { //smid = 0
    init_pool(flags, task_pool);
    //printf("push tasks finish!\n");
    ElasticKernel<<<Dim_block, Dim_thread>>>(flags, task_pool); // 如果指定stream则会导致无效
    // cudaStreamSynchronize(stream);
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
    
    int *task_pool = new int[POOL_LENGTH];
    int *g_task_pool;
    for (int i = 0; i < POOL_LENGTH; ++i) {
        task_pool[i] = 0;
    }
    cudaMalloc((void **)&g_task_pool, sizeof(int) * POOL_LENGTH);
    cudaMemcpy(g_task_pool, task_pool, sizeof(int) * POOL_LENGTH, cudaMemcpyHostToDevice);
    // cuda launch kernel
    // warm-up
    for (int i = 0; i < 50; ++i) {
        workload <<<Dim_block, Dim_thread, 0, streams[0]>>> ();
        // LaunchKernel <<<1, 1, 0, streams[0]>>> (g_flag, g_task_pool);
    }
    cudaDeviceSynchronize();
    // flush
    cudaMemcpy(g_flag, flag, sizeof(int) * FLAG_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(g_task_pool, task_pool, sizeof(int) * POOL_LENGTH, cudaMemcpyHostToDevice);

    LaunchKernel <<<1, 1, 0, streams[0]>>> (g_flag, g_task_pool);
    cudaDeviceSynchronize();

    cudaMemcpy(flag, g_flag, sizeof(int) * FLAG_LENGTH, cudaMemcpyDeviceToHost);

    int total = 0;
    for (int i = FLAG_RESULT_BASE; i < FLAG_RESULT_BASE + RESULT_NUM; ++i) {
      printf("sm %d get %d task\n", (i - FLAG_RESULT_BASE), flag[i]);
      total += flag[i];
    }
    int i = FLAG_TIME_BASE;
    for (; ; ) {
      int sm_id = (i - FLAG_TIME_BASE) / TASK_NUM;
      if (sm_id == 80) break;
      printf("sm %d task duration:", sm_id);
      while (flag[i] != -1) {
        printf("%d ", flag[i]);
        ++i;
      }
      printf("\n");
      i = (sm_id + 1) * TASK_NUM + FLAG_TIME_BASE;
    }
    printf("total task get:%d\n", total);
    //printf("total task push:%d\n", flag[FLAG_RESULT_BASE + RESULT_NUM]);
}