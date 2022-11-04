#include <stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

#define BLOCK_NUM 80   //块数量
#define THREAD_NUM 256 // 每个块中的线程数
#define R_SIZE BLOCK_NUM * THREAD_NUM
#define M_SIZE R_SIZE * R_SIZE

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

#define GPU_RETURN_STATUS(cmd) \
{ \
    CUresult result = cmd; \
    if (result != CUDA_SUCCESS) { \
        std::cout << #cmd " error, return code:" << result << " | " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    errorStr = cudaGetErrorString(err);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}


__device__ uint get_smid(void) {

    uint ret;
  
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
  
    return ret;
  
}

__global__ void mat_mul(int *mat1, int *mat2, int *result, int *sm) {
    sm[blockIdx.x] = get_smid();
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // 每个线程计算一行
    const int row = bid * THREAD_NUM + tid;
    for (int c = 0; c < R_SIZE; c++) {
        for (int n = 0; n < R_SIZE; n++) {
            result[row*R_SIZE+c] += mat1[row*R_SIZE+n] * mat2[n*R_SIZE+c];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
        exit(1);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));
    int *mat1, *mat2, *result, *sm;
    int *g_mat1, *g_mat2, *g_mat_result, *g_sm;
    
    // 用一位数组表示二维矩阵
    mat1 = (int*) malloc(M_SIZE * sizeof(int));
    mat2 = (int*) malloc(M_SIZE * sizeof(int));
    result = (int*) malloc(M_SIZE * sizeof(int));
    sm = (int*) malloc(BLOCK_NUM * sizeof(int));

    // initialize
    for (int i = 0; i < M_SIZE; i++) {
        mat1[i] = rand()/1000000;
        mat2[i] = rand()/1000000;
        result[i] = 0;
    }
    for (int i = 0; i < BLOCK_NUM; ++i) {
        sm[i] = -1;
    }

    cudaMalloc((void **)&g_mat1, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_mat2, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_mat_result, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_sm, sizeof(int) * BLOCK_NUM);

    cudaMemcpy(g_mat1, mat1, sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat2, mat2, sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(g_sm, sm, sizeof(int) * BLOCK_NUM, cudaMemcpyHostToDevice);

    mat_mul<<<BLOCK_NUM, THREAD_NUM>>>(g_mat1, g_mat2, g_mat_result, g_sm);

    cudaMemcpy(result, g_mat_result, sizeof(int) * M_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(sm, g_sm, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);
    for (int i = 0; i < BLOCK_NUM; ++i) {
        printf("block %d -- sm %d\n", i, sm[i]);
    }
    return 0;
}