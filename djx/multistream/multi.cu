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
    int nstreams = 2;
        // allocate and initialize an array of stream handles
    cudaStream_t *streams =
        (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    for (int i = 0; i < nstreams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    int *mat1[nstreams], *mat2[nstreams], *result[nstreams], *sm[nstreams];
    int *g_mat1[nstreams], *g_mat2[nstreams], *g_mat_result[nstreams], *g_sm[nstreams];


    for (int times = 0; times < nstreams; ++times) {
        
        // 用一位数组表示二维矩阵
        mat1[times] = (int*) malloc(M_SIZE * sizeof(int));
        mat2[times] = (int*) malloc(M_SIZE * sizeof(int));
        result[times] = (int*) malloc(M_SIZE * sizeof(int));
        sm[times] = (int*) malloc(BLOCK_NUM * sizeof(int));

        // initialize
        for (int i = 0; i < M_SIZE; i++) {
            mat1[times][i] = rand()/1000000;
            mat2[times][i] = rand()/1000000;
            result[times][i] = 0;
        }
        for (int i = 0; i < BLOCK_NUM; ++i) {
            sm[times][i] = -1;
        }

        cudaMalloc((void **)&(g_mat1[times]), sizeof(int) * M_SIZE);
        cudaMalloc((void **)&(g_mat2[times]), sizeof(int) * M_SIZE);
        cudaMalloc((void **)&(g_mat_result[times]), sizeof(int) * M_SIZE);
        cudaMalloc((void **)&(g_sm[times]), sizeof(int) * BLOCK_NUM);

        cudaMemcpy(g_mat1[times], mat1[times], sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(g_mat2[times], mat2[times], sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(g_sm[times], sm[times], sizeof(int) * BLOCK_NUM, cudaMemcpyHostToDevice);
    }

    cudaEvent_t start_event[nstreams], stop_event[nstreams], all_start_event, all_end_event;

    for (int times = 0; times < nstreams; ++times) {
        checkCudaErrors(cudaEventCreate(&start_event[times]));
        checkCudaErrors(cudaEventCreate(&stop_event[times]));
    }
    checkCudaErrors(cudaEventCreate(&all_start_event));
    checkCudaErrors(cudaEventCreate(&all_end_event));
    checkCudaErrors(cudaEventRecord(all_start_event, 0));
    float elapsed_time[nstreams];

    for (int i = 0; i < nstreams; ++i) {
        cudaStream_t stream = streams[i];
        cudaEventRecord(start_event[i], stream);
        mat_mul<<<BLOCK_NUM, THREAD_NUM>>>(g_mat1[i], g_mat2[i], g_mat_result[i], g_sm[i]);
        checkCudaErrors(cudaEventRecord(stop_event[i], stream));
    }

    checkCudaErrors(cudaEventRecord(all_end_event, 0));
    for (int i = 0; i < nstreams; ++i) {
        checkCudaErrors(cudaEventSynchronize(stop_event[i])); // Waits until the completion of all work currently captured in event
    }
    checkCudaErrors(cudaEventSynchronize(all_end_event));

    for (int ii = 0; ii < nstreams; ++ii) {
        cudaMemcpy(result[ii], g_mat_result[ii], sizeof(int) * M_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(sm[ii], g_sm[ii], sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);
        printf("-----stream %d-----\n", ii);
        for (int i = 0; i < BLOCK_NUM; ++i) {
            printf("block %d -- sm %d\n", i, sm[ii][i]);
        }
        printf("\n");
    }

    for (int i = 0; i < nstreams; ++i) {
        checkCudaErrors(cudaEventElapsedTime(&elapsed_time[i], start_event[i], stop_event[i]));
        printf("Stream%d Measured time for sample = %.3fms\n", i, elapsed_time[i]);
    }
    float elapsed;
    checkCudaErrors(cudaEventElapsedTime(&elapsed, all_start_event, all_end_event));
    printf("Total GPU Measured time for sample = %.3fms\n", elapsed); 
    return 0;
}