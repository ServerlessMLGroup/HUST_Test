#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
// nvcc -arch=native ex.cu -o ex_sleep
// 1:540
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

__device__ uint get_warpid(void) {

	    uint ret;
	      
	        asm("mov.u32 %0, %warpid;" : "=r"(ret) );
		  
		    return ret;
		      
}

__global__ void kernel(float n1, float n2, float n3, long long unsigned ** times, int stop, int* flag, long long unsigned * sleep_sm) {
	unsigned long long mclk; 
	//if (threadIdx.x %16 == 0) {
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
		times[blockIdx.x][threadIdx.x] = (mclk) / 1000; // us
	//}
        //__syncthreads();
	//for (int i = 0; i < 1000; i++) {
	//	n1=sinf(n1);
	//	n2=n3/n2;
	//}
	 __syncthreads();
	for (int i = 0; i < stop; i++) {
		n1=sinf(n1);
		n2=n3/n2;
	}
        __syncthreads();
	// flag[0] = 1在此在ms级别无变化
	//if (threadIdx.x %16 == 0) {
		unsigned long long mclk2;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk2));
		times[blockIdx.x + 80][threadIdx.x] = (mclk2) / 1000; // us
	//}
    // flag[0] = 1;
    // __syncthreads();
    // flag[0] = 1;
}

__global__ void kernel_(float n1, float n2, float n3, long long unsigned * times, int stop, int* flag, long long unsigned * sleep_sm) {

	unsigned long long mclk; 
	if (threadIdx.x == 0) {
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
		times[blockIdx.x] = (mclk) / 1000; // us
	}
        __syncthreads();
	for (int i = 0; i < 1000; i++) {
		n1=sinf(n1);
		n2=n3/n2;
	}
	for (int i = 0; i < stop; i++) {
		n1=sinf(n1);
		n2=n3/n2;
	} 
	__syncthreads();
	if (threadIdx.x == 0) {
		unsigned long long mclk2;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk2));
		times[blockIdx.x + 10] = (mclk2) / 1000; // us
	}
}

void run_kernel(int a_blocks, int b_blocks, int a_threads, int b_threads) {
	int num_streams = 2;
	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}
	
    // allocate resource
	long long unsigned ** h_sm_ids = (long long unsigned **)malloc(sizeof(long long unsigned *)* a_blocks * 2);
        long long unsigned * h_data = (long long unsigned *)malloc(sizeof(long long unsigned )* a_blocks * 2*512);
	//for(int i=0;i<a_blocks * 2;i++){
	//	h_sm_ids[i] = h_data + i*32 ;
	//}
	long long unsigned ** d_sm_ids ;
	long long unsigned * d_data ;
	cudaMalloc((void **)&d_sm_ids, a_blocks * sizeof(long long unsigned*) * 2);
	cudaMalloc((void **)&d_data, sizeof(long long unsigned) * 512 * a_blocks * 2);
	for(int i=0;i<a_blocks * 2;i++){
		 h_sm_ids[i] = d_data + i*512 ;
         }
        cudaMemcpy((void*)(d_sm_ids), (void*)(h_sm_ids), a_blocks * sizeof(long long unsigned*) * 2, cudaMemcpyHostToDevice);

	//size_t pitch = 32;
	//cudaMalloc(&d_sm_ids, a_blocks * sizeof(long long unsigned) * 2);
	//cudaMallocPitch(&d_sm_ids, &pitch, sizeof(long long unsigned) * 32, a_blocks);


	long long unsigned *h_sm_ids2 = new long long unsigned[b_blocks];
	long long unsigned *d_sm_ids2;
	cudaMalloc(&d_sm_ids2, b_blocks * sizeof(long long unsigned));

    // allocate flag
    int *flag;
    int *g_flag;
    flag = (int*) malloc(1 * sizeof(int));
    flag[0] = 0;
    cudaMalloc((void **)&g_flag, sizeof(int) * 1);
    cudaMemcpy(g_flag, flag, sizeof(int) * 1, cudaMemcpyHostToDevice);

	// allocate sleep_time
	long long unsigned *h_sleep_time = new long long unsigned[b_blocks];
	long long unsigned *d_sleep_time;
	cudaMalloc(&d_sleep_time, b_blocks * sizeof(long long unsigned));

	// allocate kernel_sleep sm
	long long unsigned *h_sleep_sm = new long long unsigned[b_blocks];
	long long unsigned *d_sleep_sm;
	cudaMalloc(&d_sleep_sm, b_blocks * sizeof(long long unsigned));


    // allocate warm flag
    int *flag_warm;
    int *g_flag_warm;
    flag_warm = (int*) malloc(1 * sizeof(int));
    flag_warm[0] = 0;
    cudaMalloc((void **)&g_flag_warm, sizeof(int) * 1);
    cudaMemcpy(g_flag_warm, flag_warm, sizeof(int) * 1, cudaMemcpyHostToDevice);

    // cuda launch kernel
	dim3 Dba = dim3(a_threads,1,1);
	dim3 Dga = dim3(a_blocks,1,1);
	dim3 Dbb = dim3(b_threads,1,1);
	dim3 Dgb = dim3(b_blocks,1,1);
    // warm-up
    for (int i = 0; i < 100; ++i) {
        kernel <<<Dga, Dba, 0, streams[0]>>>(15.6, 64.9, 134.7, d_sm_ids, 8000, g_flag_warm, d_sleep_sm);
    }
	cudaDeviceSynchronize();
    // test kernel
	//kernel_ <<<Dga, Dba, 0, streams[0]>>>(15.6, 64.9, 134.7, d_sm_ids, 8000, g_flag, d_sleep_sm);
    // sleep until kernel finish
	//kernel_sleep <<<Dgb, Dbb, 0, streams[1]>>>(15.6, 64.9, 134.7, d_sm_ids2, 50000, g_flag, d_sleep_time, d_sleep_sm);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy((void*)(h_data),(void*) (d_data), a_blocks * sizeof(long long unsigned) *512* 2, cudaMemcpyDeviceToHost);
	//cudaMemcpy2D(h_sm_ids,sizeof(long long unsigned) * 32, d_sm_ids,pitch, sizeof(long long unsigned) * 32, a_blocks * 2 , cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sm_ids2, d_sm_ids2, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_sleep_time, d_sleep_time, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sleep_sm, d_sleep_sm, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);

    long long unsigned maxm = 0, minm = 2668828023469159, max2 = 0, min_2= 2668828023469159, max_2 = 0,max_thread = 0, min_thread = 2668828023469159;
	long long unsigned maxm_e = 0, minm_e = 2668828023469159;
    printf("---1---\n");
	for (int i = 0; i < a_blocks; i++) {
		
		printf("block%d: %llu - %llu  %llu \n", i, h_data[i*512], h_data[i*512 + a_blocks*512] , h_data[i*512 + a_blocks*512]- h_data[i*512]);
        // printf("block-%d : %llu\n", i, h_sleep_sm[i]);
        //maxm = max(maxm, h_sm_ids[i][0]);
       // minm = min(minm, h_sm_ids[i][0]);
	//	maxm_e = max(maxm_e, h_sm_ids[i + a_blocks][0]);
        //minm_e = min(minm_e, h_sm_ids[i + a_blocks][0]);
	max_thread = 0;
	max2 = 0;
	min_thread = 2668828023469159;
	//min_2 = 2668828023469159;
	max_2 = max(max_2,  h_data[i*512 + a_blocks*512]- h_data[i*512]);
	min_2 = min(min_2,  h_data[i*512 + a_blocks*512]- h_data[i*512]);
		for(int j=0;j<512;j++){
			max2 = max(max2, h_data[i*512+a_blocks*512+j]- h_data[i*512+j]);
			max_thread = max(max_thread, h_data[i*512+a_blocks*512+j]);
			min_thread = min(min_thread, h_data[i*512+j]);

		}
		printf("***block%d: %llu - %llu  %llu  %llu \n", i, min_thread, max_thread , max2, max_thread-min_thread );
	}
    printf("START_TIMING:max-%llu, min-%llu\n", maxm, minm);
	printf("END_TIMING:max-%llu, min-%llu\n", maxm_e, minm_e);
	printf("DURATION:%llu, %llu\n", max_2, max_2-min_2);
	// printf("---2---\n");
	// for (int i = 0; i < b_blocks; i++) {
	// 	printf("%llu\n", h_sm_ids2[i]);
	// }
	// printf("---sleep_times---\n");
	// for (int i = 0; i < b_blocks; i++) {
	// 	printf("block-%d : %llu\n", i, h_sleep_time[i]);
	// }
	// printf("---sm---\n");
	// for (int i = 0; i < b_blocks; ++i) {
	// 	printf("block-%d : %llu\n", i, h_sleep_sm[i]);
	// }
	
	cudaFree(d_sm_ids);
	cudaFree(d_sm_ids2);

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
        exit(1);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));
	run_kernel(80, 80, 512, 512);

	return 0;
}

