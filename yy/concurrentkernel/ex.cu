#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(float n1, float n2, float n3, long long unsigned *times, int stop) {
	unsigned long long mclk;
	if (threadIdx.x == 0) {
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk));
	}

	for (int i = 0; i < stop; i++) {
		n1=sinf(n1);
		n2=n3/n2;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		unsigned long long mclk2;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(mclk2));
		times[blockIdx.x] = mclk2 - mclk;
	}

}

void run_kernel(int a_blocks, int b_blocks, int a_threads, int b_threads) {
	int num_streams = 2;
	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);
	}

	long long unsigned *h_sm_ids = new long long unsigned[a_blocks];
	long long unsigned *d_sm_ids;
	cudaMalloc(&d_sm_ids, a_blocks * sizeof(long long unsigned));

	long long unsigned *h_sm_ids2 = new long long unsigned[b_blocks];
	long long unsigned *d_sm_ids2;
	cudaMalloc(&d_sm_ids2, b_blocks * sizeof(long long unsigned));

	dim3 Dba = dim3(a_threads);
	dim3 Dga = dim3(a_blocks,1,1);
	dim3 Dbb = dim3(b_threads);
	dim3 Dgb = dim3(b_blocks,1,1);
	kernel <<<Dga, Dba, 0, streams[0]>>>(15.6, 64.9, 134.7, d_sm_ids, 5000000);
	kernel <<<Dgb, Dbb, 0, streams[1]>>>(98.2, 3.6, 17.8, d_sm_ids2, 5000000);

	cudaDeviceSynchronize();

	cudaMemcpy(h_sm_ids, d_sm_ids, a_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sm_ids2, d_sm_ids2, b_blocks * sizeof(long long unsigned), cudaMemcpyDeviceToHost);

	for (int i = 0; i < a_blocks; i++) {
		printf("%llu\n", h_sm_ids[i]);
	}

	for (int i = 0; i < b_blocks; i++) {
		printf("%llu\n", h_sm_ids2[i]);
	}

	cudaFree(d_sm_ids);
	cudaFree(d_sm_ids2);

}

int main() {
	run_kernel(67, 8, 512, 32);

	return 0;
}

