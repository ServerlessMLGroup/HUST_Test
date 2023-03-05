#include <stdio.h>
#include <assert.h>
#include <unistd.h>

__global__ void kernel(int * inout, const int N)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int gstride = gridDim.x * blockDim.x;

   for (; gid < N; gid+= gstride) inout[gid] *= 2;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(void)
{
    const int N = 2<<20, sz = N * sizeof(int);

    int * inputs, * outputs, * _inout;

    gpuErrchk( cudaMallocHost((void **)&inputs, sz) );
    gpuErrchk( cudaMallocHost((void **)&outputs, sz) );
    gpuErrchk( cudaMalloc((void **)&_inout, sz) );

    for(int i=0; i<N; i++) { inputs[i] = i; outputs[i] = 0; }

    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++)
        gpuErrchk( cudaStreamCreate(&stream[i]) );

    gpuErrchk( cudaMemcpyAsync(_inout, inputs, sz, cudaMemcpyHostToDevice, stream[1]) );

    kernel<<<128, 128, 0, stream[1]>>>(_inout, N);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk( cudaMemcpyAsync(outputs, _inout, sz, cudaMemcpyDeviceToHost, stream[1]) );

    for(int i = 0; i < 2; i++)
        gpuErrchk( cudaStreamDestroy(stream[i]) );

    cudaDeviceSynchronize();
    //sleep(1); // remove the sleep and see what happens....

    for(int i = 0; i < N; i++)
        assert( (2 * inputs[i]) == outputs[i] );

    cudaDeviceReset();

    return 0;
}