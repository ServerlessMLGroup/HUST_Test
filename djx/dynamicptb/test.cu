#include <cuda_runtime.h>
#include <iostream>

class Array {
public:
  Array(size_t size) : size_(size) {
    cudaMalloc(&data_, size_ * sizeof(int));
  }

  ~Array() {
    cudaFree(data_);
  }

  int* data() { return data_; }
  size_t size() const { return size_; }

private:
  int* data_;
  size_t size_;
};

__global__ void kernel(Array array) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < array.size()) {
    array.data()[idx] = idx;
  }
}

int main() {
  const size_t size = 1024;
  Array array(size);

  kernel<<<1, size>>>(array);
  cudaDeviceSynchronize();

  int* data = new int[size];
  cudaMemcpy(data, array.data(), size * sizeof(int), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < size; ++i) {
    std::cout << data[i] << std::endl;
  }

  delete[] data;
  return 0;
}
