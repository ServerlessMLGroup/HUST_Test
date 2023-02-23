
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

extern "C" __global__ void fused_nn_contrib_conv2d_winograd_without_weight_transform_add_2_kernel2(float* __restrict__ bgemm, float* __restrict__ T_add, float* __restrict__ placeholder) {
  float inverse[4];
  inverse[(0)] = 0.000000e+00f;
  inverse[(0)] = (inverse[(0)] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(0)] = (inverse[(0)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = 0.000000e+00f;
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))]);
  inverse[(1)] = (inverse[(1)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))] * -1.000000e+00f));
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(1)] = (inverse[(1)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(2)] = 0.000000e+00f;
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))] * -1.000000e+00f));
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))]);
  inverse[(2)] = (inverse[(2)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(3)] = 0.000000e+00f;
  inverse[(3)] = (inverse[(3)] + ((bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440))] * -1.000000e+00f) * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968))]);
  inverse[(3)] = (inverse[(3)] + (bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144))] * -1.000000e+00f));
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232))]);
  inverse[(3)] = (inverse[(3)] + bgemm[((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320))]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_add[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner))] = (inverse[(((ax2_inner * 2) + ax3_inner))] + placeholder[((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 14) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner))]);
    }
  }
}