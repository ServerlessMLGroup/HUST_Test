extern "C" __global__ void fused_add_nn_relu_1_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {
    int vx=blockIdx.x;
  int vy=blockIdx.y;
  int vz=blockIdx.z;
  int offset=0;

  if((blocknum[0]*blocknum[1]*blocknum[2])>blocksize[0])
  {
    offset=vx;
    while(offset<(blocknum[0]*blocknum[1]*blocknum[2]))
    {
    vz=(offset)/(blocknum[0]*blocknum[1]);
    vy= (offset-(vz*blocknum[0]*blocknum[1]))/blocknum[0];
    vx=offset - (vz*blocknum[0]*blocknum[1])-vy*blocknum[0];
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)vx) * 1024)) + ((int)threadIdx.x)) < 1605632) {
      T_relu[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)vx) * 1024)) + ((int)threadIdx.x)))] = max((placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)vx) * 1024)) + ((int)threadIdx.x)))] + placeholder1[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)vx) * 1024)) + ((int)threadIdx.x)) % 50176) / 196))]), 0.000000e+00f);
    }
  }
    offset+=blocksize[0];
    }
  }
  else
  {
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer < 7; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)vx) * 1024)) + ((int)threadIdx.x)) < 1605632) {
      T_relu[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)vx) * 1024)) + ((int)threadIdx.x)))] = max((placeholder[((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)vx) * 1024)) + ((int)threadIdx.x)))] + placeholder1[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 262144) + (((int)vx) * 1024)) + ((int)threadIdx.x)) % 50176) / 196))]), 0.000000e+00f);
    }
  }
  }

}
extern "C" __global__ void fused_nn_conv2d_add_1_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {

}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_7_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_nn_max_pool2d_add_nn_relu_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {

}

extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {

}
extern "C" __global__ void fused_add_nn_relu_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {

}
extern "C" __global__ void fused_nn_softmax_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ T_softmax_norm) {

}
extern "C" __global__ void fused_add_nn_relu_2_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {

}
extern "C" __global__ void fused_nn_conv2d_1_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {

}

extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_1_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {

}
extern "C" __global__ void fused_nn_conv2d_add_nn_relu_2_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_nn_dense_add_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_add_10_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ T_add, float* __restrict__ placeholder, float* __restrict__ placeholder1) {

}
extern "C" __global__ void fused_nn_conv2d_add_nn_relu_1_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_nn_conv2d_add_nn_relu_4_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_nn_conv2d_add_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_nn_conv2d_add_add_nn_relu_2_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3) {

}
extern "C" __global__ void fused_nn_conv2d_add_2_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_nn_conv2d_3_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {

}
extern "C" __global__ void fused_nn_conv2d_add_nn_relu_6_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_nn_conv2d_add_3_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_nn_conv2d_add_nn_relu_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_nn_conv2d_2_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {

}
extern "C" __global__ void fused_nn_conv2d_add_multiply_add_nn_relu_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2, float* __restrict__ placeholder3, float* __restrict__ placeholder4) {

}
extern "C" __global__ void fused_nn_global_avg_pool2d_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ tensor) {

}
extern "C" __global__ void fused_nn_batch_flatten_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ tensor, float* __restrict__ placeholder) {

}
extern "C" __global__ void fused_nn_conv2d_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {

}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_5_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {

}
extern "C" __global__ void fused_add_nn_relu_3_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ T_relu, float* __restrict__ placeholder, float* __restrict__ placeholder1) {

}
extern "C" __global__ void fused_nn_conv2d_add_nn_relu_3_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {

}

























