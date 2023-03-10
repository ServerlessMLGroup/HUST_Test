
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_4_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[128];
  #pragma unroll
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    compute[(ff_init)] = 0.000000e+00f;
    compute[((ff_init + 2))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 196) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 14)) / 7) * 196)) + (((int)blockIdx.y) * 98)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 14)) % 7) * 14)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 14)))];
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 3)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 128) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 3) * 512)) + (rc_outer * 8)) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      #pragma unroll
      for (int ff = 0; ff < 2; ++ff) {
        compute[(ff)] = (compute[(ff)] + (pad_temp_shared[((((rc_inner * 98) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner))]));
        compute[((ff + 2))] = (compute[((ff + 2))] + (pad_temp_shared[((((rc_inner * 98) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 64))]));
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    T_relu[(((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (ax1_inner_inner_inner * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = max((compute[(ax1_inner_inner_inner)] + placeholder2[((((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[((((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 392)) + (ax1_inner_inner_inner * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1568))] = max((compute[((ax1_inner_inner_inner + 2))] + placeholder2[(((((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 8))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_18_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[297];
  __shared__ float placeholder_shared[288];
  compute[(0)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 38) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 297) {
        if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 38) {
          pad_temp_shared[((((((int)threadIdx.z) * 38) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((1 <= ((((int)blockIdx.y) * 8) + ((((((int)threadIdx.z) * 38) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 33))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.z) * 38) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 33)))) ? placeholder[(((((((rc_outer * 50176) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.z) * 38) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 33) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.z) * 38) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 33)) - 225))] : 0.000000e+00f);
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 32) {
        if (((((int)threadIdx.z) * 12) + ((int)threadIdx.x)) < 96) {
          if ((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 288) {
            if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 36) {
              placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)threadIdx.z) * 108) + ((((int)threadIdx.x) / 3) * 27)) + (rc_outer * 9)) + ((((int)threadIdx.x) % 3) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))];
            }
          }
        }
      }
    }
    __syncthreads();
    for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
      for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
        compute[(0)] = (compute[(0)] + (pad_temp_shared[((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner))] * placeholder_shared[((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner))]));
        compute[(4)] = (compute[(4)] + (pad_temp_shared[((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 72))]));
        compute[(8)] = (compute[(8)] + (pad_temp_shared[((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 144))]));
        compute[(12)] = (compute[(12)] + (pad_temp_shared[((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 216))]));
        compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 66))] * placeholder_shared[((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner))]));
        compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 66))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 72))]));
        compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 66))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 144))]));
        compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 66))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 216))]));
        compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 132))] * placeholder_shared[((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner))]));
        compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 132))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 72))]));
        compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 132))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 144))]));
        compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 132))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 216))]));
        compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 198))] * placeholder_shared[((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner))]));
        compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 198))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 72))]));
        compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 198))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 144))]));
        compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((ry_inner * 33) + (((int)threadIdx.x) * 2)) + rx_inner) + 198))] * placeholder_shared[(((((((int)threadIdx.z) * 9) + (ry_inner * 3)) + rx_inner) + 216))]));
      }
    }
  }
  T_relu[(((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 100352))] = max((compute[(4)] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 200704))] = max((compute[(8)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 301056))] = max((compute[(12)] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 112))] = max((compute[(1)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 100464))] = max((compute[(5)] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 200816))] = max((compute[(9)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 301168))] = max((compute[(13)] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 224))] = max((compute[(2)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 100576))] = max((compute[(6)] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 200928))] = max((compute[(10)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 301280))] = max((compute[(14)] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 336))] = max((compute[(3)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 100688))] = max((compute[(7)] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 201040))] = max((compute[(11)] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 16)) + ((int)threadIdx.x)) + 301392))] = max((compute[(15)] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_16_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[1792];
  __shared__ float placeholder_shared[1024];
  for (int yy_init = 0; yy_init < 2; ++yy_init) {
    compute[(yy_init)] = 0.000000e+00f;
    compute[((yy_init + 4))] = 0.000000e+00f;
    compute[((yy_init + 8))] = 0.000000e+00f;
    compute[((yy_init + 12))] = 0.000000e+00f;
    compute[((yy_init + 2))] = 0.000000e+00f;
    compute[((yy_init + 6))] = 0.000000e+00f;
    compute[((yy_init + 10))] = 0.000000e+00f;
    compute[((yy_init + 14))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 2; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 200704) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.y) * 224)) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 56) * 112)) + (((int)blockIdx.x) * 56)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 56)))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((int)threadIdx.z) * 128) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 32)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      for (int yy = 0; yy < 2; ++yy) {
        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
        compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
        compute[((yy + 8))] = (compute[((yy + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
        compute[((yy + 12))] = (compute[((yy + 12))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
        compute[((yy + 2))] = (compute[((yy + 2))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
        compute[((yy + 6))] = (compute[((yy + 6))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
        compute[((yy + 10))] = (compute[((yy + 10))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
        compute[((yy + 14))] = (compute[((yy + 14))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 56)) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      }
    }
  }
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    T_relu[((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 112)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)))] = max((compute[(ax2_inner_inner_inner)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 112)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 200704))] = max((compute[((ax2_inner_inner_inner + 4))] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 112)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 401408))] = max((compute[((ax2_inner_inner_inner + 8))] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 112)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 602112))] = max((compute[((ax2_inner_inner_inner + 12))] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 112)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 28))] = max((compute[((ax2_inner_inner_inner + 2))] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 112)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 200732))] = max((compute[((ax2_inner_inner_inner + 6))] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 112)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 401436))] = max((compute[((ax2_inner_inner_inner + 10))] + placeholder2[((((int)threadIdx.z) + 32))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 12544) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 112)) + (((int)blockIdx.x) * 56)) + ((int)threadIdx.x)) + 602140))] = max((compute[((ax2_inner_inner_inner + 14))] + placeholder2[((((int)threadIdx.z) + 48))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_9_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  __shared__ float PaddedInput_shared[900];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[24];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[4];
  PaddedInput_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = ((((30 <= ((((int)threadIdx.y) * 28) + ((int)threadIdx.x))) && (1 <= (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 30))) && ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 30) < 29)) ? placeholder[(((((((int)blockIdx.z) * 784) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) / 30) * 28)) + (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 30)) - 29))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 196))] = (((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 16) % 30)) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 16) % 30) < 29)) ? placeholder[(((((((int)blockIdx.z) * 784) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 196) / 30) * 28)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 16) % 30)) - 29))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 392))] = (((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2) % 30)) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2) % 30) < 29)) ? placeholder[(((((((int)blockIdx.z) * 784) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 392) / 30) * 28)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2) % 30)) - 29))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 588))] = (((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 18) % 30)) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 18) % 30) < 29)) ? placeholder[(((((((int)blockIdx.z) * 784) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 588) / 30) * 28)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 18) % 30)) - 29))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 116) {
    if (((int)threadIdx.y) < 5) {
      PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784))] = ((((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 86) && (1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 4) % 30))) && (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 4) % 30) < 29)) ? placeholder[(((((((int)blockIdx.z) * 784) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784) / 30) * 28)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 4) % 30)) - 29))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 28) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 60) + ((int)threadIdx.x)))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 420))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 1))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 421))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 2))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 422))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 30))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 450))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 31))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 451))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 32))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 452))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 60))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 480))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 61))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 481))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 62))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 482))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 90))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 510))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 91))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 511))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 92))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[((((((int)threadIdx.y) * 60) + ((int)threadIdx.x)) + 512))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(2)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(3)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(8)]));
  T_relu[((((((int)blockIdx.z) * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) + 392))] = max((DepthwiseConv2d[(2)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = max((DepthwiseConv2d[(1)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) + 420))] = max((DepthwiseConv2d[(3)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[256];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((rc_outer * 784) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + ((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)))];
    }
    if (((((int)threadIdx.z) * 2) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 256) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 4) * 1024)) + (rc_outer * 16)) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((rc_inner * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 128))]));
    }
  }
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392))] = max((compute[(1)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_11_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  __shared__ float PaddedInput_shared[3249];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[15];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[2];
  PaddedInput_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = (((57 <= ((((int)threadIdx.y) * 28) + ((int)threadIdx.x))) && (1 <= (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 57))) ? placeholder[(((((((int)blockIdx.z) * 3136) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) / 57) * 56)) + (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 392))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 50) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 392) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 50) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 43) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 784) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 43) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1176))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 36) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1176) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 36) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1568))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 29) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1568) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 29) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1960))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 22) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1960) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 22) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2352))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 15) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2352) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 15) % 57)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2744))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 8) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 2744) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 8) % 57)) - 57))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 113) {
    if (((int)threadIdx.y) < 5) {
      PaddedInput_shared[((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 3136))] = ((1 <= ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1) % 57)) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 3136) / 57) * 56)) + ((((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) + 1) % 57)) - 57))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 28) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 28) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 57))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 58))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 59))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 114))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 115))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 116))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 171))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 172))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 173))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 228))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 229))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 228) + (((int)threadIdx.x) * 2)) + 230))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(8)]));
  T_relu[((((((int)blockIdx.z) * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = max((DepthwiseConv2d[(1)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  __shared__ float PaddedInput_shared[225];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[9];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[1];
  PaddedInput_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))] = (((15 <= ((((int)threadIdx.y) * 7) + ((int)threadIdx.x))) && (1 <= (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) % 15))) ? placeholder[(((((((int)blockIdx.z) * 196) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) / 15) * 14)) + (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) % 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49))] = ((1 <= ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4) % 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 49) / 15) * 14)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 4) % 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98))] = ((1 <= ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 8) % 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 98) / 15) * 14)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 8) % 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147))] = ((1 <= ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 12) % 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 147) / 15) * 14)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 12) % 15)) - 15))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) < 29) {
    if (((int)threadIdx.y) < 5) {
      PaddedInput_shared[((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196))] = ((1 <= ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1) % 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 196) / 15) * 14)) + ((((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) + 1) % 15)) - 15))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 7) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 2) {
      placeholder_shared[(((((int)threadIdx.y) * 7) + ((int)threadIdx.x)))] = placeholder1[((((((int)blockIdx.z) * 9) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 2)) + 15))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 2)) + 16))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 2)) + 17))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 2)) + 30))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 2)) + 31))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 30) + (((int)threadIdx.x) * 2)) + 32))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  T_relu[((((((int)blockIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_7_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  __shared__ float PaddedInput_shared[841];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[9];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[1];
  PaddedInput_shared[(((((int)threadIdx.y) * 14) + ((int)threadIdx.x)))] = (((29 <= ((((int)threadIdx.y) * 14) + ((int)threadIdx.x))) && (1 <= (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) % 29))) ? placeholder[(((((((int)blockIdx.z) * 784) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) / 29) * 28)) + (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) % 29)) - 29))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 196))] = ((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 22) % 29)) ? placeholder[(((((((int)blockIdx.z) * 784) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 196) / 29) * 28)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 22) % 29)) - 29))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 392))] = ((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 15) % 29)) ? placeholder[(((((((int)blockIdx.z) * 784) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 392) / 29) * 28)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 15) % 29)) - 29))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 588))] = ((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 8) % 29)) ? placeholder[(((((((int)blockIdx.z) * 784) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 588) / 29) * 28)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 8) % 29)) - 29))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) < 57) {
    if (((int)threadIdx.y) < 5) {
      PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 784))] = ((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 1) % 29)) ? placeholder[(((((((int)blockIdx.z) * 784) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 784) / 29) * 28)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 1) % 29)) - 29))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 14) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 14) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 58) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 58) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 58) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 58) + (((int)threadIdx.x) * 2)) + 29))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 58) + (((int)threadIdx.x) * 2)) + 30))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 58) + (((int)threadIdx.x) * 2)) + 31))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 58) + (((int)threadIdx.x) * 2)) + 58))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 58) + (((int)threadIdx.x) * 2)) + 59))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 58) + (((int)threadIdx.x) * 2)) + 60))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  T_relu[((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_6_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[128];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 196) + (((int)threadIdx.y) * 28)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 1568) + (((int)threadIdx.z) * 392)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 14)) / 7) * 196)) + (((int)blockIdx.y) * 98)) + ((((((int)threadIdx.y) * 2) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 14)) % 7) * 14)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 14)))];
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 3)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 128) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 3) * 256)) + (rc_outer * 8)) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((rc_inner * 98) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 8) + rc_inner))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 98) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 32))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((rc_inner * 98) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 64))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((rc_inner * 98) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 8) + rc_inner) + 96))]));
    }
  }
  T_relu[((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 784))] = max((compute[(1)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 4))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 1568))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[(((((((((int)blockIdx.z) * 3136) + (((int)threadIdx.z) * 196)) + (((int)blockIdx.y) * 98)) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 2352))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 12))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_12_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[32];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[512];
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    compute[(ff_init)] = 0.000000e+00f;
    compute[((ff_init + 8))] = 0.000000e+00f;
    compute[((ff_init + 16))] = 0.000000e+00f;
    compute[((ff_init + 24))] = 0.000000e+00f;
    compute[((ff_init + 4))] = 0.000000e+00f;
    compute[((ff_init + 12))] = 0.000000e+00f;
    compute[((ff_init + 20))] = 0.000000e+00f;
    compute[((ff_init + 28))] = 0.000000e+00f;
    compute[((ff_init + 2))] = 0.000000e+00f;
    compute[((ff_init + 10))] = 0.000000e+00f;
    compute[((ff_init + 18))] = 0.000000e+00f;
    compute[((ff_init + 26))] = 0.000000e+00f;
    compute[((ff_init + 6))] = 0.000000e+00f;
    compute[((ff_init + 14))] = 0.000000e+00f;
    compute[((ff_init + 22))] = 0.000000e+00f;
    compute[((ff_init + 30))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((rc_outer * 25088) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 512) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 3) * 128)) + (rc_outer * 8)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int ff = 0; ff < 2; ++ff) {
        compute[(ff)] = (compute[(ff)] + (pad_temp_shared[(((rc_inner * 112) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner))]));
        compute[((ff + 8))] = (compute[((ff + 8))] + (pad_temp_shared[(((rc_inner * 112) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 128))]));
        compute[((ff + 16))] = (compute[((ff + 16))] + (pad_temp_shared[(((rc_inner * 112) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 256))]));
        compute[((ff + 24))] = (compute[((ff + 24))] + (pad_temp_shared[(((rc_inner * 112) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 384))]));
        compute[((ff + 4))] = (compute[((ff + 4))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner))]));
        compute[((ff + 12))] = (compute[((ff + 12))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 128))]));
        compute[((ff + 20))] = (compute[((ff + 20))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 256))]));
        compute[((ff + 28))] = (compute[((ff + 28))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 384))]));
        compute[((ff + 2))] = (compute[((ff + 2))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner))]));
        compute[((ff + 10))] = (compute[((ff + 10))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 128))]));
        compute[((ff + 18))] = (compute[((ff + 18))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 256))]));
        compute[((ff + 26))] = (compute[((ff + 26))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 384))]));
        compute[((ff + 6))] = (compute[((ff + 6))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner))]));
        compute[((ff + 14))] = (compute[((ff + 14))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 84))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 128))]));
        compute[((ff + 22))] = (compute[((ff + 22))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 84))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 256))]));
        compute[((ff + 30))] = (compute[((ff + 30))] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 84))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 384))]));
      }
    }
  }
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)))] = max((compute[(ax1_inner_inner_inner)] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 50176))] = max((compute[((ax1_inner_inner_inner + 8))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 100352))] = max((compute[((ax1_inner_inner_inner + 16))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 150528))] = max((compute[((ax1_inner_inner_inner + 24))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 56))] = max((compute[((ax1_inner_inner_inner + 4))] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 50232))] = max((compute[((ax1_inner_inner_inner + 12))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 100408))] = max((compute[((ax1_inner_inner_inner + 20))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 150584))] = max((compute[((ax1_inner_inner_inner + 28))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 28))] = max((compute[((ax1_inner_inner_inner + 2))] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 50204))] = max((compute[((ax1_inner_inner_inner + 10))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 100380))] = max((compute[((ax1_inner_inner_inner + 18))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 150556))] = max((compute[((ax1_inner_inner_inner + 26))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 84))] = max((compute[((ax1_inner_inner_inner + 6))] + placeholder2[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 50260))] = max((compute[((ax1_inner_inner_inner + 14))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 16))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 100436))] = max((compute[((ax1_inner_inner_inner + 22))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 32))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 6272)) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 150612))] = max((compute[((ax1_inner_inner_inner + 30))] + placeholder2[(((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 48))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_17_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  __shared__ float PaddedInput_shared[2052];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[30];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[8];
  PaddedInput_shared[(((((int)threadIdx.y) * 112) + ((int)threadIdx.x)))] = ((((1 <= ((((int)blockIdx.y) * 16) + (((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) / 114))) && (1 <= (((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) % 114))) && ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) / 114) * 112)) + (((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 224))] = (((1 <= ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 110) % 114)) && (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 110) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 224) / 114) * 112)) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 110) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 448))] = (((1 <= ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 106) % 114)) && (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 106) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 448) / 114) * 112)) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 106) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 672))] = (((1 <= ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 102) % 114)) && (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 102) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 672) / 114) * 112)) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 102) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 896))] = (((1 <= ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 98) % 114)) && (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 98) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 896) / 114) * 112)) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 98) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 1120))] = (((1 <= ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 94) % 114)) && (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 94) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 1120) / 114) * 112)) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 94) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 1344))] = (((1 <= ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 90) % 114)) && (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 90) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 1344) / 114) * 112)) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 90) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 1568))] = (((1 <= ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 86) % 114)) && (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 86) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 1568) / 114) * 112)) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 86) % 114)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 1792))] = ((((((((int)blockIdx.y) * 16) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 1792) / 114)) < 113) && (1 <= ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 82) % 114))) && (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 82) % 114) < 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 1792) / 114) * 112)) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 82) % 114)) - 113))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) < 36) {
    if (((int)threadIdx.y) < 1) {
      PaddedInput_shared[((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 2016))] = (((((((int)blockIdx.y) * 16) + ((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 2016) / 114)) < 113) && (((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) < 35)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 2016) / 114) * 112)) + (((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) + 78)) - 113))] : 0.000000e+00f);
    }
  }
  if (((((int)threadIdx.y) * 112) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 112) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 112) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 912) + ((int)threadIdx.x)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 114))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 115))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 116))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 228))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 229))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 230))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 342))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 343))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 344))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 456))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 457))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 458))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 570))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 571))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 572))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 684))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 685))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 686))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 798))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 799))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 800))];
  PaddedInput_shared_local[(24)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 912))];
  PaddedInput_shared_local[(25)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 913))];
  PaddedInput_shared_local[(26)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 914))];
  PaddedInput_shared_local[(27)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 1026))];
  PaddedInput_shared_local[(28)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 1027))];
  PaddedInput_shared_local[(29)] = PaddedInput_shared[((((((int)threadIdx.y) * 912) + ((int)threadIdx.x)) + 1028))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(2)] = 0.000000e+00f;
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(3)] = 0.000000e+00f;
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(4)] = 0.000000e+00f;
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(5)] = 0.000000e+00f;
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(6)] = 0.000000e+00f;
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(24)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(25)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(26)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(7)] = 0.000000e+00f;
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(24)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(25)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(26)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(27)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(28)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(29)] * placeholder_shared_local[(8)]));
  T_relu[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 896)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 896)) + ((int)threadIdx.x)) + 112))] = max((DepthwiseConv2d[(1)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 896)) + ((int)threadIdx.x)) + 224))] = max((DepthwiseConv2d[(2)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 896)) + ((int)threadIdx.x)) + 336))] = max((DepthwiseConv2d[(3)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 896)) + ((int)threadIdx.x)) + 448))] = max((DepthwiseConv2d[(4)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 896)) + ((int)threadIdx.x)) + 560))] = max((DepthwiseConv2d[(5)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 896)) + ((int)threadIdx.x)) + 672))] = max((DepthwiseConv2d[(6)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 1792)) + (((int)threadIdx.y) * 896)) + ((int)threadIdx.x)) + 784))] = max((DepthwiseConv2d[(7)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_15_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  __shared__ float PaddedInput_shared[1017];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[18];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[2];
  PaddedInput_shared[(((((int)threadIdx.y) * 56) + ((int)threadIdx.x)))] = (((1 <= ((int)blockIdx.y)) && (1 <= ((((int)threadIdx.y) * 56) + ((int)threadIdx.x)))) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 112))] = (((1 <= ((((int)blockIdx.y) * 8) + ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 112) / 113))) && (1 <= ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 112) % 113))) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 112) / 113) * 112)) + ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 112) % 113)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 224))] = ((1 <= ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 111) % 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 224) / 113) * 112)) + ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 111) % 113)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 336))] = ((1 <= ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 110) % 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 336) / 113) * 112)) + ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 110) % 113)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 448))] = ((1 <= ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 109) % 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 448) / 113) * 112)) + ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 109) % 113)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 560))] = ((1 <= ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 108) % 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 560) / 113) * 112)) + ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 108) % 113)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 672))] = ((1 <= ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 107) % 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 672) / 113) * 112)) + ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 107) % 113)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 784))] = ((1 <= ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 106) % 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 784) / 113) * 112)) + ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 106) % 113)) - 113))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 896))] = ((1 <= ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 105) % 113)) ? placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 896) / 113) * 112)) + ((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 105) % 113)) - 113))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      PaddedInput_shared[((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 1008))] = placeholder[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 896)) + (((((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 1008) / 113) * 112)) + (((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) + 104)) - 113))];
    }
  }
  if (((((int)threadIdx.y) * 56) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 56) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 56) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 452))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 1))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 453))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 2))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 454))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 113))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 565))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 114))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 566))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 115))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 567))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 226))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 678))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 227))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 679))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 228))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((((int)threadIdx.y) * 226) + (((int)threadIdx.x) * 2)) + 680))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(8)]));
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (((int)threadIdx.y) * 56)) + ((int)threadIdx.x)) + 112))] = max((DepthwiseConv2d[(1)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_batch_flatten_kernel0(float* __restrict__ tensor, float* __restrict__ placeholder) {
  tensor[(((int)threadIdx.x))] = placeholder[(((int)threadIdx.x))];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_14_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[1792];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((rc_outer * 50176) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 64)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((rc_inner * 112) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((rc_inner * 112) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[(((rc_inner * 112) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[(((rc_inner * 112) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((rc_inner * 112) + ((int)threadIdx.x)) + 84))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
    }
  }
  T_relu[(((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 50176))] = max((compute[(4)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 100352))] = max((compute[(8)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 150528))] = max((compute[(12)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 56))] = max((compute[(2)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 50232))] = max((compute[(6)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 100408))] = max((compute[(10)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 150584))] = max((compute[(14)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 28))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 50204))] = max((compute[(5)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 100380))] = max((compute[(9)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 150556))] = max((compute[(13)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 84))] = max((compute[(3)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 50260))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 100436))] = max((compute[(11)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 200704) + (((int)threadIdx.z) * 3136)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 150612))] = max((compute[(15)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  __shared__ float PaddedInput_shared[324];
  __shared__ float placeholder_shared[36];
  float PaddedInput_shared_local[9];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[1];
  PaddedInput_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = (((((9 <= ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81)) && (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) < 72)) && (1 <= ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9))) && (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9) < 8)) ? placeholder[((((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) / 81) * 49)) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 81) / 9) * 7)) + ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) % 9)) - 8))] : 0.000000e+00f);
  if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 128) {
    if (((((int)threadIdx.z) * 7) + ((int)threadIdx.y)) < 19) {
      if (((int)threadIdx.z) < 3) {
        PaddedInput_shared[(((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196))] = (((((9 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 34) % 81)) && ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 34) % 81) < 72)) && (1 <= (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 7) % 9))) && ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 7) % 9) < 8)) ? placeholder[((((((((int)blockIdx.z) * 196) + ((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 196) / 81) * 49)) + (((((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 34) % 81) / 9) * 7)) + (((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 7) % 9)) - 8))] : 0.000000e+00f);
      }
    }
  }
  if ((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) < 36) {
    if (((((int)threadIdx.z) * 7) + ((int)threadIdx.y)) < 6) {
      if (((int)threadIdx.z) < 1) {
        placeholder_shared[((((((int)threadIdx.z) * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = placeholder1[(((((((int)threadIdx.z) * 49) + (((int)blockIdx.z) * 36)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))];
      }
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 1))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 2))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 9))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 10))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 11))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 18))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 19))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[(((((((int)threadIdx.z) * 81) + (((int)threadIdx.y) * 9)) + ((int)threadIdx.x)) + 20))];
  placeholder_shared_local[(0)] = placeholder_shared[((((int)threadIdx.z) * 9))];
  placeholder_shared_local[(1)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 1))];
  placeholder_shared_local[(2)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 2))];
  placeholder_shared_local[(3)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 3))];
  placeholder_shared_local[(4)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 4))];
  placeholder_shared_local[(5)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 5))];
  placeholder_shared_local[(6)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 6))];
  placeholder_shared_local[(7)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 7))];
  placeholder_shared_local[(8)] = placeholder_shared[(((((int)threadIdx.z) * 9) + 8))];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  T_relu[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((((int)blockIdx.z) * 4) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_10_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[8];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 56) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 512)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 128)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
    }
  }
  T_relu[(((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 12544))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25088))] = max((compute[(4)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 37632))] = max((compute[(6)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 12572))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25116))] = max((compute[(5)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 37660))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_softmax_kernel0(float* __restrict__ placeholder, float* __restrict__ T_softmax_norm) {
  float normal_reduce_temp0[1];
  float red_buf0[1];
  float T_softmax_exp[32];
  float normal_reduce_temp01[1];
  float red_buf01[1];
  normal_reduce_temp0[(0)] = -3.402823e+38f;
  for (int k_inner = 0; k_inner < 32; ++k_inner) {
    if (((((int)threadIdx.x) * 32) + k_inner) < 1000) {
      normal_reduce_temp0[(0)] = max(normal_reduce_temp0[(0)], placeholder[(((((int)threadIdx.x) * 32) + k_inner))]);
    }
  }
  unsigned int mask[1];
  float t0[1];
  red_buf0[(0)] = normal_reduce_temp0[(0)];
  mask[(0)] = __activemask();
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 16, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 8, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 4, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 2, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  t0[(0)] = __shfl_down_sync(mask[(0)], red_buf0[(0)], 1, 32);
  red_buf0[(0)] = max(red_buf0[(0)], t0[(0)]);
  red_buf0[(0)] = __shfl_sync(mask[(0)], red_buf0[(0)], 0, 32);
  for (int i1_inner_outer = 0; i1_inner_outer < 8; ++i1_inner_outer) {
    for (int i1_inner_inner_s = 0; i1_inner_inner_s < 4; ++i1_inner_inner_s) {
      if ((((((int)threadIdx.x) * 32) + (i1_inner_outer * 4)) + i1_inner_inner_s) < 1000) {
        T_softmax_exp[(((i1_inner_outer * 4) + i1_inner_inner_s))] = __expf((placeholder[((((((int)threadIdx.x) * 32) + (i1_inner_outer * 4)) + i1_inner_inner_s))] - red_buf0[(0)]));
      }
    }
  }
  normal_reduce_temp01[(0)] = 0.000000e+00f;
  for (int k_inner1 = 0; k_inner1 < 32; ++k_inner1) {
    if (((((int)threadIdx.x) * 32) + k_inner1) < 1000) {
      normal_reduce_temp01[(0)] = (normal_reduce_temp01[(0)] + __shfl_sync(__activemask(), T_softmax_exp[(k_inner1)], ((int)threadIdx.x), 32));
    }
  }
  unsigned int mask1[1];
  float t01[1];
  red_buf01[(0)] = normal_reduce_temp01[(0)];
  mask1[(0)] = __activemask();
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 16, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 8, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 4, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 2, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  t01[(0)] = __shfl_down_sync(mask1[(0)], red_buf01[(0)], 1, 32);
  red_buf01[(0)] = (red_buf01[(0)] + t01[(0)]);
  red_buf01[(0)] = __shfl_sync(mask1[(0)], red_buf01[(0)], 0, 32);
  for (int i1_inner_outer1 = 0; i1_inner_outer1 < 8; ++i1_inner_outer1) {
    for (int i1_inner_inner_s1 = 0; i1_inner_inner_s1 < 4; ++i1_inner_inner_s1) {
      if ((((((int)threadIdx.x) * 32) + (i1_inner_outer1 * 4)) + i1_inner_inner_s1) < 1000) {
        T_softmax_norm[((((((int)threadIdx.x) * 32) + (i1_inner_outer1 * 4)) + i1_inner_inner_s1))] = (__shfl_sync(__activemask(), T_softmax_exp[(((i1_inner_outer1 * 4) + i1_inner_inner_s1))], ((int)threadIdx.x), 32) / red_buf01[(0)]);
      }
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[392];
  __shared__ float placeholder_shared[128];
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    compute[(ff_init)] = 0.000000e+00f;
    compute[((ff_init + 2))] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((((int)threadIdx.z) * 98) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((rc_outer * 392) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 14)) + ((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)))];
    }
    if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 3)) < 16) {
      if ((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)) < 128) {
        if (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) < 32) {
          if (((int)threadIdx.x) < 5) {
            placeholder_shared[((((((int)threadIdx.z) * 32) + (((int)threadIdx.y) * 5)) + ((int)threadIdx.x)))] = placeholder1[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 2048)) + ((((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) >> 3) * 512)) + (rc_outer * 8)) + (((((int)threadIdx.y) * 5) + ((int)threadIdx.x)) & 7)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int ff = 0; ff < 2; ++ff) {
        compute[(ff)] = (compute[(ff)] + (pad_temp_shared[((((rc_inner * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner))]));
        compute[((ff + 2))] = (compute[((ff + 2))] + (pad_temp_shared[((((rc_inner * 49) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] * placeholder_shared[(((((((int)threadIdx.z) * 16) + (ff * 8)) + rc_inner) + 64))]));
      }
    }
  }
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    T_relu[((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 98)) + (ax1_inner_inner_inner * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(ax1_inner_inner_inner)] + placeholder2[((((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner))]), 0.000000e+00f);
    T_relu[(((((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 98)) + (ax1_inner_inner_inner * 49)) + (((int)threadIdx.y) * 7)) + ((int)threadIdx.x)) + 392))] = max((compute[((ax1_inner_inner_inner + 2))] + placeholder2[(((((((int)blockIdx.z) * 16) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner) + 8))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_13_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  __shared__ float PaddedInput_shared[580];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[36];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[8];
  PaddedInput_shared[(((int)threadIdx.x))] = (((1 <= ((int)blockIdx.y)) && (1 <= ((int)threadIdx.x))) ? placeholder[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + ((int)threadIdx.x)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 56))] = ((((1 <= ((((int)blockIdx.y) * 8) + ((((int)threadIdx.x) + 56) / 58))) && (1 <= ((((int)threadIdx.x) + 56) % 58))) && (((((int)threadIdx.x) + 56) % 58) < 57)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 56) / 58) * 56)) + ((((int)threadIdx.x) + 56) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 112))] = (((1 <= ((((int)threadIdx.x) + 54) % 58)) && (((((int)threadIdx.x) + 54) % 58) < 57)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 112) / 58) * 56)) + ((((int)threadIdx.x) + 54) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 168))] = (((1 <= ((((int)threadIdx.x) + 52) % 58)) && (((((int)threadIdx.x) + 52) % 58) < 57)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 168) / 58) * 56)) + ((((int)threadIdx.x) + 52) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 224))] = (((1 <= ((((int)threadIdx.x) + 50) % 58)) && (((((int)threadIdx.x) + 50) % 58) < 57)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 224) / 58) * 56)) + ((((int)threadIdx.x) + 50) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 280))] = (((1 <= ((((int)threadIdx.x) + 48) % 58)) && (((((int)threadIdx.x) + 48) % 58) < 57)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 280) / 58) * 56)) + ((((int)threadIdx.x) + 48) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 336))] = (((1 <= ((((int)threadIdx.x) + 46) % 58)) && (((((int)threadIdx.x) + 46) % 58) < 57)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 336) / 58) * 56)) + ((((int)threadIdx.x) + 46) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 392))] = (((1 <= ((((int)threadIdx.x) + 44) % 58)) && (((((int)threadIdx.x) + 44) % 58) < 57)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 392) / 58) * 56)) + ((((int)threadIdx.x) + 44) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 448))] = (((1 <= ((((int)threadIdx.x) + 42) % 58)) && (((((int)threadIdx.x) + 42) % 58) < 57)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 448) / 58) * 56)) + ((((int)threadIdx.x) + 42) % 58)) - 57))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 504))] = ((((((((int)blockIdx.y) * 8) + ((((int)threadIdx.x) + 504) / 58)) < 57) && (1 <= ((((int)threadIdx.x) + 40) % 58))) && (((((int)threadIdx.x) + 40) % 58) < 57)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 504) / 58) * 56)) + ((((int)threadIdx.x) + 40) % 58)) - 57))] : 0.000000e+00f);
  if (((int)threadIdx.x) < 20) {
    PaddedInput_shared[((((int)threadIdx.x) + 560))] = (((((((int)blockIdx.y) * 8) + ((((int)threadIdx.x) + 560) / 58)) < 57) && (((int)threadIdx.x) < 19)) ? placeholder[((((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + (((((int)threadIdx.x) + 560) / 58) * 56)) + (((int)threadIdx.x) + 38)) - 57))] : 0.000000e+00f);
  }
  if (((int)threadIdx.x) < 9) {
    placeholder_shared[(((int)threadIdx.x))] = placeholder1[(((((int)blockIdx.z) * 9) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((int)threadIdx.x))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[((((int)threadIdx.x) + 232))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((int)threadIdx.x) + 1))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[((((int)threadIdx.x) + 233))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((int)threadIdx.x) + 2))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[((((int)threadIdx.x) + 234))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((int)threadIdx.x) + 58))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[((((int)threadIdx.x) + 290))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((int)threadIdx.x) + 59))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[((((int)threadIdx.x) + 291))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((int)threadIdx.x) + 60))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[((((int)threadIdx.x) + 292))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((int)threadIdx.x) + 116))];
  PaddedInput_shared_local[(24)] = PaddedInput_shared[((((int)threadIdx.x) + 348))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((int)threadIdx.x) + 117))];
  PaddedInput_shared_local[(25)] = PaddedInput_shared[((((int)threadIdx.x) + 349))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((int)threadIdx.x) + 118))];
  PaddedInput_shared_local[(26)] = PaddedInput_shared[((((int)threadIdx.x) + 350))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((int)threadIdx.x) + 174))];
  PaddedInput_shared_local[(27)] = PaddedInput_shared[((((int)threadIdx.x) + 406))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((int)threadIdx.x) + 175))];
  PaddedInput_shared_local[(28)] = PaddedInput_shared[((((int)threadIdx.x) + 407))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((int)threadIdx.x) + 176))];
  PaddedInput_shared_local[(29)] = PaddedInput_shared[((((int)threadIdx.x) + 408))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((int)threadIdx.x) + 232))];
  PaddedInput_shared_local[(30)] = PaddedInput_shared[((((int)threadIdx.x) + 464))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((int)threadIdx.x) + 233))];
  PaddedInput_shared_local[(31)] = PaddedInput_shared[((((int)threadIdx.x) + 465))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((int)threadIdx.x) + 234))];
  PaddedInput_shared_local[(32)] = PaddedInput_shared[((((int)threadIdx.x) + 466))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((int)threadIdx.x) + 290))];
  PaddedInput_shared_local[(33)] = PaddedInput_shared[((((int)threadIdx.x) + 522))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((int)threadIdx.x) + 291))];
  PaddedInput_shared_local[(34)] = PaddedInput_shared[((((int)threadIdx.x) + 523))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((int)threadIdx.x) + 292))];
  PaddedInput_shared_local[(35)] = PaddedInput_shared[((((int)threadIdx.x) + 524))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(4)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(24)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(25)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(26)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(5)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(24)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(25)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(26)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(27)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(28)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(29)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(2)] = 0.000000e+00f;
  DepthwiseConv2d[(6)] = 0.000000e+00f;
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(24)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(25)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(26)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(27)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(28)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(29)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(30)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(31)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(32)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(3)] = 0.000000e+00f;
  DepthwiseConv2d[(7)] = 0.000000e+00f;
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(27)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(28)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(29)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(30)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(31)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(32)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(33)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(34)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(7)] = (DepthwiseConv2d[(7)] + (PaddedInput_shared_local[(35)] * placeholder_shared_local[(8)]));
  T_relu[((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + ((int)threadIdx.x)) + 224))] = max((DepthwiseConv2d[(4)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + ((int)threadIdx.x)) + 56))] = max((DepthwiseConv2d[(1)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + ((int)threadIdx.x)) + 280))] = max((DepthwiseConv2d[(5)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + ((int)threadIdx.x)) + 112))] = max((DepthwiseConv2d[(2)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + ((int)threadIdx.x)) + 336))] = max((DepthwiseConv2d[(6)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + ((int)threadIdx.x)) + 168))] = max((DepthwiseConv2d[(3)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 3136) + (((int)blockIdx.y) * 448)) + ((int)threadIdx.x)) + 392))] = max((DepthwiseConv2d[(7)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_5_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  __shared__ float PaddedInput_shared[256];
  __shared__ float placeholder_shared[9];
  float PaddedInput_shared_local[63];
  float placeholder_shared_local[9];
  float DepthwiseConv2d[7];
  PaddedInput_shared[(((((int)threadIdx.y) * 14) + ((int)threadIdx.x)))] = ((((16 <= ((((int)threadIdx.y) * 14) + ((int)threadIdx.x))) && (1 <= (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 15))) && ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) >> 4) * 14)) + (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 28))] = (((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 12) & 15)) && (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 12) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 28) >> 4) * 14)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 12) & 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 56))] = (((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 8) & 15)) && (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 8) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 56) >> 4) * 14)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 8) & 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 84))] = (((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 4) & 15)) && (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 4) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 84) >> 4) * 14)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 4) & 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 112))] = (((1 <= (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 15)) && ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) >> 4) * 14)) + (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 15)) + 83))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 140))] = (((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 12) & 15)) && (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 12) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 140) >> 4) * 14)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 12) & 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 168))] = (((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 8) & 15)) && (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 8) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 168) >> 4) * 14)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 8) & 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 196))] = (((1 <= ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 4) & 15)) && (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 4) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 196) >> 4) * 14)) + ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 4) & 15)) - 15))] : 0.000000e+00f);
  PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 224))] = ((((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) < 16) && (1 <= (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 15))) && ((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) & 15) < 15)) ? placeholder[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 181))] : 0.000000e+00f);
  if (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) < 4) {
    if (((int)threadIdx.y) < 1) {
      PaddedInput_shared[((((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) + 252))] = 0.000000e+00f;
    }
  }
  if (((((int)threadIdx.y) * 14) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      placeholder_shared[(((((int)threadIdx.y) * 14) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.y) * 14) + (((int)blockIdx.z) * 9)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  PaddedInput_shared_local[(0)] = PaddedInput_shared[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)))];
  PaddedInput_shared_local[(9)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 32))];
  PaddedInput_shared_local[(18)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 64))];
  PaddedInput_shared_local[(27)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 96))];
  PaddedInput_shared_local[(36)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 128))];
  PaddedInput_shared_local[(45)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 160))];
  PaddedInput_shared_local[(54)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 192))];
  PaddedInput_shared_local[(1)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 1))];
  PaddedInput_shared_local[(10)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 33))];
  PaddedInput_shared_local[(19)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 65))];
  PaddedInput_shared_local[(28)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 97))];
  PaddedInput_shared_local[(37)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 129))];
  PaddedInput_shared_local[(46)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 161))];
  PaddedInput_shared_local[(55)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 193))];
  PaddedInput_shared_local[(2)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 2))];
  PaddedInput_shared_local[(11)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 34))];
  PaddedInput_shared_local[(20)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 66))];
  PaddedInput_shared_local[(29)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 98))];
  PaddedInput_shared_local[(38)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 130))];
  PaddedInput_shared_local[(47)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 162))];
  PaddedInput_shared_local[(56)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 194))];
  PaddedInput_shared_local[(3)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 16))];
  PaddedInput_shared_local[(12)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 48))];
  PaddedInput_shared_local[(21)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 80))];
  PaddedInput_shared_local[(30)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 112))];
  PaddedInput_shared_local[(39)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 144))];
  PaddedInput_shared_local[(48)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 176))];
  PaddedInput_shared_local[(57)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 208))];
  PaddedInput_shared_local[(4)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 17))];
  PaddedInput_shared_local[(13)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 49))];
  PaddedInput_shared_local[(22)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 81))];
  PaddedInput_shared_local[(31)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 113))];
  PaddedInput_shared_local[(40)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 145))];
  PaddedInput_shared_local[(49)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 177))];
  PaddedInput_shared_local[(58)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 209))];
  PaddedInput_shared_local[(5)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 18))];
  PaddedInput_shared_local[(14)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 50))];
  PaddedInput_shared_local[(23)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 82))];
  PaddedInput_shared_local[(32)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 114))];
  PaddedInput_shared_local[(41)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 146))];
  PaddedInput_shared_local[(50)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 178))];
  PaddedInput_shared_local[(59)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 210))];
  PaddedInput_shared_local[(6)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 32))];
  PaddedInput_shared_local[(15)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 64))];
  PaddedInput_shared_local[(24)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 96))];
  PaddedInput_shared_local[(33)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 128))];
  PaddedInput_shared_local[(42)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 160))];
  PaddedInput_shared_local[(51)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 192))];
  PaddedInput_shared_local[(60)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 224))];
  PaddedInput_shared_local[(7)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 33))];
  PaddedInput_shared_local[(16)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 65))];
  PaddedInput_shared_local[(25)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 97))];
  PaddedInput_shared_local[(34)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 129))];
  PaddedInput_shared_local[(43)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 161))];
  PaddedInput_shared_local[(52)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 193))];
  PaddedInput_shared_local[(61)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 225))];
  PaddedInput_shared_local[(8)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 34))];
  PaddedInput_shared_local[(17)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 66))];
  PaddedInput_shared_local[(26)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 98))];
  PaddedInput_shared_local[(35)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 130))];
  PaddedInput_shared_local[(44)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 162))];
  PaddedInput_shared_local[(53)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 194))];
  PaddedInput_shared_local[(62)] = PaddedInput_shared[((((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) + 226))];
  placeholder_shared_local[(0)] = placeholder_shared[(0)];
  placeholder_shared_local[(1)] = placeholder_shared[(1)];
  placeholder_shared_local[(2)] = placeholder_shared[(2)];
  placeholder_shared_local[(3)] = placeholder_shared[(3)];
  placeholder_shared_local[(4)] = placeholder_shared[(4)];
  placeholder_shared_local[(5)] = placeholder_shared[(5)];
  placeholder_shared_local[(6)] = placeholder_shared[(6)];
  placeholder_shared_local[(7)] = placeholder_shared[(7)];
  placeholder_shared_local[(8)] = placeholder_shared[(8)];
  DepthwiseConv2d[(0)] = 0.000000e+00f;
  DepthwiseConv2d[(1)] = 0.000000e+00f;
  DepthwiseConv2d[(2)] = 0.000000e+00f;
  DepthwiseConv2d[(3)] = 0.000000e+00f;
  DepthwiseConv2d[(4)] = 0.000000e+00f;
  DepthwiseConv2d[(5)] = 0.000000e+00f;
  DepthwiseConv2d[(6)] = 0.000000e+00f;
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(0)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(9)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(18)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(27)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(36)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(45)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(54)] * placeholder_shared_local[(0)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(1)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(10)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(19)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(28)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(37)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(46)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(55)] * placeholder_shared_local[(1)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(2)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(11)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(20)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(29)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(38)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(47)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(56)] * placeholder_shared_local[(2)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(3)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(12)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(21)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(30)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(39)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(48)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(57)] * placeholder_shared_local[(3)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(4)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(13)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(22)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(31)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(40)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(49)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(58)] * placeholder_shared_local[(4)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(5)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(14)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(23)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(32)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(41)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(50)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(59)] * placeholder_shared_local[(5)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(6)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(15)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(24)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(33)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(42)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(51)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(60)] * placeholder_shared_local[(6)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(7)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(16)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(25)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(34)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(43)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(52)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(61)] * placeholder_shared_local[(7)]));
  DepthwiseConv2d[(0)] = (DepthwiseConv2d[(0)] + (PaddedInput_shared_local[(8)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(1)] = (DepthwiseConv2d[(1)] + (PaddedInput_shared_local[(17)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(2)] = (DepthwiseConv2d[(2)] + (PaddedInput_shared_local[(26)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(3)] = (DepthwiseConv2d[(3)] + (PaddedInput_shared_local[(35)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(4)] = (DepthwiseConv2d[(4)] + (PaddedInput_shared_local[(44)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(5)] = (DepthwiseConv2d[(5)] + (PaddedInput_shared_local[(53)] * placeholder_shared_local[(8)]));
  DepthwiseConv2d[(6)] = (DepthwiseConv2d[(6)] + (PaddedInput_shared_local[(62)] * placeholder_shared_local[(8)]));
  T_relu[((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)))] = max((DepthwiseConv2d[(0)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 28))] = max((DepthwiseConv2d[(1)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 56))] = max((DepthwiseConv2d[(2)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 84))] = max((DepthwiseConv2d[(3)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 112))] = max((DepthwiseConv2d[(4)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 140))] = max((DepthwiseConv2d[(5)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 196) + (((int)threadIdx.y) * 14)) + ((int)threadIdx.x)) + 168))] = max((DepthwiseConv2d[(6)] + placeholder2[(((int)blockIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_global_avg_pool2d_kernel0(float* __restrict__ placeholder, float* __restrict__ tensor) {
  float tensor1[1];
  tensor1[(0)] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 7; ++rv0) {
    for (int rv1 = 0; rv1 < 7; ++rv1) {
      if (((int)threadIdx.y) < 1) {
        tensor1[(0)] = (tensor1[(0)] + placeholder[((((((((int)threadIdx.y) * 50176) + (((int)blockIdx.x) * 392)) + (((int)threadIdx.x) * 49)) + (rv0 * 7)) + rv1))]);
      }
    }
  }
  if (((int)threadIdx.y) < 1) {
    tensor[((((((int)threadIdx.y) * 1024) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x)))] = (tensor1[(0)] * 2.040816e-02f);
  }
}

extern "C" __global__ void fused_nn_dense_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    T_dense_rf[(0)] = (T_dense_rf[(0)] + (placeholder[(((k_outer * 64) + ((int)threadIdx.x)))] * placeholder1[((((((int)blockIdx.x) * 1024) + (k_outer * 64)) + ((int)threadIdx.x)))]));
  }
  __syncthreads();
  ((volatile float*)red_buf0)[(((int)threadIdx.x))] = T_dense_rf[(0)];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 32))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 16))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 8))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 4))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 2))]);
    ((volatile float*)red_buf0)[(((int)threadIdx.x))] = (((volatile float*)red_buf0)[(((int)threadIdx.x))] + ((volatile float*)red_buf0)[((((int)threadIdx.x) + 1))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_dense[(0)] = ((volatile float*)red_buf0)[(0)];
  }
  if (((int)threadIdx.x) == 0) {
    T_add[(((int)blockIdx.x))] = (T_dense[(0)] + placeholder2[(((int)blockIdx.x))]);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_8_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[8];
  __shared__ float pad_temp_shared[896];
  __shared__ float placeholder_shared[1024];
  compute[(0)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[((((((int)threadIdx.z) * 56) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[((((((rc_outer * 12544) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4)) < 64) {
        if ((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 1024) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 64) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[((((((((int)blockIdx.z) * 16384) + (((int)threadIdx.z) * 1024)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) >> 4) * 256)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) & 15)))];
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 16) + rc_inner))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 256))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 512))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((rc_inner * 56) + ((int)threadIdx.x)) + 28))] * placeholder_shared[((((((int)threadIdx.z) * 16) + rc_inner) + 768))]));
    }
  }
  T_relu[(((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 12544))] = max((compute[(2)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25088))] = max((compute[(4)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 37632))] = max((compute[(6)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 64) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 12572))] = max((compute[(3)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25116))] = max((compute[(5)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 32))]), 0.000000e+00f);
  T_relu[((((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 37660))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 64) + ((int)threadIdx.z)) + 48))]), 0.000000e+00f);
}

