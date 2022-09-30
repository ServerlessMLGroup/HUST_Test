
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif


#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(512) tvmgen_default_fused_nn_global_avg_pool2d_kernel1(float* __restrict__ tensor, float* __restrict__ tensor_1) {
  tensor[((int)threadIdx.x)] = (tensor_1[((int)threadIdx.x)] * 2.040816e-02f);
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_nn_max_pool2d_add_nn_relu_kernel0(float* __restrict__ p0, float* __restrict__ T_relu, float* __restrict__ p1) {
  float tensor[1];
  tensor[0] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor[0] = max(tensor[0], (((1 <= ((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) % 392) / 7) * 2) + rv0)) && (1 <= (((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 56) * 2) + rv1))) ? p0[((((((((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 7) * 224) + (rv0 * 112)) + ((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) % 56) * 2)) + rv1) - 113)] : -3.402823e+38f));
    }
  }
  T_relu[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max((tensor[0] + p1[(((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 49)]), 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_conv2d_add_nn_relu_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ T_relu, float* __restrict__ p2) {
  float conv2d_nchw[56];
  __shared__ float pad_temp_shared[229];
  __shared__ float p1_shared[448];
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    conv2d_nchw[ff_init] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 14)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 28)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 42)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 2)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 16)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 30)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 44)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 4)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 18)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 32)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 46)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 6)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 20)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 34)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 48)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 8)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 22)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 36)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 50)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 10)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 24)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 38)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 52)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 12)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 26)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 40)] = 0.000000e+00f;
    conv2d_nchw[(ff_init + 54)] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 3; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 7; ++ry_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
        if ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 229) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 29) {
            pad_temp_shared[(((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = (((((3 <= ((((int)blockIdx.y) * 2) + ry_outer)) && (((((int)blockIdx.y) * 2) + ry_outer) < 227)) && (3 <= (((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))) && ((((((int)threadIdx.z) * 29) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 227)) ? p0[(((((((rc_outer * 50176) + (((int)blockIdx.y) * 448)) + (ry_outer * 224)) + (((int)threadIdx.z) * 29)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) - 675)] : 0.000000e+00f);
          }
        }
      }
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) {
        if (((((int)threadIdx.x) / 14) + ((int)threadIdx.z)) < 8) {
          if (((int)threadIdx.x) < 14) {
            p1_shared[(((((int)threadIdx.z) * 56) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1)] = p1[(((((((int)threadIdx.z) * 1176) + ((((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) / 7) * 147)) + (rc_outer * 49)) + (ry_outer * 7)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) % 7))];
          }
        }
      }
      __syncthreads();
      for (int rx_inner = 0; rx_inner < 7; ++rx_inner) {
        for (int ff = 0; ff < 2; ++ff) {
          conv2d_nchw[ff] = (conv2d_nchw[ff] + (pad_temp_shared[((((int)threadIdx.x) * 2) + rx_inner)] * p1_shared[(((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner)]));
          conv2d_nchw[(ff + 14)] = (conv2d_nchw[(ff + 14)] + (pad_temp_shared[((((int)threadIdx.x) * 2) + rx_inner)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112)]));
          conv2d_nchw[(ff + 28)] = (conv2d_nchw[(ff + 28)] + (pad_temp_shared[((((int)threadIdx.x) * 2) + rx_inner)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224)]));
          conv2d_nchw[(ff + 42)] = (conv2d_nchw[(ff + 42)] + (pad_temp_shared[((((int)threadIdx.x) * 2) + rx_inner)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336)]));
          conv2d_nchw[(ff + 2)] = (conv2d_nchw[(ff + 2)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 32)] * p1_shared[(((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner)]));
          conv2d_nchw[(ff + 16)] = (conv2d_nchw[(ff + 16)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 32)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112)]));
          conv2d_nchw[(ff + 30)] = (conv2d_nchw[(ff + 30)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 32)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224)]));
          conv2d_nchw[(ff + 44)] = (conv2d_nchw[(ff + 44)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 32)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336)]));
          conv2d_nchw[(ff + 4)] = (conv2d_nchw[(ff + 4)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 64)] * p1_shared[(((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner)]));
          conv2d_nchw[(ff + 18)] = (conv2d_nchw[(ff + 18)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 64)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112)]));
          conv2d_nchw[(ff + 32)] = (conv2d_nchw[(ff + 32)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 64)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224)]));
          conv2d_nchw[(ff + 46)] = (conv2d_nchw[(ff + 46)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 64)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336)]));
          conv2d_nchw[(ff + 6)] = (conv2d_nchw[(ff + 6)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 96)] * p1_shared[(((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner)]));
          conv2d_nchw[(ff + 20)] = (conv2d_nchw[(ff + 20)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 96)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112)]));
          conv2d_nchw[(ff + 34)] = (conv2d_nchw[(ff + 34)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 96)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224)]));
          conv2d_nchw[(ff + 48)] = (conv2d_nchw[(ff + 48)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 96)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336)]));
          conv2d_nchw[(ff + 8)] = (conv2d_nchw[(ff + 8)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 128)] * p1_shared[(((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner)]));
          conv2d_nchw[(ff + 22)] = (conv2d_nchw[(ff + 22)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 128)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112)]));
          conv2d_nchw[(ff + 36)] = (conv2d_nchw[(ff + 36)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 128)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224)]));
          conv2d_nchw[(ff + 50)] = (conv2d_nchw[(ff + 50)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 128)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336)]));
          conv2d_nchw[(ff + 10)] = (conv2d_nchw[(ff + 10)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 160)] * p1_shared[(((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner)]));
          conv2d_nchw[(ff + 24)] = (conv2d_nchw[(ff + 24)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 160)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112)]));
          conv2d_nchw[(ff + 38)] = (conv2d_nchw[(ff + 38)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 160)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224)]));
          conv2d_nchw[(ff + 52)] = (conv2d_nchw[(ff + 52)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 160)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336)]));
          conv2d_nchw[(ff + 12)] = (conv2d_nchw[(ff + 12)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 192)] * p1_shared[(((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner)]));
          conv2d_nchw[(ff + 26)] = (conv2d_nchw[(ff + 26)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 192)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 112)]));
          conv2d_nchw[(ff + 40)] = (conv2d_nchw[(ff + 40)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 192)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 224)]));
          conv2d_nchw[(ff + 54)] = (conv2d_nchw[(ff + 54)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + rx_inner) + 192)] * p1_shared[((((((int)threadIdx.z) * 14) + (ff * 7)) + rx_inner) + 336)]));
        }
      }
    }
  }
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    T_relu[((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x))] = max((conv2d_nchw[ax1_inner_inner_inner] + p2[((((int)threadIdx.z) * 2) + ax1_inner_inner_inner)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200704)] = max((conv2d_nchw[(ax1_inner_inner_inner + 14)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401408)] = max((conv2d_nchw[(ax1_inner_inner_inner + 28)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602112)] = max((conv2d_nchw[(ax1_inner_inner_inner + 42)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 16)] = max((conv2d_nchw[(ax1_inner_inner_inner + 2)] + p2[((((int)threadIdx.z) * 2) + ax1_inner_inner_inner)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200720)] = max((conv2d_nchw[(ax1_inner_inner_inner + 16)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401424)] = max((conv2d_nchw[(ax1_inner_inner_inner + 30)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602128)] = max((conv2d_nchw[(ax1_inner_inner_inner + 44)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 32)] = max((conv2d_nchw[(ax1_inner_inner_inner + 4)] + p2[((((int)threadIdx.z) * 2) + ax1_inner_inner_inner)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200736)] = max((conv2d_nchw[(ax1_inner_inner_inner + 18)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401440)] = max((conv2d_nchw[(ax1_inner_inner_inner + 32)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602144)] = max((conv2d_nchw[(ax1_inner_inner_inner + 46)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 48)] = max((conv2d_nchw[(ax1_inner_inner_inner + 6)] + p2[((((int)threadIdx.z) * 2) + ax1_inner_inner_inner)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200752)] = max((conv2d_nchw[(ax1_inner_inner_inner + 20)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401456)] = max((conv2d_nchw[(ax1_inner_inner_inner + 34)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602160)] = max((conv2d_nchw[(ax1_inner_inner_inner + 48)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 64)] = max((conv2d_nchw[(ax1_inner_inner_inner + 8)] + p2[((((int)threadIdx.z) * 2) + ax1_inner_inner_inner)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200768)] = max((conv2d_nchw[(ax1_inner_inner_inner + 22)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401472)] = max((conv2d_nchw[(ax1_inner_inner_inner + 36)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602176)] = max((conv2d_nchw[(ax1_inner_inner_inner + 50)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 80)] = max((conv2d_nchw[(ax1_inner_inner_inner + 10)] + p2[((((int)threadIdx.z) * 2) + ax1_inner_inner_inner)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200784)] = max((conv2d_nchw[(ax1_inner_inner_inner + 24)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401488)] = max((conv2d_nchw[(ax1_inner_inner_inner + 38)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602192)] = max((conv2d_nchw[(ax1_inner_inner_inner + 52)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 96)] = max((conv2d_nchw[(ax1_inner_inner_inner + 12)] + p2[((((int)threadIdx.z) * 2) + ax1_inner_inner_inner)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 200800)] = max((conv2d_nchw[(ax1_inner_inner_inner + 26)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 16)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 401504)] = max((conv2d_nchw[(ax1_inner_inner_inner + 40)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 32)]), 0.000000e+00f);
    T_relu[(((((((int)threadIdx.z) * 25088) + (ax1_inner_inner_inner * 12544)) + (((int)blockIdx.y) * 112)) + ((int)threadIdx.x)) + 602208)] = max((conv2d_nchw[(ax1_inner_inner_inner + 54)] + p2[(((((int)threadIdx.z) * 2) + ax1_inner_inner_inner) + 48)]), 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_2_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[((eps * 4) + nu)] = (((((1 <= ((((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps)) && (((((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps) < 15)) && (1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2) + nu))) && ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2) + nu) < 15)) ? p0[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 196) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 28)) + (eps * 14)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2)) + nu) - 15)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[8] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[10] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + (d[1] * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[2]);
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.000000e+00f));
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + d[1]);
  data_pack_local[2] = (data_pack_local[2] + d[2]);
  data_pack_local[2] = (data_pack_local[2] + (d[9] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.000000e+00f));
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[3]);
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[11] * -1.000000e+00f));
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[4] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + d[8]);
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.000000e+00f));
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[6] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[9] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[10]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + (d[5] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[6] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[9]);
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[7] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + d[11]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + d[4]);
  data_pack_local[8] = (data_pack_local[8] + (d[6] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[8]);
  data_pack_local[8] = (data_pack_local[8] + (d[10] * -1.000000e+00f));
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[5] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[6]);
  data_pack_local[9] = (data_pack_local[9] + (d[9] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + d[5]);
  data_pack_local[10] = (data_pack_local[10] + d[6]);
  data_pack_local[10] = (data_pack_local[10] + d[9]);
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + (d[5] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[4] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[12]);
  data_pack_local[12] = (data_pack_local[12] + (d[14] * -1.000000e+00f));
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[6] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[14]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + (d[5] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[6] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[13]);
  data_pack_local[14] = (data_pack_local[14] + d[14]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[7] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[13] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[15]);
  for (int eps_1 = 0; eps_1 < 4; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 4; ++nu_1) {
      data_pack[((((eps_1 * 50176) + (nu_1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 4) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[36];
  float data_pack_local[36];
  for (int eps = 0; eps < 6; ++eps) {
    for (int nu = 0; nu < 6; ++nu) {
      d[((eps * 6) + nu)] = (((((1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 4) + eps)) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 4) + eps) < 57)) && (1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4) + nu))) && ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4) + nu) < 57)) ? p0[((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 224) + (eps * 56)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4)) + nu) - 57)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[1] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[3] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + d[4]);
  data_pack_local[0] = (data_pack_local[0] + (d[6] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[7] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[8] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[9] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[10] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[12] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[13] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[14] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[15] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[16] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[18] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[19] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[20] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[21] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[22] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + d[24]);
  data_pack_local[0] = (data_pack_local[0] + (d[25] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[26] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[27] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + d[28]);
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + d[1]);
  data_pack_local[1] = (data_pack_local[1] + (d[2] * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[3] * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + d[4]);
  data_pack_local[1] = (data_pack_local[1] + (d[7] * -1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[8] * -1.500000e+00f) * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[13] * -2.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[14] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[15] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + (d[16] * -2.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[19] * 1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[20] * 1.500000e+00f) * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[21] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + (d[22] * 1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[25]);
  data_pack_local[1] = (data_pack_local[1] + (d[26] * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[27] * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + d[28]);
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + (d[1] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[2] * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + (d[3] * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + d[4]);
  data_pack_local[2] = (data_pack_local[2] + ((d[7] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[8] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + ((d[9] * -1.500000e+00f) * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[13] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[14] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + ((d[15] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[16] * -2.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[19] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[20] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + ((d[21] * 1.500000e+00f) * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[22] * 1.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[25] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[26] * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + (d[27] * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + d[28]);
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[2] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[3] * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[4]);
  data_pack_local[3] = (data_pack_local[3] + ((d[7] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[8] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.500000e+00f) * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[10] * -1.500000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[13] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[14] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[15] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[16] * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[19] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[20] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[21] * 1.500000e+00f) * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[22] * 1.500000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[25] * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[26] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[27] * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[28]);
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[1] * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[2] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + (d[3] * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + d[4]);
  data_pack_local[4] = (data_pack_local[4] + ((d[7] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + ((d[8] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[9] * -1.500000e+00f) * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.500000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[13] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + ((d[14] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[15] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[16] * -2.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[19] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + ((d[20] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[21] * 1.500000e+00f) * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[22] * 1.500000e+00f));
  data_pack_local[4] = (data_pack_local[4] + (d[25] * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[26] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + (d[27] * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + d[28]);
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + d[1]);
  data_pack_local[5] = (data_pack_local[5] + (d[2] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[3] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[4] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[5]);
  data_pack_local[5] = (data_pack_local[5] + (d[7] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[8] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[9] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[10] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[11] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[13] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[14] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[15] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[16] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[17] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[19] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[20] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[21] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[22] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[23] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[25]);
  data_pack_local[5] = (data_pack_local[5] + (d[26] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[27] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[28] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[29]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + d[6]);
  data_pack_local[6] = (data_pack_local[6] + (d[7] * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[8] * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[9] * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[6] = (data_pack_local[6] + (d[12] * -2.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[13] * -2.500000e+00f) * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[14] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[15] * -2.500000e+00f) * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[16] * -2.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[18] * 5.000000e-01f));
  data_pack_local[6] = (data_pack_local[6] + ((d[19] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[20] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[21] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[22] * 5.000000e-01f));
  data_pack_local[6] = (data_pack_local[6] + d[24]);
  data_pack_local[6] = (data_pack_local[6] + (d[25] * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[26] * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[27] * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[28]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + d[7]);
  data_pack_local[7] = (data_pack_local[7] + (d[8] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + d[10]);
  data_pack_local[7] = (data_pack_local[7] + (d[13] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + ((d[14] * -2.500000e+00f) * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + ((d[15] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + (d[16] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[19] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + ((d[20] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + ((d[21] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + (d[22] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + d[25]);
  data_pack_local[7] = (data_pack_local[7] + (d[26] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[27] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + d[28]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + (d[7] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[8] * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + (d[9] * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[10]);
  data_pack_local[8] = (data_pack_local[8] + ((d[13] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + ((d[14] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + ((d[15] * -2.500000e+00f) * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[16] * -2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + ((d[19] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + ((d[20] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + ((d[21] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[22] * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + (d[25] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[26] * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + (d[27] * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[28]);
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[7] * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[8] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[9] * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[9] = (data_pack_local[9] + ((d[13] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[14] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[15] * -2.500000e+00f) * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[16] * -2.500000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[19] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[20] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[21] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[22] * 5.000000e-01f));
  data_pack_local[9] = (data_pack_local[9] + (d[25] * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[26] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[27] * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[28]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + (d[7] * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[8] * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + (d[9] * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[10] = (data_pack_local[10] + ((d[13] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + ((d[14] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + ((d[15] * -2.500000e+00f) * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[16] * -2.500000e+00f));
  data_pack_local[10] = (data_pack_local[10] + ((d[19] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + ((d[20] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + ((d[21] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[22] * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[25] * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[26] * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + (d[27] * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + d[28]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[8] * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[10] * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[11] = (data_pack_local[11] + (d[13] * -2.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[14] * -2.500000e+00f) * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[15] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[16] * -2.500000e+00f) * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[17] * -2.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[19] * 5.000000e-01f));
  data_pack_local[11] = (data_pack_local[11] + ((d[20] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[21] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[22] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[23] * 5.000000e-01f));
  data_pack_local[11] = (data_pack_local[11] + d[25]);
  data_pack_local[11] = (data_pack_local[11] + (d[26] * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[27] * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[28] * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[29]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[6] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[7] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[8] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[9] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[10] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[12] * 5.000000e-01f));
  data_pack_local[12] = (data_pack_local[12] + ((d[13] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[14] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[15] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[16] * 5.000000e-01f));
  data_pack_local[12] = (data_pack_local[12] + (d[18] * 2.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[19] * 2.500000e+00f) * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[20] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[21] * 2.500000e+00f) * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[22] * 2.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[24]);
  data_pack_local[12] = (data_pack_local[12] + (d[25] * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[26] * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[27] * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[28]);
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + (d[7] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[8] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[9] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[10] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + ((d[14] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[15] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[16] * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[19] * 2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[20] * 2.500000e+00f) * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[21] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[22] * 2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[25]);
  data_pack_local[13] = (data_pack_local[13] + (d[26] * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[27] * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + d[28]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + ((d[7] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[8] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[9] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[10] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[13] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[14] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[15] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[16] * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[19] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[20] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[21] * 2.500000e+00f) * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[22] * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[25] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[26] * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + (d[27] * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[28]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[7] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[8] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[9] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[10] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[13] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[14] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[15] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[16] * 5.000000e-01f));
  data_pack_local[15] = (data_pack_local[15] + ((d[19] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[20] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[21] * 2.500000e+00f) * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[22] * 2.500000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[25] * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[26] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[27] * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[28]);
  data_pack_local[16] = 0.000000e+00f;
  data_pack_local[16] = (data_pack_local[16] + ((d[7] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[8] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[9] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[10] * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[13] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[14] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[15] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[16] * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[19] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[20] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[21] * 2.500000e+00f) * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[22] * 2.500000e+00f));
  data_pack_local[16] = (data_pack_local[16] + (d[25] * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[26] * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + (d[27] * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + d[28]);
  data_pack_local[17] = 0.000000e+00f;
  data_pack_local[17] = (data_pack_local[17] + (d[7] * -1.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[8] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[9] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[10] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[11] * -1.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[13] * 5.000000e-01f));
  data_pack_local[17] = (data_pack_local[17] + ((d[14] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[15] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[16] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[17] * 5.000000e-01f));
  data_pack_local[17] = (data_pack_local[17] + (d[19] * 2.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[20] * 2.500000e+00f) * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[21] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[22] * 2.500000e+00f) * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[23] * 2.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + d[25]);
  data_pack_local[17] = (data_pack_local[17] + (d[26] * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[27] * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[28] * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + d[29]);
  data_pack_local[18] = 0.000000e+00f;
  data_pack_local[18] = (data_pack_local[18] + (d[6] * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[7] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[8] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[9] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[10] * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[12] * -1.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[13] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[14] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[15] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[16] * -1.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[18] * 2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[19] * 2.000000e+00f) * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[20] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[21] * 2.000000e+00f) * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[22] * 2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + d[24]);
  data_pack_local[18] = (data_pack_local[18] + (d[25] * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[26] * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[27] * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + d[28]);
  data_pack_local[19] = 0.000000e+00f;
  data_pack_local[19] = (data_pack_local[19] + (d[7] * -2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[8] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[9] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + (d[10] * -2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + (d[13] * -1.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[14] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[15] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + (d[16] * -1.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + (d[19] * 2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[20] * 2.000000e+00f) * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[21] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + (d[22] * 2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + d[25]);
  data_pack_local[19] = (data_pack_local[19] + (d[26] * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + (d[27] * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + d[28]);
  data_pack_local[20] = 0.000000e+00f;
  data_pack_local[20] = (data_pack_local[20] + ((d[7] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[8] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + ((d[9] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[10] * -2.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[13] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[14] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + ((d[15] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[16] * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[19] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[20] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + ((d[21] * 2.000000e+00f) * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[22] * 2.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[25] * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[26] * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + (d[27] * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + d[28]);
  data_pack_local[21] = 0.000000e+00f;
  data_pack_local[21] = (data_pack_local[21] + ((d[7] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[8] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[9] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[10] * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[13] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[15] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[16] * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[19] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[20] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[21] * 2.000000e+00f) * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[22] * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[25] * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[26] * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[27] * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + d[28]);
  data_pack_local[22] = 0.000000e+00f;
  data_pack_local[22] = (data_pack_local[22] + ((d[7] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + ((d[8] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[9] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[10] * -2.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[13] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[15] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[16] * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[19] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + ((d[20] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[21] * 2.000000e+00f) * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[22] * 2.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + (d[25] * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[26] * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + (d[27] * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + d[28]);
  data_pack_local[23] = 0.000000e+00f;
  data_pack_local[23] = (data_pack_local[23] + (d[7] * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[8] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[9] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[10] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[11] * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[13] * -1.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[14] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[15] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[16] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[17] * -1.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[19] * 2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[20] * 2.000000e+00f) * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[21] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[22] * 2.000000e+00f) * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[23] * 2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + d[25]);
  data_pack_local[23] = (data_pack_local[23] + (d[26] * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[27] * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[28] * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + d[29]);
  data_pack_local[24] = 0.000000e+00f;
  data_pack_local[24] = (data_pack_local[24] + (d[6] * 5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + ((d[7] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[8] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[9] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[10] * 5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + (d[12] * -1.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[13] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[14] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[15] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[16] * -1.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[18] * -5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + ((d[19] * -5.000000e-01f) * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[20] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[21] * -5.000000e-01f) * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[22] * -5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + d[24]);
  data_pack_local[24] = (data_pack_local[24] + (d[25] * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[26] * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[27] * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + d[28]);
  data_pack_local[25] = 0.000000e+00f;
  data_pack_local[25] = (data_pack_local[25] + (d[7] * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + ((d[8] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[9] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[10] * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[13] * -1.000000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[14] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[15] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[16] * -1.000000e+00f));
  data_pack_local[25] = (data_pack_local[25] + (d[19] * -5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + ((d[20] * -5.000000e-01f) * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[21] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[22] * -5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + d[25]);
  data_pack_local[25] = (data_pack_local[25] + (d[26] * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + (d[27] * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + d[28]);
  data_pack_local[26] = 0.000000e+00f;
  data_pack_local[26] = (data_pack_local[26] + ((d[7] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[8] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[9] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[10] * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[13] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[14] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[15] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[16] * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[19] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[20] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[21] * -5.000000e-01f) * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[22] * -5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + (d[25] * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[26] * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + (d[27] * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + d[28]);
  data_pack_local[27] = 0.000000e+00f;
  data_pack_local[27] = (data_pack_local[27] + ((d[7] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[8] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[9] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[10] * 5.000000e-01f));
  data_pack_local[27] = (data_pack_local[27] + ((d[13] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[15] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[16] * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[19] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[20] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[21] * -5.000000e-01f) * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[22] * -5.000000e-01f));
  data_pack_local[27] = (data_pack_local[27] + (d[25] * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[26] * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[27] * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + d[28]);
  data_pack_local[28] = 0.000000e+00f;
  data_pack_local[28] = (data_pack_local[28] + ((d[7] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[8] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[9] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[10] * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[13] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[15] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[16] * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[19] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[20] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[21] * -5.000000e-01f) * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[22] * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[25] * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[26] * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + (d[27] * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + d[28]);
  data_pack_local[29] = 0.000000e+00f;
  data_pack_local[29] = (data_pack_local[29] + (d[7] * 5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + ((d[8] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[9] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[10] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[11] * 5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + (d[13] * -1.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[14] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[15] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[16] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[17] * -1.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[19] * -5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + ((d[20] * -5.000000e-01f) * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[21] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[22] * -5.000000e-01f) * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[23] * -5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + d[25]);
  data_pack_local[29] = (data_pack_local[29] + (d[26] * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[27] * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[28] * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + d[29]);
  data_pack_local[30] = 0.000000e+00f;
  data_pack_local[30] = (data_pack_local[30] + d[6]);
  data_pack_local[30] = (data_pack_local[30] + (d[7] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[8] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[9] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + d[10]);
  data_pack_local[30] = (data_pack_local[30] + (d[12] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[13] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[14] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[15] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[16] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[18] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[19] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[20] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[21] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[22] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[24] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[25] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[26] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[27] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[28] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + d[30]);
  data_pack_local[30] = (data_pack_local[30] + (d[31] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[32] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[33] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + d[34]);
  data_pack_local[31] = 0.000000e+00f;
  data_pack_local[31] = (data_pack_local[31] + d[7]);
  data_pack_local[31] = (data_pack_local[31] + (d[8] * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[9] * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + d[10]);
  data_pack_local[31] = (data_pack_local[31] + (d[13] * -1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[14] * -1.500000e+00f) * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[15] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + (d[16] * -1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[19] * -2.000000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[20] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[21] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + (d[22] * -2.000000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[25] * 1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[26] * 1.500000e+00f) * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[27] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + (d[28] * 1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + d[31]);
  data_pack_local[31] = (data_pack_local[31] + (d[32] * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[33] * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + d[34]);
  data_pack_local[32] = 0.000000e+00f;
  data_pack_local[32] = (data_pack_local[32] + (d[7] * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[8] * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + (d[9] * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + d[10]);
  data_pack_local[32] = (data_pack_local[32] + ((d[13] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[14] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + ((d[15] * -1.500000e+00f) * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[16] * -1.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[19] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[20] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + ((d[21] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[22] * -2.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[25] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[26] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + ((d[27] * 1.500000e+00f) * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[28] * 1.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[31] * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[32] * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + (d[33] * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + d[34]);
  data_pack_local[33] = 0.000000e+00f;
  data_pack_local[33] = (data_pack_local[33] + (d[7] * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[8] * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[9] * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + d[10]);
  data_pack_local[33] = (data_pack_local[33] + ((d[13] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[14] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[15] * -1.500000e+00f) * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[16] * -1.500000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[19] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[20] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[21] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[22] * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[25] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[26] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[27] * 1.500000e+00f) * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[28] * 1.500000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[31] * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[32] * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[33] * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + d[34]);
  data_pack_local[34] = 0.000000e+00f;
  data_pack_local[34] = (data_pack_local[34] + (d[7] * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[8] * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + (d[9] * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + d[10]);
  data_pack_local[34] = (data_pack_local[34] + ((d[13] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + ((d[14] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[15] * -1.500000e+00f) * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[16] * -1.500000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[19] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + ((d[20] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[21] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[22] * -2.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[25] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + ((d[26] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[27] * 1.500000e+00f) * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[28] * 1.500000e+00f));
  data_pack_local[34] = (data_pack_local[34] + (d[31] * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[32] * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + (d[33] * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + d[34]);
  data_pack_local[35] = 0.000000e+00f;
  data_pack_local[35] = (data_pack_local[35] + d[7]);
  data_pack_local[35] = (data_pack_local[35] + (d[8] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[9] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[10] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + d[11]);
  data_pack_local[35] = (data_pack_local[35] + (d[13] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[14] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[15] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[16] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[17] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[19] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[20] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[21] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[22] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[23] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[25] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[26] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[27] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[28] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[29] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + d[31]);
  data_pack_local[35] = (data_pack_local[35] + (d[32] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[33] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[34] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + d[35]);
  for (int eps_1 = 0; eps_1 < 6; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 6; ++nu_1) {
      data_pack[((((eps_1 * 75264) + (nu_1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 6) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_2_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[((eps * 4) + nu)] = (((((1 <= ((((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps)) && (((((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps) < 15)) && (1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2) + nu))) && ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2) + nu) < 15)) ? p0[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 196) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 28)) + (eps * 14)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2)) + nu) - 15)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[8] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[10] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + (d[1] * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[2]);
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.000000e+00f));
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + d[1]);
  data_pack_local[2] = (data_pack_local[2] + d[2]);
  data_pack_local[2] = (data_pack_local[2] + (d[9] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.000000e+00f));
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[3]);
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[11] * -1.000000e+00f));
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[4] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + d[8]);
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.000000e+00f));
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[6] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[9] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[10]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + (d[5] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[6] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[9]);
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[7] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + d[11]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + d[4]);
  data_pack_local[8] = (data_pack_local[8] + (d[6] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[8]);
  data_pack_local[8] = (data_pack_local[8] + (d[10] * -1.000000e+00f));
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[5] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[6]);
  data_pack_local[9] = (data_pack_local[9] + (d[9] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + d[5]);
  data_pack_local[10] = (data_pack_local[10] + d[6]);
  data_pack_local[10] = (data_pack_local[10] + d[9]);
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + (d[5] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[4] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[12]);
  data_pack_local[12] = (data_pack_local[12] + (d[14] * -1.000000e+00f));
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[6] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[14]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + (d[5] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[6] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[13]);
  data_pack_local[14] = (data_pack_local[14] + d[14]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[7] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[13] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[15]);
  for (int eps_1 = 0; eps_1 < 4; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 4; ++nu_1) {
      data_pack[((((eps_1 * 50176) + (nu_1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 4) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(224) tvmgen_default_fused_nn_conv2d_add_nn_relu_1_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ T_relu, float* __restrict__ p2) {
  float conv2d_nchw[4];
  __shared__ float pad_temp_shared[342];
  __shared__ float p1_shared[576];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) < 171) {
      if (((int)threadIdx.x) < 11) {
        pad_temp_shared[((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2))] = (((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) % 171) / 57))) && (1 <= (((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) % 57))) ? p0[((((((rc_outer * 6272) + ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) / 171) * 3136)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) % 171) / 57) * 56)) + (((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) % 57)) - 57)] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.z) * 11) + ((int)threadIdx.x)) < 171) {
      if (((int)threadIdx.x) < 11) {
        pad_temp_shared[(((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) + 1)] = (((1 <= ((((int)blockIdx.y) * 2) + (((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) + 1) % 171) / 57))) && (1 <= ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) + 1) % 57))) ? p0[((((((rc_outer * 6272) + (((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) + 1) / 171) * 3136)) + (((int)blockIdx.y) * 112)) + ((((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) + 1) % 171) / 57) * 56)) + ((((((int)threadIdx.z) * 22) + (((int)threadIdx.x) * 2)) + 1) % 57)) - 57)] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.x) / 12) + ((int)threadIdx.z)) < 16) {
      if (((int)threadIdx.x) < 12) {
        p1_shared[((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3))] = p1[(((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 1152)) + ((((int)threadIdx.x) / 6) * 576)) + (rc_outer * 18)) + ((((int)threadIdx.x) % 6) * 3))];
      }
    }
    if (((((int)threadIdx.x) / 12) + ((int)threadIdx.z)) < 16) {
      if (((int)threadIdx.x) < 12) {
        p1_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 1)] = p1[((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 1152)) + ((((int)threadIdx.x) / 6) * 576)) + (rc_outer * 18)) + ((((int)threadIdx.x) % 6) * 3)) + 1)];
      }
    }
    if (((((int)threadIdx.x) / 12) + ((int)threadIdx.z)) < 16) {
      if (((int)threadIdx.x) < 12) {
        p1_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 3)) + 2)] = p1[((((((((int)blockIdx.z) * 18432) + (((int)threadIdx.z) * 1152)) + ((((int)threadIdx.x) / 6) * 576)) + (rc_outer * 18)) + ((((int)threadIdx.x) % 6) * 3)) + 2)];
      }
    }
    __syncthreads();
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] * p1_shared[(((int)threadIdx.z) * 18)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) * 2)] * p1_shared[((((int)threadIdx.z) * 18) + 288)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 28)] * p1_shared[(((int)threadIdx.z) * 18)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 28)] * p1_shared[((((int)threadIdx.z) * 18) + 288)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * p1_shared[((((int)threadIdx.z) * 18) + 1)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * p1_shared[((((int)threadIdx.z) * 18) + 289)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 29)] * p1_shared[((((int)threadIdx.z) * 18) + 1)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 29)] * p1_shared[((((int)threadIdx.z) * 18) + 289)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 2)] * p1_shared[((((int)threadIdx.z) * 18) + 2)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 2)] * p1_shared[((((int)threadIdx.z) * 18) + 290)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 30)] * p1_shared[((((int)threadIdx.z) * 18) + 2)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 30)] * p1_shared[((((int)threadIdx.z) * 18) + 290)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 57)] * p1_shared[((((int)threadIdx.z) * 18) + 3)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 57)] * p1_shared[((((int)threadIdx.z) * 18) + 291)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 85)] * p1_shared[((((int)threadIdx.z) * 18) + 3)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 85)] * p1_shared[((((int)threadIdx.z) * 18) + 291)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 58)] * p1_shared[((((int)threadIdx.z) * 18) + 4)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 58)] * p1_shared[((((int)threadIdx.z) * 18) + 292)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 86)] * p1_shared[((((int)threadIdx.z) * 18) + 4)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 86)] * p1_shared[((((int)threadIdx.z) * 18) + 292)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 59)] * p1_shared[((((int)threadIdx.z) * 18) + 5)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 59)] * p1_shared[((((int)threadIdx.z) * 18) + 293)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 87)] * p1_shared[((((int)threadIdx.z) * 18) + 5)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 87)] * p1_shared[((((int)threadIdx.z) * 18) + 293)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 114)] * p1_shared[((((int)threadIdx.z) * 18) + 6)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 114)] * p1_shared[((((int)threadIdx.z) * 18) + 294)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 142)] * p1_shared[((((int)threadIdx.z) * 18) + 6)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 142)] * p1_shared[((((int)threadIdx.z) * 18) + 294)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 115)] * p1_shared[((((int)threadIdx.z) * 18) + 7)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 115)] * p1_shared[((((int)threadIdx.z) * 18) + 295)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 143)] * p1_shared[((((int)threadIdx.z) * 18) + 7)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 143)] * p1_shared[((((int)threadIdx.z) * 18) + 295)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 116)] * p1_shared[((((int)threadIdx.z) * 18) + 8)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 116)] * p1_shared[((((int)threadIdx.z) * 18) + 296)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] * p1_shared[((((int)threadIdx.z) * 18) + 8)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 144)] * p1_shared[((((int)threadIdx.z) * 18) + 296)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 171)] * p1_shared[((((int)threadIdx.z) * 18) + 9)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 171)] * p1_shared[((((int)threadIdx.z) * 18) + 297)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 199)] * p1_shared[((((int)threadIdx.z) * 18) + 9)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 199)] * p1_shared[((((int)threadIdx.z) * 18) + 297)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 172)] * p1_shared[((((int)threadIdx.z) * 18) + 10)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 172)] * p1_shared[((((int)threadIdx.z) * 18) + 298)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 200)] * p1_shared[((((int)threadIdx.z) * 18) + 10)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 200)] * p1_shared[((((int)threadIdx.z) * 18) + 298)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 173)] * p1_shared[((((int)threadIdx.z) * 18) + 11)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 173)] * p1_shared[((((int)threadIdx.z) * 18) + 299)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 201)] * p1_shared[((((int)threadIdx.z) * 18) + 11)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 201)] * p1_shared[((((int)threadIdx.z) * 18) + 299)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 228)] * p1_shared[((((int)threadIdx.z) * 18) + 12)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 228)] * p1_shared[((((int)threadIdx.z) * 18) + 300)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 256)] * p1_shared[((((int)threadIdx.z) * 18) + 12)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 256)] * p1_shared[((((int)threadIdx.z) * 18) + 300)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 229)] * p1_shared[((((int)threadIdx.z) * 18) + 13)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 229)] * p1_shared[((((int)threadIdx.z) * 18) + 301)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 257)] * p1_shared[((((int)threadIdx.z) * 18) + 13)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 257)] * p1_shared[((((int)threadIdx.z) * 18) + 301)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 230)] * p1_shared[((((int)threadIdx.z) * 18) + 14)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 230)] * p1_shared[((((int)threadIdx.z) * 18) + 302)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 258)] * p1_shared[((((int)threadIdx.z) * 18) + 14)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 258)] * p1_shared[((((int)threadIdx.z) * 18) + 302)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 285)] * p1_shared[((((int)threadIdx.z) * 18) + 15)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 285)] * p1_shared[((((int)threadIdx.z) * 18) + 303)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 313)] * p1_shared[((((int)threadIdx.z) * 18) + 15)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 313)] * p1_shared[((((int)threadIdx.z) * 18) + 303)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 286)] * p1_shared[((((int)threadIdx.z) * 18) + 16)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 286)] * p1_shared[((((int)threadIdx.z) * 18) + 304)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 314)] * p1_shared[((((int)threadIdx.z) * 18) + 16)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 314)] * p1_shared[((((int)threadIdx.z) * 18) + 304)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 287)] * p1_shared[((((int)threadIdx.z) * 18) + 17)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 287)] * p1_shared[((((int)threadIdx.z) * 18) + 305)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 315)] * p1_shared[((((int)threadIdx.z) * 18) + 17)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 315)] * p1_shared[((((int)threadIdx.z) * 18) + 305)]));
  }
  T_relu[((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x))] = max((conv2d_nchw[0] + p2[((((int)blockIdx.z) * 32) + ((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 12544)] = max((conv2d_nchw[2] + p2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16)]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 14)] = max((conv2d_nchw[1] + p2[((((int)blockIdx.z) * 32) + ((int)threadIdx.z))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 25088) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 12558)] = max((conv2d_nchw[3] + p2[(((((int)blockIdx.z) * 32) + ((int)threadIdx.z)) + 16)]), 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_add_kernel0(float* __restrict__ T_add, float* __restrict__ p0, float* __restrict__ p1) {
  T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (p0[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + p1[(((int)blockIdx.x) / 49)]);
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_3_kernel2(float* __restrict__ bgemm, float* __restrict__ T_add, float* __restrict__ p2) {
  float inverse[4];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 24576)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112)]);
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 98304)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688)]);
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 122880)]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      if (((((((int)threadIdx.x) & 15) >> 2) * 2) + ax2_inner) < 7) {
        if ((((((int)threadIdx.x) & 3) * 2) + ax3_inner) < 7) {
          T_add[((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (ax2_inner * 7)) + ((((int)threadIdx.x) & 3) * 2)) + ax3_inner)] = (inverse[((ax2_inner * 2) + ax3_inner)] + p2[((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (ax2_inner * 7)) + ((((int)threadIdx.x) & 3) * 2)) + ax3_inner)]);
        }
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ p2) {
  float inverse[16];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f));
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f));
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)]);
  inverse[4] = 0.000000e+00f;
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f));
  inverse[5] = 0.000000e+00f;
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -2.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[5] = (inverse[5] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f) * -1.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f) * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f) * -2.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f) * -1.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f) * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f) * -2.000000e+00f));
  inverse[6] = 0.000000e+00f;
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 2.500000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * 4.000000e+00f));
  inverse[6] = (inverse[6] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[6] = (inverse[6] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f) * 2.500000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f) * 4.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f) * 2.500000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f) * 4.000000e+00f));
  inverse[7] = 0.000000e+00f;
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 1.250000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -8.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)] * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[7] = (inverse[7] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[7] = (inverse[7] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f) * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f) * 1.250000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f) * -8.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)] * 5.000000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f) * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f) * 1.250000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f) * -8.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)] * -2.000000e+00f));
  inverse[8] = 0.000000e+00f;
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f));
  inverse[9] = 0.000000e+00f;
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[9] = (inverse[9] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 5.000000e-01f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -2.000000e+00f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[9] = (inverse[9] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f) * -1.000000e+00f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f) * 5.000000e-01f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f) * -2.000000e+00f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f) * -1.000000e+00f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f) * 5.000000e-01f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f) * -2.000000e+00f));
  inverse[10] = 0.000000e+00f;
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f) * 2.500000e-01f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f) * 4.000000e+00f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f) * 2.500000e-01f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f) * 4.000000e+00f));
  inverse[11] = 0.000000e+00f;
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 1.250000e-01f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -8.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)]);
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f) * -1.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f) * 1.250000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f) * -8.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)] * 2.500000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f) * -1.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f) * 1.250000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f) * -8.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)] * 4.000000e+00f));
  inverse[12] = 0.000000e+00f;
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)]);
  inverse[13] = 0.000000e+00f;
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 5.000000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -2.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[13] = (inverse[13] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f) * -1.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f) * 5.000000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f) * -2.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f) * -1.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f) * 5.000000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f) * -2.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)] * -1.000000e+00f));
  inverse[13] = (inverse[13] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)] * 5.000000e-01f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)] * -2.000000e+00f));
  inverse[14] = 0.000000e+00f;
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 2.500000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * 4.000000e+00f));
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f) * 2.500000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f) * 4.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f) * 2.500000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f) * 4.000000e+00f));
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)]);
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)] * 2.500000e-01f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)] * 4.000000e+00f));
  inverse[15] = 0.000000e+00f;
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f) * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f) * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f) * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f) * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f) * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f) * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 439040)]);
  for (int ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 4; ++ax3_inner) {
      T_relu[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 224) + (ax2_inner * 56)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4)) + ax3_inner)] = max((inverse[((ax2_inner * 4) + ax3_inner)] + p2[(((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2)) / 49)]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_1_kernel2(float* __restrict__ bgemm, float* __restrict__ T_add, float* __restrict__ p2) {
  float inverse[4];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)]);
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)]);
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320)]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_add[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner)] = (inverse[((ax2_inner * 2) + ax3_inner)] + p2[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner)]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_add_nn_relu_kernel0(float* __restrict__ T_relu, float* __restrict__ p0, float* __restrict__ p1) {
  T_relu[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max((p0[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + p1[(((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 49)]), 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_multiply_add_nn_re_a4e88f85cf7823fc__kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[((eps * 4) + nu)] = (((((1 <= ((((((int)threadIdx.x) & 15) >> 2) * 2) + eps)) && ((((((int)threadIdx.x) & 15) >> 2) + (eps >> 1)) < 4)) && (1 <= (((((int)threadIdx.x) & 3) * 2) + nu))) && (((nu >> 1) + (((int)threadIdx.x) & 3)) < 4)) ? p0[(((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (eps * 7)) + ((((int)threadIdx.x) & 3) * 2)) + nu) - 8)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[8] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[10] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + (d[1] * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[2]);
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.000000e+00f));
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + d[1]);
  data_pack_local[2] = (data_pack_local[2] + d[2]);
  data_pack_local[2] = (data_pack_local[2] + (d[9] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.000000e+00f));
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[3]);
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[11] * -1.000000e+00f));
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[4] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + d[8]);
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.000000e+00f));
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[6] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[9] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[10]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + (d[5] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[6] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[9]);
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[7] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + d[11]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + d[4]);
  data_pack_local[8] = (data_pack_local[8] + (d[6] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[8]);
  data_pack_local[8] = (data_pack_local[8] + (d[10] * -1.000000e+00f));
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[5] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[6]);
  data_pack_local[9] = (data_pack_local[9] + (d[9] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + d[5]);
  data_pack_local[10] = (data_pack_local[10] + d[6]);
  data_pack_local[10] = (data_pack_local[10] + d[9]);
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + (d[5] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[4] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[12]);
  data_pack_local[12] = (data_pack_local[12] + (d[14] * -1.000000e+00f));
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[6] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[14]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + (d[5] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[6] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[13]);
  data_pack_local[14] = (data_pack_local[14] + d[14]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[7] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[13] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[15]);
  for (int eps_1 = 0; eps_1 < 4; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 4; ++nu_1) {
      data_pack[((((eps_1 * 32768) + (nu_1 * 8192)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 4) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(448) tvmgen_default_fused_nn_conv2d_1_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[4];
  __shared__ float pad_temp_shared[880];
  __shared__ float p1_shared[1024];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((int)threadIdx.z) * 7) + (((int)threadIdx.x) >> 1)) < 220) {
        pad_temp_shared[(((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = p0[((((rc_outer * 50176) + (((((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 55) * 3136)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.z) * 28) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 55))];
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) {
      if (((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) >> 5) + ((int)threadIdx.z)) < 32) {
        if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) < 32) {
          p1_shared[(((((int)threadIdx.z) * 32) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1)] = p1[(((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 128)) + ((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) >> 4) * 64)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) & 15))];
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_inner * 55) + (((int)threadIdx.x) * 2))] * p1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_inner * 55) + (((int)threadIdx.x) * 2))] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 512)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 28)] * p1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_inner * 55) + (((int)threadIdx.x) * 2)) + 28)] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 512)]));
    }
  }
  conv2d_nchw[((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x))] = conv2d_nchw_local[0];
  conv2d_nchw[(((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 25088)] = conv2d_nchw_local[2];
  conv2d_nchw[(((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 14)] = conv2d_nchw_local[1];
  conv2d_nchw[(((((((int)blockIdx.z) * 50176) + (((int)threadIdx.z) * 784)) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 25102)] = conv2d_nchw_local[3];
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_2_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ p2, float* __restrict__ p3) {
  float inverse[4];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7))]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 12544)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 25088)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 50176)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 100352)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 12544)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 25088)]);
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 37632)]);
  inverse[1] = (inverse[1] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)]);
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 87808)]);
  inverse[1] = (inverse[1] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 137984)]);
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 50176)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 100352)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 150528)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 163072)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 175616)]);
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + ((bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 87808)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 137984)]);
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 163072)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 175616)]);
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 188160)]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_relu[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 7) * 28) + (ax2_inner * 14)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2)) + ax3_inner)] = max(((inverse[((ax2_inner * 2) + ax3_inner)] + p2[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 7) * 28) + (ax2_inner * 14)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2)) + ax3_inner)]) + p3[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49)]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel2(float* __restrict__ bgemm, float* __restrict__ T_add, float* __restrict__ p2) {
  float inverse[16];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f));
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f));
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)]);
  inverse[4] = 0.000000e+00f;
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f));
  inverse[5] = 0.000000e+00f;
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -2.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[5] = (inverse[5] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f) * -1.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f) * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f) * -2.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f) * -1.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f) * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f) * -2.000000e+00f));
  inverse[6] = 0.000000e+00f;
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 2.500000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * 4.000000e+00f));
  inverse[6] = (inverse[6] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[6] = (inverse[6] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f) * 2.500000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f) * 4.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f) * 2.500000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f) * 4.000000e+00f));
  inverse[7] = 0.000000e+00f;
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 1.250000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -8.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)] * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[7] = (inverse[7] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[7] = (inverse[7] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f) * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f) * 1.250000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f) * -8.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)] * 5.000000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f) * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f) * 1.250000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f) * -8.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)] * -2.000000e+00f));
  inverse[8] = 0.000000e+00f;
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f));
  inverse[9] = 0.000000e+00f;
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[9] = (inverse[9] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 5.000000e-01f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -2.000000e+00f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[9] = (inverse[9] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f) * -1.000000e+00f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f) * 5.000000e-01f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f) * -2.000000e+00f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f) * -1.000000e+00f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f) * 5.000000e-01f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f) * -2.000000e+00f));
  inverse[10] = 0.000000e+00f;
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f) * 2.500000e-01f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f) * 4.000000e+00f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f) * 2.500000e-01f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f) * 4.000000e+00f));
  inverse[11] = 0.000000e+00f;
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 1.250000e-01f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -8.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)]);
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f) * -1.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f) * 1.250000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f) * -8.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)] * 2.500000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f) * -1.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f) * 1.250000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f) * -8.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)] * 4.000000e+00f));
  inverse[12] = 0.000000e+00f;
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)]);
  inverse[13] = 0.000000e+00f;
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 5.000000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -2.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[13] = (inverse[13] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f) * -1.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f) * 5.000000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f) * -2.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f) * -1.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f) * 5.000000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f) * -2.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)] * -1.000000e+00f));
  inverse[13] = (inverse[13] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)] * 5.000000e-01f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)] * -2.000000e+00f));
  inverse[14] = 0.000000e+00f;
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 2.500000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * 4.000000e+00f));
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f) * 2.500000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f) * 4.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f) * 2.500000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f) * 4.000000e+00f));
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)]);
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)] * 2.500000e-01f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)] * 4.000000e+00f));
  inverse[15] = 0.000000e+00f;
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f) * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f) * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f) * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f) * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f) * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f) * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 439040)]);
  for (int ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 4; ++ax3_inner) {
      T_add[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 224) + (ax2_inner * 56)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4)) + ax3_inner)] = (inverse[((ax2_inner * 4) + ax3_inner)] + p2[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 224) + (ax2_inner * 56)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4)) + ax3_inner)]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(49) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_2_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[16];
  __shared__ float p1_shared[128];
  __shared__ float data_pack_shared[392];
  #pragma unroll
  for (int co_c_init = 0; co_c_init < 8; ++co_c_init) {
    bgemm_local[co_c_init] = 0.000000e+00f;
    bgemm_local[(co_c_init + 8)] = 0.000000e+00f;
  }
  for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)) < 128) {
        p1_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.z) * 65536) + (ci_outer * 2048)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) + ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) & 15))];
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1) {
      data_pack_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 49) + ((int)threadIdx.x))] = data_pack[((((((int)blockIdx.z) * 12544) + (ci_outer * 392)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 49)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    #pragma unroll
    for (int ci_inner = 0; ci_inner < 8; ++ci_inner) {
      #pragma unroll
      for (int co_c = 0; co_c < 8; ++co_c) {
        bgemm_local[co_c] = (bgemm_local[co_c] + (p1_shared[((ci_inner * 16) + co_c)] * data_pack_shared[((ci_inner * 49) + ((int)threadIdx.x))]));
        bgemm_local[(co_c + 8)] = (bgemm_local[(co_c + 8)] + (p1_shared[(((ci_inner * 16) + co_c) + 8)] * data_pack_shared[((ci_inner * 49) + ((int)threadIdx.x))]));
      }
    }
  }
  #pragma unroll
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 8; ++co_inner_inner_inner) {
    bgemm[((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (co_inner_inner_inner * 49)) + ((int)threadIdx.x))] = bgemm_local[co_inner_inner_inner];
    bgemm[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (co_inner_inner_inner * 49)) + ((int)threadIdx.x)) + 392)] = bgemm_local[(co_inner_inner_inner + 8)];
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[36];
  float data_pack_local[36];
  for (int eps = 0; eps < 6; ++eps) {
    for (int nu = 0; nu < 6; ++nu) {
      d[((eps * 6) + nu)] = (((((1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 4) + eps)) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 4) + eps) < 57)) && (1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4) + nu))) && ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4) + nu) < 57)) ? p0[((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 224) + (eps * 56)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4)) + nu) - 57)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[1] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[3] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + d[4]);
  data_pack_local[0] = (data_pack_local[0] + (d[6] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[7] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[8] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[9] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[10] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[12] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[13] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[14] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[15] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[16] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[18] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[19] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[20] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[21] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[22] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + d[24]);
  data_pack_local[0] = (data_pack_local[0] + (d[25] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[26] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[27] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + d[28]);
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + d[1]);
  data_pack_local[1] = (data_pack_local[1] + (d[2] * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[3] * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + d[4]);
  data_pack_local[1] = (data_pack_local[1] + (d[7] * -1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[8] * -1.500000e+00f) * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[13] * -2.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[14] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[15] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + (d[16] * -2.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[19] * 1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[20] * 1.500000e+00f) * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[21] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + (d[22] * 1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[25]);
  data_pack_local[1] = (data_pack_local[1] + (d[26] * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[27] * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + d[28]);
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + (d[1] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[2] * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + (d[3] * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + d[4]);
  data_pack_local[2] = (data_pack_local[2] + ((d[7] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[8] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + ((d[9] * -1.500000e+00f) * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[13] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[14] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + ((d[15] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[16] * -2.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[19] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[20] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + ((d[21] * 1.500000e+00f) * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[22] * 1.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[25] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[26] * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + (d[27] * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + d[28]);
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[2] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[3] * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[4]);
  data_pack_local[3] = (data_pack_local[3] + ((d[7] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[8] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.500000e+00f) * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[10] * -1.500000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[13] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[14] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[15] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[16] * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[19] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[20] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[21] * 1.500000e+00f) * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[22] * 1.500000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[25] * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[26] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[27] * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[28]);
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[1] * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[2] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + (d[3] * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + d[4]);
  data_pack_local[4] = (data_pack_local[4] + ((d[7] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + ((d[8] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[9] * -1.500000e+00f) * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.500000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[13] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + ((d[14] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[15] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[16] * -2.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[19] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + ((d[20] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[21] * 1.500000e+00f) * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[22] * 1.500000e+00f));
  data_pack_local[4] = (data_pack_local[4] + (d[25] * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[26] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + (d[27] * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + d[28]);
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + d[1]);
  data_pack_local[5] = (data_pack_local[5] + (d[2] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[3] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[4] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[5]);
  data_pack_local[5] = (data_pack_local[5] + (d[7] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[8] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[9] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[10] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[11] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[13] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[14] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[15] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[16] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[17] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[19] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[20] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[21] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[22] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[23] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[25]);
  data_pack_local[5] = (data_pack_local[5] + (d[26] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[27] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[28] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[29]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + d[6]);
  data_pack_local[6] = (data_pack_local[6] + (d[7] * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[8] * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[9] * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[6] = (data_pack_local[6] + (d[12] * -2.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[13] * -2.500000e+00f) * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[14] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[15] * -2.500000e+00f) * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[16] * -2.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[18] * 5.000000e-01f));
  data_pack_local[6] = (data_pack_local[6] + ((d[19] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[20] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[21] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[22] * 5.000000e-01f));
  data_pack_local[6] = (data_pack_local[6] + d[24]);
  data_pack_local[6] = (data_pack_local[6] + (d[25] * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[26] * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[27] * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[28]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + d[7]);
  data_pack_local[7] = (data_pack_local[7] + (d[8] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + d[10]);
  data_pack_local[7] = (data_pack_local[7] + (d[13] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + ((d[14] * -2.500000e+00f) * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + ((d[15] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + (d[16] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[19] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + ((d[20] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + ((d[21] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + (d[22] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + d[25]);
  data_pack_local[7] = (data_pack_local[7] + (d[26] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[27] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + d[28]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + (d[7] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[8] * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + (d[9] * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[10]);
  data_pack_local[8] = (data_pack_local[8] + ((d[13] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + ((d[14] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + ((d[15] * -2.500000e+00f) * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[16] * -2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + ((d[19] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + ((d[20] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + ((d[21] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[22] * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + (d[25] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[26] * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + (d[27] * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[28]);
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[7] * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[8] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[9] * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[9] = (data_pack_local[9] + ((d[13] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[14] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[15] * -2.500000e+00f) * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[16] * -2.500000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[19] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[20] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[21] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[22] * 5.000000e-01f));
  data_pack_local[9] = (data_pack_local[9] + (d[25] * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[26] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[27] * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[28]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + (d[7] * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[8] * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + (d[9] * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[10] = (data_pack_local[10] + ((d[13] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + ((d[14] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + ((d[15] * -2.500000e+00f) * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[16] * -2.500000e+00f));
  data_pack_local[10] = (data_pack_local[10] + ((d[19] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + ((d[20] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + ((d[21] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[22] * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[25] * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[26] * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + (d[27] * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + d[28]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[8] * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[10] * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[11] = (data_pack_local[11] + (d[13] * -2.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[14] * -2.500000e+00f) * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[15] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[16] * -2.500000e+00f) * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[17] * -2.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[19] * 5.000000e-01f));
  data_pack_local[11] = (data_pack_local[11] + ((d[20] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[21] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[22] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[23] * 5.000000e-01f));
  data_pack_local[11] = (data_pack_local[11] + d[25]);
  data_pack_local[11] = (data_pack_local[11] + (d[26] * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[27] * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[28] * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[29]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[6] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[7] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[8] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[9] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[10] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[12] * 5.000000e-01f));
  data_pack_local[12] = (data_pack_local[12] + ((d[13] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[14] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[15] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[16] * 5.000000e-01f));
  data_pack_local[12] = (data_pack_local[12] + (d[18] * 2.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[19] * 2.500000e+00f) * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[20] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[21] * 2.500000e+00f) * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[22] * 2.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[24]);
  data_pack_local[12] = (data_pack_local[12] + (d[25] * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[26] * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[27] * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[28]);
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + (d[7] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[8] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[9] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[10] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + ((d[14] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[15] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[16] * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[19] * 2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[20] * 2.500000e+00f) * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[21] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[22] * 2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[25]);
  data_pack_local[13] = (data_pack_local[13] + (d[26] * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[27] * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + d[28]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + ((d[7] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[8] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[9] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[10] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[13] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[14] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[15] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[16] * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[19] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[20] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[21] * 2.500000e+00f) * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[22] * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[25] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[26] * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + (d[27] * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[28]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[7] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[8] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[9] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[10] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[13] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[14] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[15] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[16] * 5.000000e-01f));
  data_pack_local[15] = (data_pack_local[15] + ((d[19] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[20] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[21] * 2.500000e+00f) * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[22] * 2.500000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[25] * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[26] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[27] * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[28]);
  data_pack_local[16] = 0.000000e+00f;
  data_pack_local[16] = (data_pack_local[16] + ((d[7] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[8] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[9] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[10] * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[13] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[14] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[15] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[16] * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[19] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[20] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[21] * 2.500000e+00f) * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[22] * 2.500000e+00f));
  data_pack_local[16] = (data_pack_local[16] + (d[25] * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[26] * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + (d[27] * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + d[28]);
  data_pack_local[17] = 0.000000e+00f;
  data_pack_local[17] = (data_pack_local[17] + (d[7] * -1.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[8] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[9] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[10] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[11] * -1.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[13] * 5.000000e-01f));
  data_pack_local[17] = (data_pack_local[17] + ((d[14] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[15] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[16] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[17] * 5.000000e-01f));
  data_pack_local[17] = (data_pack_local[17] + (d[19] * 2.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[20] * 2.500000e+00f) * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[21] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[22] * 2.500000e+00f) * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[23] * 2.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + d[25]);
  data_pack_local[17] = (data_pack_local[17] + (d[26] * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[27] * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[28] * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + d[29]);
  data_pack_local[18] = 0.000000e+00f;
  data_pack_local[18] = (data_pack_local[18] + (d[6] * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[7] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[8] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[9] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[10] * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[12] * -1.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[13] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[14] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[15] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[16] * -1.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[18] * 2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[19] * 2.000000e+00f) * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[20] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[21] * 2.000000e+00f) * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[22] * 2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + d[24]);
  data_pack_local[18] = (data_pack_local[18] + (d[25] * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[26] * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[27] * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + d[28]);
  data_pack_local[19] = 0.000000e+00f;
  data_pack_local[19] = (data_pack_local[19] + (d[7] * -2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[8] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[9] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + (d[10] * -2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + (d[13] * -1.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[14] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[15] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + (d[16] * -1.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + (d[19] * 2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[20] * 2.000000e+00f) * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[21] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + (d[22] * 2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + d[25]);
  data_pack_local[19] = (data_pack_local[19] + (d[26] * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + (d[27] * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + d[28]);
  data_pack_local[20] = 0.000000e+00f;
  data_pack_local[20] = (data_pack_local[20] + ((d[7] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[8] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + ((d[9] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[10] * -2.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[13] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[14] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + ((d[15] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[16] * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[19] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[20] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + ((d[21] * 2.000000e+00f) * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[22] * 2.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[25] * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[26] * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + (d[27] * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + d[28]);
  data_pack_local[21] = 0.000000e+00f;
  data_pack_local[21] = (data_pack_local[21] + ((d[7] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[8] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[9] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[10] * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[13] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[15] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[16] * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[19] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[20] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[21] * 2.000000e+00f) * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[22] * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[25] * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[26] * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[27] * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + d[28]);
  data_pack_local[22] = 0.000000e+00f;
  data_pack_local[22] = (data_pack_local[22] + ((d[7] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + ((d[8] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[9] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[10] * -2.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[13] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[15] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[16] * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[19] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + ((d[20] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[21] * 2.000000e+00f) * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[22] * 2.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + (d[25] * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[26] * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + (d[27] * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + d[28]);
  data_pack_local[23] = 0.000000e+00f;
  data_pack_local[23] = (data_pack_local[23] + (d[7] * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[8] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[9] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[10] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[11] * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[13] * -1.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[14] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[15] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[16] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[17] * -1.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[19] * 2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[20] * 2.000000e+00f) * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[21] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[22] * 2.000000e+00f) * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[23] * 2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + d[25]);
  data_pack_local[23] = (data_pack_local[23] + (d[26] * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[27] * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[28] * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + d[29]);
  data_pack_local[24] = 0.000000e+00f;
  data_pack_local[24] = (data_pack_local[24] + (d[6] * 5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + ((d[7] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[8] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[9] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[10] * 5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + (d[12] * -1.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[13] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[14] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[15] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[16] * -1.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[18] * -5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + ((d[19] * -5.000000e-01f) * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[20] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[21] * -5.000000e-01f) * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[22] * -5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + d[24]);
  data_pack_local[24] = (data_pack_local[24] + (d[25] * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[26] * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[27] * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + d[28]);
  data_pack_local[25] = 0.000000e+00f;
  data_pack_local[25] = (data_pack_local[25] + (d[7] * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + ((d[8] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[9] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[10] * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[13] * -1.000000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[14] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[15] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[16] * -1.000000e+00f));
  data_pack_local[25] = (data_pack_local[25] + (d[19] * -5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + ((d[20] * -5.000000e-01f) * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[21] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[22] * -5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + d[25]);
  data_pack_local[25] = (data_pack_local[25] + (d[26] * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + (d[27] * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + d[28]);
  data_pack_local[26] = 0.000000e+00f;
  data_pack_local[26] = (data_pack_local[26] + ((d[7] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[8] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[9] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[10] * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[13] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[14] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[15] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[16] * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[19] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[20] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[21] * -5.000000e-01f) * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[22] * -5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + (d[25] * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[26] * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + (d[27] * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + d[28]);
  data_pack_local[27] = 0.000000e+00f;
  data_pack_local[27] = (data_pack_local[27] + ((d[7] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[8] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[9] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[10] * 5.000000e-01f));
  data_pack_local[27] = (data_pack_local[27] + ((d[13] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[15] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[16] * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[19] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[20] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[21] * -5.000000e-01f) * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[22] * -5.000000e-01f));
  data_pack_local[27] = (data_pack_local[27] + (d[25] * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[26] * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[27] * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + d[28]);
  data_pack_local[28] = 0.000000e+00f;
  data_pack_local[28] = (data_pack_local[28] + ((d[7] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[8] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[9] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[10] * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[13] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[15] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[16] * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[19] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[20] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[21] * -5.000000e-01f) * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[22] * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[25] * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[26] * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + (d[27] * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + d[28]);
  data_pack_local[29] = 0.000000e+00f;
  data_pack_local[29] = (data_pack_local[29] + (d[7] * 5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + ((d[8] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[9] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[10] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[11] * 5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + (d[13] * -1.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[14] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[15] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[16] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[17] * -1.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[19] * -5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + ((d[20] * -5.000000e-01f) * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[21] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[22] * -5.000000e-01f) * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[23] * -5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + d[25]);
  data_pack_local[29] = (data_pack_local[29] + (d[26] * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[27] * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[28] * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + d[29]);
  data_pack_local[30] = 0.000000e+00f;
  data_pack_local[30] = (data_pack_local[30] + d[6]);
  data_pack_local[30] = (data_pack_local[30] + (d[7] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[8] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[9] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + d[10]);
  data_pack_local[30] = (data_pack_local[30] + (d[12] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[13] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[14] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[15] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[16] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[18] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[19] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[20] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[21] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[22] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[24] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[25] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[26] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[27] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[28] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + d[30]);
  data_pack_local[30] = (data_pack_local[30] + (d[31] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[32] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[33] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + d[34]);
  data_pack_local[31] = 0.000000e+00f;
  data_pack_local[31] = (data_pack_local[31] + d[7]);
  data_pack_local[31] = (data_pack_local[31] + (d[8] * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[9] * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + d[10]);
  data_pack_local[31] = (data_pack_local[31] + (d[13] * -1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[14] * -1.500000e+00f) * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[15] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + (d[16] * -1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[19] * -2.000000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[20] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[21] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + (d[22] * -2.000000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[25] * 1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[26] * 1.500000e+00f) * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[27] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + (d[28] * 1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + d[31]);
  data_pack_local[31] = (data_pack_local[31] + (d[32] * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[33] * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + d[34]);
  data_pack_local[32] = 0.000000e+00f;
  data_pack_local[32] = (data_pack_local[32] + (d[7] * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[8] * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + (d[9] * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + d[10]);
  data_pack_local[32] = (data_pack_local[32] + ((d[13] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[14] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + ((d[15] * -1.500000e+00f) * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[16] * -1.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[19] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[20] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + ((d[21] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[22] * -2.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[25] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[26] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + ((d[27] * 1.500000e+00f) * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[28] * 1.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[31] * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[32] * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + (d[33] * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + d[34]);
  data_pack_local[33] = 0.000000e+00f;
  data_pack_local[33] = (data_pack_local[33] + (d[7] * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[8] * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[9] * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + d[10]);
  data_pack_local[33] = (data_pack_local[33] + ((d[13] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[14] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[15] * -1.500000e+00f) * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[16] * -1.500000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[19] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[20] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[21] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[22] * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[25] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[26] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[27] * 1.500000e+00f) * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[28] * 1.500000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[31] * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[32] * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[33] * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + d[34]);
  data_pack_local[34] = 0.000000e+00f;
  data_pack_local[34] = (data_pack_local[34] + (d[7] * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[8] * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + (d[9] * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + d[10]);
  data_pack_local[34] = (data_pack_local[34] + ((d[13] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + ((d[14] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[15] * -1.500000e+00f) * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[16] * -1.500000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[19] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + ((d[20] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[21] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[22] * -2.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[25] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + ((d[26] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[27] * 1.500000e+00f) * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[28] * 1.500000e+00f));
  data_pack_local[34] = (data_pack_local[34] + (d[31] * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[32] * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + (d[33] * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + d[34]);
  data_pack_local[35] = 0.000000e+00f;
  data_pack_local[35] = (data_pack_local[35] + d[7]);
  data_pack_local[35] = (data_pack_local[35] + (d[8] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[9] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[10] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + d[11]);
  data_pack_local[35] = (data_pack_local[35] + (d[13] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[14] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[15] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[16] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[17] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[19] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[20] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[21] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[22] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[23] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[25] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[26] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[27] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[28] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[29] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + d[31]);
  data_pack_local[35] = (data_pack_local[35] + (d[32] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[33] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[34] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + d[35]);
  for (int eps_1 = 0; eps_1 < 6; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 6; ++nu_1) {
      data_pack[((((eps_1 * 75264) + (nu_1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 6) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_1_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[((eps * 4) + nu)] = (((((1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 2) + eps)) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 2) + eps) < 29)) && (1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2) + nu))) && ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2) + nu) < 29)) ? p0[((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 56) + (eps * 28)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2)) + nu) - 29)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[8] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[10] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + (d[1] * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[2]);
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.000000e+00f));
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + d[1]);
  data_pack_local[2] = (data_pack_local[2] + d[2]);
  data_pack_local[2] = (data_pack_local[2] + (d[9] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.000000e+00f));
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[3]);
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[11] * -1.000000e+00f));
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[4] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + d[8]);
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.000000e+00f));
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[6] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[9] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[10]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + (d[5] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[6] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[9]);
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[7] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + d[11]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + d[4]);
  data_pack_local[8] = (data_pack_local[8] + (d[6] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[8]);
  data_pack_local[8] = (data_pack_local[8] + (d[10] * -1.000000e+00f));
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[5] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[6]);
  data_pack_local[9] = (data_pack_local[9] + (d[9] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + d[5]);
  data_pack_local[10] = (data_pack_local[10] + d[6]);
  data_pack_local[10] = (data_pack_local[10] + d[9]);
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + (d[5] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[4] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[12]);
  data_pack_local[12] = (data_pack_local[12] + (d[14] * -1.000000e+00f));
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[6] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[14]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + (d[5] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[6] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[13]);
  data_pack_local[14] = (data_pack_local[14] + d[14]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[7] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[13] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[15]);
  for (int eps_1 = 0; eps_1 < 4; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 4; ++nu_1) {
      data_pack[((((eps_1 * 100352) + (nu_1 * 25088)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 4) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(49) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[16];
  __shared__ float p1_shared[128];
  __shared__ float data_pack_shared[392];
  #pragma unroll
  for (int co_c_init = 0; co_c_init < 8; ++co_c_init) {
    bgemm_local[co_c_init] = 0.000000e+00f;
    bgemm_local[(co_c_init + 8)] = 0.000000e+00f;
  }
  for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)) < 128) {
        p1_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.z) * 65536) + (ci_outer * 2048)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) + ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) & 15))];
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1) {
      data_pack_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 49) + ((int)threadIdx.x))] = data_pack[((((((int)blockIdx.z) * 12544) + (ci_outer * 392)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 49)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    #pragma unroll
    for (int ci_inner = 0; ci_inner < 8; ++ci_inner) {
      #pragma unroll
      for (int co_c = 0; co_c < 8; ++co_c) {
        bgemm_local[co_c] = (bgemm_local[co_c] + (p1_shared[((ci_inner * 16) + co_c)] * data_pack_shared[((ci_inner * 49) + ((int)threadIdx.x))]));
        bgemm_local[(co_c + 8)] = (bgemm_local[(co_c + 8)] + (p1_shared[(((ci_inner * 16) + co_c) + 8)] * data_pack_shared[((ci_inner * 49) + ((int)threadIdx.x))]));
      }
    }
  }
  #pragma unroll
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 8; ++co_inner_inner_inner) {
    bgemm[((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (co_inner_inner_inner * 49)) + ((int)threadIdx.x))] = bgemm_local[co_inner_inner_inner];
    bgemm[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (co_inner_inner_inner * 49)) + ((int)threadIdx.x)) + 392)] = bgemm_local[(co_inner_inner_inner + 8)];
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_add_nn_relu_2_kernel0(float* __restrict__ T_relu, float* __restrict__ p0, float* __restrict__ p1) {
  T_relu[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max((p0[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + p1[(((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) / 49)]), 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(112) tvmgen_default_fused_nn_conv2d_add_nn_relu_3_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ T_relu, float* __restrict__ p2) {
  float conv2d_nchw[1];
  __shared__ float pad_temp_shared[180];
  __shared__ float p1_shared[576];
  conv2d_nchw[0] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 15) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2))] = (((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 2) / 3)) % 15) / 5))) && (1 <= (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) % 15))) ? p0[((((((rc_outer * 784) + ((((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 2) / 3)) / 15) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 2) / 3)) % 15) / 5) * 14)) + (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) % 15)) - 15)] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 15) {
      if (((int)threadIdx.x) < 6) {
        pad_temp_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1)] = (((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) % 15) / 5))) && (1 <= ((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1) % 15))) ? p0[((((((rc_outer * 784) + ((((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) / 15) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) % 15) / 5) * 14)) + ((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1) % 15)) - 15)] : 0.000000e+00f);
      }
    }
    if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 16) {
      if (((int)threadIdx.x) < 6) {
        p1_shared[((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6))] = p1[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6))];
      }
    }
    if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 16) {
      if (((int)threadIdx.x) < 6) {
        p1_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 1)] = p1[(((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 1)];
      }
    }
    if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 16) {
      if (((int)threadIdx.x) < 6) {
        p1_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 2)] = p1[(((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 2)];
      }
    }
    if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 16) {
      if (((int)threadIdx.x) < 6) {
        p1_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 3)] = p1[(((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 3)];
      }
    }
    if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 16) {
      if (((int)threadIdx.x) < 6) {
        p1_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 4)] = p1[(((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 4)];
      }
    }
    if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 16) {
      if (((int)threadIdx.x) < 6) {
        p1_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 5)] = p1[(((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 5)];
      }
    }
    __syncthreads();
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) * 2)] * p1_shared[(((int)threadIdx.z) * 36)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 1)] * p1_shared[((((int)threadIdx.z) * 36) + 1)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 2)] * p1_shared[((((int)threadIdx.z) * 36) + 2)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 15)] * p1_shared[((((int)threadIdx.z) * 36) + 3)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 16)] * p1_shared[((((int)threadIdx.z) * 36) + 4)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 17)] * p1_shared[((((int)threadIdx.z) * 36) + 5)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 30)] * p1_shared[((((int)threadIdx.z) * 36) + 6)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 31)] * p1_shared[((((int)threadIdx.z) * 36) + 7)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 32)] * p1_shared[((((int)threadIdx.z) * 36) + 8)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 45)] * p1_shared[((((int)threadIdx.z) * 36) + 9)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 46)] * p1_shared[((((int)threadIdx.z) * 36) + 10)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 47)] * p1_shared[((((int)threadIdx.z) * 36) + 11)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 60)] * p1_shared[((((int)threadIdx.z) * 36) + 12)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 61)] * p1_shared[((((int)threadIdx.z) * 36) + 13)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 62)] * p1_shared[((((int)threadIdx.z) * 36) + 14)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 75)] * p1_shared[((((int)threadIdx.z) * 36) + 15)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 76)] * p1_shared[((((int)threadIdx.z) * 36) + 16)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 77)] * p1_shared[((((int)threadIdx.z) * 36) + 17)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 90)] * p1_shared[((((int)threadIdx.z) * 36) + 18)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 91)] * p1_shared[((((int)threadIdx.z) * 36) + 19)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 92)] * p1_shared[((((int)threadIdx.z) * 36) + 20)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 105)] * p1_shared[((((int)threadIdx.z) * 36) + 21)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 106)] * p1_shared[((((int)threadIdx.z) * 36) + 22)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 107)] * p1_shared[((((int)threadIdx.z) * 36) + 23)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 120)] * p1_shared[((((int)threadIdx.z) * 36) + 24)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 121)] * p1_shared[((((int)threadIdx.z) * 36) + 25)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 122)] * p1_shared[((((int)threadIdx.z) * 36) + 26)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 135)] * p1_shared[((((int)threadIdx.z) * 36) + 27)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 136)] * p1_shared[((((int)threadIdx.z) * 36) + 28)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 137)] * p1_shared[((((int)threadIdx.z) * 36) + 29)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 150)] * p1_shared[((((int)threadIdx.z) * 36) + 30)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 151)] * p1_shared[((((int)threadIdx.z) * 36) + 31)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 152)] * p1_shared[((((int)threadIdx.z) * 36) + 32)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 165)] * p1_shared[((((int)threadIdx.z) * 36) + 33)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 166)] * p1_shared[((((int)threadIdx.z) * 36) + 34)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) * 2) + 167)] * p1_shared[((((int)threadIdx.z) * 36) + 35)]));
  }
  T_relu[((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x))] = max((conv2d_nchw[0] + p2[((((int)blockIdx.z) * 16) + ((int)threadIdx.z))]), 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_nn_global_avg_pool2d_kernel0(float* __restrict__ p0, float* __restrict__ tensor) {
  float tensor_rf[1];
  float red_buf0[1];
  tensor_rf[0] = 0.000000e+00f;
  for (int rv0_rv1_fused_outer = 0; rv0_rv1_fused_outer < 2; ++rv0_rv1_fused_outer) {
    if ((((rv0_rv1_fused_outer * 32) + ((int)threadIdx.x)) < 49) && (((rv0_rv1_fused_outer * 32) + ((int)threadIdx.x)) < 49)) {
      tensor_rf[0] = (tensor_rf[0] + p0[((((((int)blockIdx.x) * 1568) + (((int)threadIdx.y) * 49)) + (rv0_rv1_fused_outer * 32)) + ((int)threadIdx.x))]);
    }
  }
  uint mask[1];
  float t0[1];
  red_buf0[0] = tensor_rf[0];
  mask[0] = (__activemask() & ((uint)(0 << (((int)threadIdx.y) * 32))));
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  if (((int)threadIdx.x) == 0) {
    tensor[((((int)blockIdx.x) * 32) + ((int)threadIdx.y))] = red_buf0[0];
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_multiply_add_nn_re_a4e88f85cf7823fc__kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[8];
  __shared__ float p1_shared[1024];
  __shared__ float data_pack_shared[256];
  for (int co_c_init = 0; co_c_init < 4; ++co_c_init) {
    for (int p_c_init = 0; p_c_init < 2; ++p_c_init) {
      bgemm_local[((co_c_init * 2) + p_c_init)] = 0.000000e+00f;
    }
  }
  for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      p1_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = p1[(((((((((int)blockIdx.z) * 262144) + (ci_outer * 8192)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 1024)) + ((((int)threadIdx.y) >> 3) * 512)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.y) & 7) * 8)) + ((int)threadIdx.x))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1) {
      data_pack_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = data_pack[(((((((int)blockIdx.z) * 8192) + (ci_outer * 256)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 16; ++ci_inner) {
      for (int co_c = 0; co_c < 4; ++co_c) {
        for (int p_c = 0; p_c < 2; ++p_c) {
          bgemm_local[((co_c * 2) + p_c)] = (bgemm_local[((co_c * 2) + p_c)] + (p1_shared[(((ci_inner * 64) + (((int)threadIdx.y) * 4)) + co_c)] * data_pack_shared[(((ci_inner * 16) + (((int)threadIdx.x) * 2)) + p_c)]));
        }
      }
    }
  }
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 4; ++co_inner_inner_inner) {
    for (int p_inner_inner_inner = 0; p_inner_inner_inner < 2; ++p_inner_inner_inner) {
      bgemm[((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.y) * 64)) + (co_inner_inner_inner * 16)) + (((int)threadIdx.x) * 2)) + p_inner_inner_inner)] = bgemm_local[((co_inner_inner_inner * 2) + p_inner_inner_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(224) tvmgen_default_fused_nn_conv2d_2_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[2];
  __shared__ float pad_temp_shared[216];
  __shared__ float p1_shared[256];
  for (int ff_c_init = 0; ff_c_init < 2; ++ff_c_init) {
    conv2d_nchw_local[ff_c_init] = 0.000000e+00f;
  }
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 7) + (((int)threadIdx.x) >> 1)) < 108) {
      pad_temp_shared[((((int)threadIdx.z) * 14) + ((int)threadIdx.x))] = p0[((((rc_outer * 6272) + ((((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) / 27) * 784)) + (((int)blockIdx.y) * 56)) + (((((int)threadIdx.z) * 14) + ((int)threadIdx.x)) % 27))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((int)threadIdx.x) >> 3) + ((int)threadIdx.z)) < 16) {
        if (((int)threadIdx.x) < 8) {
          p1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = p1[((((((((int)blockIdx.z) * 4096) + (((int)threadIdx.z) * 256)) + ((((int)threadIdx.x) >> 2) * 128)) + (rc_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)];
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
      for (int ff_c = 0; ff_c < 2; ++ff_c) {
        conv2d_nchw_local[ff_c] = (conv2d_nchw_local[ff_c] + (pad_temp_shared[((rc_inner * 27) + (((int)threadIdx.x) * 2))] * p1_shared[(((((int)threadIdx.z) * 16) + (ff_c * 8)) + rc_inner)]));
      }
    }
  }
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 2; ++ff_inner_inner_inner) {
    conv2d_nchw[(((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (ff_inner_inner_inner * 196)) + (((int)blockIdx.y) * 14)) + ((int)threadIdx.x))] = conv2d_nchw_local[ff_inner_inner_inner];
  }
}

extern "C" __global__ void __launch_bounds__(196) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_1_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[16];
  __shared__ float p1_shared[512];
  __shared__ float data_pack_shared[1568];
  bgemm_local[0] = 0.000000e+00f;
  bgemm_local[4] = 0.000000e+00f;
  bgemm_local[8] = 0.000000e+00f;
  bgemm_local[12] = 0.000000e+00f;
  bgemm_local[2] = 0.000000e+00f;
  bgemm_local[6] = 0.000000e+00f;
  bgemm_local[10] = 0.000000e+00f;
  bgemm_local[14] = 0.000000e+00f;
  bgemm_local[1] = 0.000000e+00f;
  bgemm_local[5] = 0.000000e+00f;
  bgemm_local[9] = 0.000000e+00f;
  bgemm_local[13] = 0.000000e+00f;
  bgemm_local[3] = 0.000000e+00f;
  bgemm_local[7] = 0.000000e+00f;
  bgemm_local[11] = 0.000000e+00f;
  bgemm_local[15] = 0.000000e+00f;
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    p1_shared[((((int)threadIdx.y) * 49) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.y) * 17) + ((int)threadIdx.x)) & 31))];
    p1_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196)] = p1[(((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 17) + ((int)threadIdx.x)) + 4) & 31))];
    if (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) < 120) {
      if (((int)threadIdx.y) < 3) {
        p1_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392)] = p1[(((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 17) + ((int)threadIdx.x)) + 8) & 31))];
      }
    }
    data_pack_shared[((((int)threadIdx.y) * 49) + ((int)threadIdx.x))] = data_pack[((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x))];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 392)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 784)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 588)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 1176)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 784)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 1568)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 980)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 1960)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1176)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 2352)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1372)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 2744)];
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 16; ++ci_inner) {
      bgemm_local[0] = (bgemm_local[0] + (p1_shared[((ci_inner * 32) + (((int)threadIdx.y) * 2))] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[4] = (bgemm_local[4] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 8)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[8] = (bgemm_local[8] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 16)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[12] = (bgemm_local[12] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 24)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[2] = (bgemm_local[2] + (p1_shared[((ci_inner * 32) + (((int)threadIdx.y) * 2))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[6] = (bgemm_local[6] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 8)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[10] = (bgemm_local[10] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 16)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[14] = (bgemm_local[14] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 24)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[1] = (bgemm_local[1] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 1)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[5] = (bgemm_local[5] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 9)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[9] = (bgemm_local[9] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 17)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[13] = (bgemm_local[13] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 25)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[3] = (bgemm_local[3] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 1)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[7] = (bgemm_local[7] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 9)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[11] = (bgemm_local[11] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 17)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[15] = (bgemm_local[15] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 25)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
    }
  }
  bgemm[(((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x))] = bgemm_local[0];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1568)] = bgemm_local[4];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3136)] = bgemm_local[8];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4704)] = bgemm_local[12];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 49)] = bgemm_local[2];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1617)] = bgemm_local[6];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3185)] = bgemm_local[10];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4753)] = bgemm_local[14];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 196)] = bgemm_local[1];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1764)] = bgemm_local[5];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3332)] = bgemm_local[9];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4900)] = bgemm_local[13];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 245)] = bgemm_local[3];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1813)] = bgemm_local[7];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3381)] = bgemm_local[11];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4949)] = bgemm_local[15];
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ p2) {
  float inverse[4];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 24576)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112)]);
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 98304)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688)]);
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 122880)]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      if (((((((int)threadIdx.x) & 15) >> 2) * 2) + ax2_inner) < 7) {
        if ((((((int)threadIdx.x) & 3) * 2) + ax3_inner) < 7) {
          T_relu[((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (ax2_inner * 7)) + ((((int)threadIdx.x) & 3) * 2)) + ax3_inner)] = max((inverse[((ax2_inner * 2) + ax3_inner)] + p2[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4))]), 0.000000e+00f);
        }
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(224) tvmgen_default_fused_nn_conv2d_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[896];
  __shared__ float p1_shared[1024];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      pad_temp_shared[(((((int)threadIdx.z) * 112) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = p0[((((((rc_outer * 50176) + (((int)threadIdx.z) * 6272)) + ((((int)threadIdx.x) / 14) * 3136)) + (((int)blockIdx.y) * 56)) + ((((int)threadIdx.x) % 14) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) {
      if (((((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) >> 7) + ((int)threadIdx.z)) < 8) {
        if (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) < 128) {
          p1_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 5)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1)] = p1[((((((int)threadIdx.z) * 512) + ((((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) >> 4) * 64)) + (rc_outer * 16)) + (((((int)threadIdx.x) * 5) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) & 15))];
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * p1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 128)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 256)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 384)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 512)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 640)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 768)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((rc_inner * 56) + ((int)threadIdx.x))] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 896)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)) + 28)] * p1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)) + 28)] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 128)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)) + 28)] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 256)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)) + 28)] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 384)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)) + 28)] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 512)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)) + 28)] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 640)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)) + 28)] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 768)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_inner * 56) + ((int)threadIdx.x)) + 28)] * p1_shared[(((((int)threadIdx.z) * 16) + rc_inner) + 896)]));
    }
  }
  conv2d_nchw[(((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x))] = conv2d_nchw_local[0];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25088)] = conv2d_nchw_local[2];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50176)] = conv2d_nchw_local[4];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 75264)] = conv2d_nchw_local[6];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 100352)] = conv2d_nchw_local[8];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 125440)] = conv2d_nchw_local[10];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 150528)] = conv2d_nchw_local[12];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 175616)] = conv2d_nchw_local[14];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28)] = conv2d_nchw_local[1];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 25116)] = conv2d_nchw_local[3];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 50204)] = conv2d_nchw_local[5];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 75292)] = conv2d_nchw_local[7];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 100380)] = conv2d_nchw_local[9];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 125468)] = conv2d_nchw_local[11];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 150556)] = conv2d_nchw_local[13];
  conv2d_nchw[((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 175644)] = conv2d_nchw_local[15];
}

extern "C" __global__ void __launch_bounds__(32) tvmgen_default_fused_nn_softmax_kernel0(float* __restrict__ p0, float* __restrict__ T_softmax_norm) {
  float normal_reduce_temp0[1];
  float red_buf0[1];
  float T_softmax_exp[32];
  float normal_reduce_temp0_1[1];
  float red_buf0_1[1];
  normal_reduce_temp0[0] = -3.402823e+38f;
  for (int k_inner = 0; k_inner < 32; ++k_inner) {
    if (((((int)threadIdx.x) * 4) + (k_inner >> 3)) < 125) {
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], p0[((((int)threadIdx.x) * 32) + k_inner)]);
    }
  }
  uint mask[1];
  float t0[1];
  red_buf0[0] = normal_reduce_temp0[0];
  mask[0] = __activemask();
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], 0, 32);
  for (int i1_inner_outer = 0; i1_inner_outer < 8; ++i1_inner_outer) {
    for (int i1_inner_inner_s = 0; i1_inner_inner_s < 4; ++i1_inner_inner_s) {
      if (((((int)threadIdx.x) * 4) + (i1_inner_outer >> 1)) < 125) {
        T_softmax_exp[((i1_inner_outer * 4) + i1_inner_inner_s)] = __expf((p0[(((((int)threadIdx.x) * 32) + (i1_inner_outer * 4)) + i1_inner_inner_s)] - red_buf0[0]));
      }
    }
  }
  normal_reduce_temp0_1[0] = 0.000000e+00f;
  for (int k_inner_1 = 0; k_inner_1 < 32; ++k_inner_1) {
    if (((((int)threadIdx.x) * 4) + (k_inner_1 >> 3)) < 125) {
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + T_softmax_exp[k_inner_1]);
    }
  }
  uint mask_1[1];
  float t0_1[1];
  red_buf0_1[0] = normal_reduce_temp0_1[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  red_buf0_1[0] = __shfl_sync(mask_1[0], red_buf0_1[0], 0, 32);
  for (int i1_inner_outer_1 = 0; i1_inner_outer_1 < 8; ++i1_inner_outer_1) {
    for (int i1_inner_inner_s_1 = 0; i1_inner_inner_s_1 < 4; ++i1_inner_inner_s_1) {
      if (((((int)threadIdx.x) * 4) + (i1_inner_outer_1 >> 1)) < 125) {
        T_softmax_norm[(((((int)threadIdx.x) * 32) + (i1_inner_outer_1 * 4)) + i1_inner_inner_s_1)] = (T_softmax_exp[((i1_inner_outer_1 * 4) + i1_inner_inner_s_1)] / red_buf0_1[0]);
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[((eps * 4) + nu)] = (((((1 <= ((((((int)threadIdx.x) & 15) >> 2) * 2) + eps)) && ((((((int)threadIdx.x) & 15) >> 2) + (eps >> 1)) < 4)) && (1 <= (((((int)threadIdx.x) & 3) * 2) + nu))) && (((nu >> 1) + (((int)threadIdx.x) & 3)) < 4)) ? p0[(((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (eps * 7)) + ((((int)threadIdx.x) & 3) * 2)) + nu) - 8)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[8] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[10] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + (d[1] * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[2]);
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.000000e+00f));
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + d[1]);
  data_pack_local[2] = (data_pack_local[2] + d[2]);
  data_pack_local[2] = (data_pack_local[2] + (d[9] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.000000e+00f));
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[3]);
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[11] * -1.000000e+00f));
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[4] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + d[8]);
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.000000e+00f));
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[6] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[9] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[10]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + (d[5] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[6] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[9]);
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[7] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + d[11]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + d[4]);
  data_pack_local[8] = (data_pack_local[8] + (d[6] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[8]);
  data_pack_local[8] = (data_pack_local[8] + (d[10] * -1.000000e+00f));
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[5] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[6]);
  data_pack_local[9] = (data_pack_local[9] + (d[9] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + d[5]);
  data_pack_local[10] = (data_pack_local[10] + d[6]);
  data_pack_local[10] = (data_pack_local[10] + d[9]);
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + (d[5] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[4] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[12]);
  data_pack_local[12] = (data_pack_local[12] + (d[14] * -1.000000e+00f));
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[6] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[14]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + (d[5] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[6] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[13]);
  data_pack_local[14] = (data_pack_local[14] + d[14]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[7] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[13] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[15]);
  for (int eps_1 = 0; eps_1 < 4; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 4; ++nu_1) {
      data_pack[((((eps_1 * 32768) + (nu_1 * 8192)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 4) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(49) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_2_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[16];
  __shared__ float p1_shared[128];
  __shared__ float data_pack_shared[392];
  #pragma unroll
  for (int co_c_init = 0; co_c_init < 8; ++co_c_init) {
    bgemm_local[co_c_init] = 0.000000e+00f;
    bgemm_local[(co_c_init + 8)] = 0.000000e+00f;
  }
  for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)) < 128) {
        p1_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.z) * 65536) + (ci_outer * 2048)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + ((int)threadIdx.x)) >> 4) * 256)) + (((int)blockIdx.y) * 16)) + ((((int)threadIdx.x) + ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) & 15))];
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1) {
      data_pack_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 49) + ((int)threadIdx.x))] = data_pack[((((((int)blockIdx.z) * 12544) + (ci_outer * 392)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 49)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    #pragma unroll
    for (int ci_inner = 0; ci_inner < 8; ++ci_inner) {
      #pragma unroll
      for (int co_c = 0; co_c < 8; ++co_c) {
        bgemm_local[co_c] = (bgemm_local[co_c] + (p1_shared[((ci_inner * 16) + co_c)] * data_pack_shared[((ci_inner * 49) + ((int)threadIdx.x))]));
        bgemm_local[(co_c + 8)] = (bgemm_local[(co_c + 8)] + (p1_shared[(((ci_inner * 16) + co_c) + 8)] * data_pack_shared[((ci_inner * 49) + ((int)threadIdx.x))]));
      }
    }
  }
  #pragma unroll
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 8; ++co_inner_inner_inner) {
    bgemm[((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (co_inner_inner_inner * 49)) + ((int)threadIdx.x))] = bgemm_local[co_inner_inner_inner];
    bgemm[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 784)) + (co_inner_inner_inner * 49)) + ((int)threadIdx.x)) + 392)] = bgemm_local[(co_inner_inner_inner + 8)];
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_3_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[((eps * 4) + nu)] = (((((1 <= ((((((int)threadIdx.x) & 15) >> 2) * 2) + eps)) && ((((((int)threadIdx.x) & 15) >> 2) + (eps >> 1)) < 4)) && (1 <= (((((int)threadIdx.x) & 3) * 2) + nu))) && (((nu >> 1) + (((int)threadIdx.x) & 3)) < 4)) ? p0[(((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (eps * 7)) + ((((int)threadIdx.x) & 3) * 2)) + nu) - 8)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[8] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[10] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + (d[1] * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[2]);
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.000000e+00f));
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + d[1]);
  data_pack_local[2] = (data_pack_local[2] + d[2]);
  data_pack_local[2] = (data_pack_local[2] + (d[9] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.000000e+00f));
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[3]);
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[11] * -1.000000e+00f));
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[4] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + d[8]);
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.000000e+00f));
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[6] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[9] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[10]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + (d[5] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[6] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[9]);
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[7] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + d[11]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + d[4]);
  data_pack_local[8] = (data_pack_local[8] + (d[6] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[8]);
  data_pack_local[8] = (data_pack_local[8] + (d[10] * -1.000000e+00f));
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[5] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[6]);
  data_pack_local[9] = (data_pack_local[9] + (d[9] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + d[5]);
  data_pack_local[10] = (data_pack_local[10] + d[6]);
  data_pack_local[10] = (data_pack_local[10] + d[9]);
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + (d[5] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[4] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[12]);
  data_pack_local[12] = (data_pack_local[12] + (d[14] * -1.000000e+00f));
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[6] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[14]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + (d[5] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[6] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[13]);
  data_pack_local[14] = (data_pack_local[14] + d[14]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[7] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[13] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[15]);
  for (int eps_1 = 0; eps_1 < 4; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 4; ++nu_1) {
      data_pack[((((eps_1 * 32768) + (nu_1 * 8192)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 4) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(196) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[32];
  __shared__ float p1_shared[256];
  __shared__ float data_pack_shared[1568];
  for (int co_c_init = 0; co_c_init < 8; ++co_c_init) {
    bgemm_local[co_c_init] = 0.000000e+00f;
    bgemm_local[(co_c_init + 16)] = 0.000000e+00f;
    bgemm_local[(co_c_init + 8)] = 0.000000e+00f;
    bgemm_local[(co_c_init + 24)] = 0.000000e+00f;
  }
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + (((((int)threadIdx.y) * 49) + (((int)threadIdx.x) >> 1)) >> 1)) < 64) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 2) + ((int)threadIdx.y)) < 3) {
          p1_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.z) * 4096) + (ci_outer * 512)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + (((((int)threadIdx.y) * 49) + (((int)threadIdx.x) >> 1)) >> 1)) >> 3) * 64)) + (((int)blockIdx.y) * 32)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 4) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) & 31))];
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1) {
      data_pack_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x))] = data_pack[(((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 196)) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 8; ++ci_inner) {
      for (int co_c = 0; co_c < 8; ++co_c) {
        bgemm_local[co_c] = (bgemm_local[co_c] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c)] * data_pack_shared[((ci_inner * 196) + ((int)threadIdx.x))]));
        bgemm_local[(co_c + 16)] = (bgemm_local[(co_c + 16)] + (p1_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c) + 16)] * data_pack_shared[((ci_inner * 196) + ((int)threadIdx.x))]));
        bgemm_local[(co_c + 8)] = (bgemm_local[(co_c + 8)] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c)] * data_pack_shared[(((ci_inner * 196) + ((int)threadIdx.x)) + 98)]));
        bgemm_local[(co_c + 24)] = (bgemm_local[(co_c + 24)] + (p1_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c) + 16)] * data_pack_shared[(((ci_inner * 196) + ((int)threadIdx.x)) + 98)]));
      }
    }
  }
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 8; ++co_inner_inner_inner) {
    bgemm[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x))] = bgemm_local[co_inner_inner_inner];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 3136)] = bgemm_local[(co_inner_inner_inner + 16)];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 98)] = bgemm_local[(co_inner_inner_inner + 8)];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 3234)] = bgemm_local[(co_inner_inner_inner + 24)];
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[36];
  float data_pack_local[36];
  for (int eps = 0; eps < 6; ++eps) {
    for (int nu = 0; nu < 6; ++nu) {
      d[((eps * 6) + nu)] = (((((1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 4) + eps)) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 4) + eps) < 57)) && (1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4) + nu))) && ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4) + nu) < 57)) ? p0[((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 224) + (eps * 56)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4)) + nu) - 57)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[1] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[3] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + d[4]);
  data_pack_local[0] = (data_pack_local[0] + (d[6] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[7] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[8] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[9] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[10] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[12] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[13] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[14] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[15] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[16] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[18] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[19] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[20] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[21] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[22] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + d[24]);
  data_pack_local[0] = (data_pack_local[0] + (d[25] * -1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[26] * -2.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[27] * 1.500000e+00f));
  data_pack_local[0] = (data_pack_local[0] + d[28]);
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + d[1]);
  data_pack_local[1] = (data_pack_local[1] + (d[2] * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[3] * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + d[4]);
  data_pack_local[1] = (data_pack_local[1] + (d[7] * -1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[8] * -1.500000e+00f) * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[13] * -2.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[14] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[15] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + (d[16] * -2.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[19] * 1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[20] * 1.500000e+00f) * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + ((d[21] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + (d[22] * 1.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[25]);
  data_pack_local[1] = (data_pack_local[1] + (d[26] * -2.500000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[27] * 5.000000e-01f));
  data_pack_local[1] = (data_pack_local[1] + d[28]);
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + (d[1] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[2] * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + (d[3] * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + d[4]);
  data_pack_local[2] = (data_pack_local[2] + ((d[7] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[8] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + ((d[9] * -1.500000e+00f) * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[13] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[14] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + ((d[15] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[16] * -2.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[19] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + ((d[20] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + ((d[21] * 1.500000e+00f) * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[22] * 1.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[25] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[26] * 5.000000e-01f));
  data_pack_local[2] = (data_pack_local[2] + (d[27] * 2.500000e+00f));
  data_pack_local[2] = (data_pack_local[2] + d[28]);
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[2] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[3] * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[4]);
  data_pack_local[3] = (data_pack_local[3] + ((d[7] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[8] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.500000e+00f) * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[10] * -1.500000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[13] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[14] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[15] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[16] * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[19] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[20] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + ((d[21] * 1.500000e+00f) * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[22] * 1.500000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[25] * -2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[26] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[27] * 2.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[28]);
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[1] * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[2] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + (d[3] * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + d[4]);
  data_pack_local[4] = (data_pack_local[4] + ((d[7] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + ((d[8] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[9] * -1.500000e+00f) * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.500000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[13] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + ((d[14] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[15] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[16] * -2.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[19] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + ((d[20] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[21] * 1.500000e+00f) * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[22] * 1.500000e+00f));
  data_pack_local[4] = (data_pack_local[4] + (d[25] * 5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + (d[26] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + (d[27] * -5.000000e-01f));
  data_pack_local[4] = (data_pack_local[4] + d[28]);
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + d[1]);
  data_pack_local[5] = (data_pack_local[5] + (d[2] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[3] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[4] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[5]);
  data_pack_local[5] = (data_pack_local[5] + (d[7] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[8] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[9] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[10] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[11] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[13] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[14] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[15] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[16] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[17] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[19] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[20] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[21] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + ((d[22] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[23] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[25]);
  data_pack_local[5] = (data_pack_local[5] + (d[26] * -1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[27] * -2.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[28] * 1.500000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[29]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + d[6]);
  data_pack_local[6] = (data_pack_local[6] + (d[7] * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[8] * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[9] * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[6] = (data_pack_local[6] + (d[12] * -2.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[13] * -2.500000e+00f) * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[14] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[15] * -2.500000e+00f) * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[16] * -2.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[18] * 5.000000e-01f));
  data_pack_local[6] = (data_pack_local[6] + ((d[19] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[20] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + ((d[21] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[22] * 5.000000e-01f));
  data_pack_local[6] = (data_pack_local[6] + d[24]);
  data_pack_local[6] = (data_pack_local[6] + (d[25] * -1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[26] * -2.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[27] * 1.500000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[28]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + d[7]);
  data_pack_local[7] = (data_pack_local[7] + (d[8] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + d[10]);
  data_pack_local[7] = (data_pack_local[7] + (d[13] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + ((d[14] * -2.500000e+00f) * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + ((d[15] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + (d[16] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[19] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + ((d[20] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + ((d[21] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + (d[22] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + d[25]);
  data_pack_local[7] = (data_pack_local[7] + (d[26] * -2.500000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[27] * 5.000000e-01f));
  data_pack_local[7] = (data_pack_local[7] + d[28]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + (d[7] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[8] * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + (d[9] * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[10]);
  data_pack_local[8] = (data_pack_local[8] + ((d[13] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + ((d[14] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + ((d[15] * -2.500000e+00f) * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[16] * -2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + ((d[19] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + ((d[20] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + ((d[21] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[22] * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + (d[25] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + (d[26] * 5.000000e-01f));
  data_pack_local[8] = (data_pack_local[8] + (d[27] * 2.500000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[28]);
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[7] * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[8] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[9] * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[9] = (data_pack_local[9] + ((d[13] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[14] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[15] * -2.500000e+00f) * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[16] * -2.500000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[19] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[20] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + ((d[21] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[22] * 5.000000e-01f));
  data_pack_local[9] = (data_pack_local[9] + (d[25] * -2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[26] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + (d[27] * 2.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[28]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + (d[7] * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[8] * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + (d[9] * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[10] = (data_pack_local[10] + ((d[13] * -2.500000e+00f) * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + ((d[14] * -2.500000e+00f) * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + ((d[15] * -2.500000e+00f) * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[16] * -2.500000e+00f));
  data_pack_local[10] = (data_pack_local[10] + ((d[19] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + ((d[20] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + ((d[21] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[22] * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[25] * 5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + (d[26] * -1.000000e+00f));
  data_pack_local[10] = (data_pack_local[10] + (d[27] * -5.000000e-01f));
  data_pack_local[10] = (data_pack_local[10] + d[28]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[8] * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[10] * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[11] = (data_pack_local[11] + (d[13] * -2.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[14] * -2.500000e+00f) * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[15] * -2.500000e+00f) * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[16] * -2.500000e+00f) * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[17] * -2.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[19] * 5.000000e-01f));
  data_pack_local[11] = (data_pack_local[11] + ((d[20] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[21] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + ((d[22] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[23] * 5.000000e-01f));
  data_pack_local[11] = (data_pack_local[11] + d[25]);
  data_pack_local[11] = (data_pack_local[11] + (d[26] * -1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[27] * -2.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + (d[28] * 1.500000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[29]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[6] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[7] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[8] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[9] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[10] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[12] * 5.000000e-01f));
  data_pack_local[12] = (data_pack_local[12] + ((d[13] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[14] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[15] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[16] * 5.000000e-01f));
  data_pack_local[12] = (data_pack_local[12] + (d[18] * 2.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[19] * 2.500000e+00f) * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[20] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[21] * 2.500000e+00f) * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[22] * 2.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[24]);
  data_pack_local[12] = (data_pack_local[12] + (d[25] * -1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[26] * -2.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + (d[27] * 1.500000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[28]);
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + (d[7] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[8] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[9] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[10] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + ((d[14] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[15] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[16] * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[19] * 2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[20] * 2.500000e+00f) * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + ((d[21] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + (d[22] * 2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[25]);
  data_pack_local[13] = (data_pack_local[13] + (d[26] * -2.500000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[27] * 5.000000e-01f));
  data_pack_local[13] = (data_pack_local[13] + d[28]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + ((d[7] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[8] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[9] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[10] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[13] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[14] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[15] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[16] * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[19] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + ((d[20] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + ((d[21] * 2.500000e+00f) * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[22] * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[25] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[26] * 5.000000e-01f));
  data_pack_local[14] = (data_pack_local[14] + (d[27] * 2.500000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[28]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[7] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[8] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[9] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[10] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[13] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[14] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[15] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[16] * 5.000000e-01f));
  data_pack_local[15] = (data_pack_local[15] + ((d[19] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[20] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + ((d[21] * 2.500000e+00f) * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[22] * 2.500000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[25] * -2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[26] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[27] * 2.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[28]);
  data_pack_local[16] = 0.000000e+00f;
  data_pack_local[16] = (data_pack_local[16] + ((d[7] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[8] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[9] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[10] * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[13] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[14] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[15] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[16] * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[19] * 2.500000e+00f) * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + ((d[20] * 2.500000e+00f) * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + ((d[21] * 2.500000e+00f) * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[22] * 2.500000e+00f));
  data_pack_local[16] = (data_pack_local[16] + (d[25] * 5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + (d[26] * -1.000000e+00f));
  data_pack_local[16] = (data_pack_local[16] + (d[27] * -5.000000e-01f));
  data_pack_local[16] = (data_pack_local[16] + d[28]);
  data_pack_local[17] = 0.000000e+00f;
  data_pack_local[17] = (data_pack_local[17] + (d[7] * -1.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[8] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[9] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[10] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[11] * -1.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[13] * 5.000000e-01f));
  data_pack_local[17] = (data_pack_local[17] + ((d[14] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[15] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[16] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[17] * 5.000000e-01f));
  data_pack_local[17] = (data_pack_local[17] + (d[19] * 2.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[20] * 2.500000e+00f) * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[21] * 2.500000e+00f) * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + ((d[22] * 2.500000e+00f) * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[23] * 2.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + d[25]);
  data_pack_local[17] = (data_pack_local[17] + (d[26] * -1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[27] * -2.000000e+00f));
  data_pack_local[17] = (data_pack_local[17] + (d[28] * 1.500000e+00f));
  data_pack_local[17] = (data_pack_local[17] + d[29]);
  data_pack_local[18] = 0.000000e+00f;
  data_pack_local[18] = (data_pack_local[18] + (d[6] * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[7] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[8] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[9] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[10] * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[12] * -1.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[13] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[14] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[15] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[16] * -1.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[18] * 2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[19] * 2.000000e+00f) * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[20] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + ((d[21] * 2.000000e+00f) * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[22] * 2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + d[24]);
  data_pack_local[18] = (data_pack_local[18] + (d[25] * -1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[26] * -2.000000e+00f));
  data_pack_local[18] = (data_pack_local[18] + (d[27] * 1.500000e+00f));
  data_pack_local[18] = (data_pack_local[18] + d[28]);
  data_pack_local[19] = 0.000000e+00f;
  data_pack_local[19] = (data_pack_local[19] + (d[7] * -2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[8] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[9] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + (d[10] * -2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + (d[13] * -1.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[14] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[15] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + (d[16] * -1.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + (d[19] * 2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[20] * 2.000000e+00f) * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + ((d[21] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + (d[22] * 2.000000e+00f));
  data_pack_local[19] = (data_pack_local[19] + d[25]);
  data_pack_local[19] = (data_pack_local[19] + (d[26] * -2.500000e+00f));
  data_pack_local[19] = (data_pack_local[19] + (d[27] * 5.000000e-01f));
  data_pack_local[19] = (data_pack_local[19] + d[28]);
  data_pack_local[20] = 0.000000e+00f;
  data_pack_local[20] = (data_pack_local[20] + ((d[7] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[8] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + ((d[9] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[10] * -2.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[13] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[14] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + ((d[15] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[16] * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[19] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + ((d[20] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + ((d[21] * 2.000000e+00f) * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[22] * 2.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[25] * -1.000000e+00f));
  data_pack_local[20] = (data_pack_local[20] + (d[26] * 5.000000e-01f));
  data_pack_local[20] = (data_pack_local[20] + (d[27] * 2.500000e+00f));
  data_pack_local[20] = (data_pack_local[20] + d[28]);
  data_pack_local[21] = 0.000000e+00f;
  data_pack_local[21] = (data_pack_local[21] + ((d[7] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[8] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[9] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[10] * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[13] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[15] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[16] * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[19] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[20] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + ((d[21] * 2.000000e+00f) * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[22] * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[25] * -2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[26] * -1.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + (d[27] * 2.000000e+00f));
  data_pack_local[21] = (data_pack_local[21] + d[28]);
  data_pack_local[22] = 0.000000e+00f;
  data_pack_local[22] = (data_pack_local[22] + ((d[7] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + ((d[8] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[9] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[10] * -2.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[13] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[15] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[16] * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[19] * 2.000000e+00f) * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + ((d[20] * 2.000000e+00f) * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + ((d[21] * 2.000000e+00f) * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[22] * 2.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + (d[25] * 5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + (d[26] * -1.000000e+00f));
  data_pack_local[22] = (data_pack_local[22] + (d[27] * -5.000000e-01f));
  data_pack_local[22] = (data_pack_local[22] + d[28]);
  data_pack_local[23] = 0.000000e+00f;
  data_pack_local[23] = (data_pack_local[23] + (d[7] * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[8] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[9] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[10] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[11] * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[13] * -1.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[14] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[15] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[16] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[17] * -1.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[19] * 2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[20] * 2.000000e+00f) * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[21] * 2.000000e+00f) * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + ((d[22] * 2.000000e+00f) * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[23] * 2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + d[25]);
  data_pack_local[23] = (data_pack_local[23] + (d[26] * -1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[27] * -2.000000e+00f));
  data_pack_local[23] = (data_pack_local[23] + (d[28] * 1.500000e+00f));
  data_pack_local[23] = (data_pack_local[23] + d[29]);
  data_pack_local[24] = 0.000000e+00f;
  data_pack_local[24] = (data_pack_local[24] + (d[6] * 5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + ((d[7] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[8] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[9] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[10] * 5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + (d[12] * -1.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[13] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[14] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[15] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[16] * -1.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[18] * -5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + ((d[19] * -5.000000e-01f) * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[20] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + ((d[21] * -5.000000e-01f) * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[22] * -5.000000e-01f));
  data_pack_local[24] = (data_pack_local[24] + d[24]);
  data_pack_local[24] = (data_pack_local[24] + (d[25] * -1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[26] * -2.000000e+00f));
  data_pack_local[24] = (data_pack_local[24] + (d[27] * 1.500000e+00f));
  data_pack_local[24] = (data_pack_local[24] + d[28]);
  data_pack_local[25] = 0.000000e+00f;
  data_pack_local[25] = (data_pack_local[25] + (d[7] * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + ((d[8] * 5.000000e-01f) * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[9] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[10] * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[13] * -1.000000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[14] * -1.000000e+00f) * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[15] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[16] * -1.000000e+00f));
  data_pack_local[25] = (data_pack_local[25] + (d[19] * -5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + ((d[20] * -5.000000e-01f) * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + ((d[21] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + (d[22] * -5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + d[25]);
  data_pack_local[25] = (data_pack_local[25] + (d[26] * -2.500000e+00f));
  data_pack_local[25] = (data_pack_local[25] + (d[27] * 5.000000e-01f));
  data_pack_local[25] = (data_pack_local[25] + d[28]);
  data_pack_local[26] = 0.000000e+00f;
  data_pack_local[26] = (data_pack_local[26] + ((d[7] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[8] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[9] * 5.000000e-01f) * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[10] * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[13] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[14] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[15] * -1.000000e+00f) * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[16] * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[19] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + ((d[20] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + ((d[21] * -5.000000e-01f) * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[22] * -5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + (d[25] * -1.000000e+00f));
  data_pack_local[26] = (data_pack_local[26] + (d[26] * 5.000000e-01f));
  data_pack_local[26] = (data_pack_local[26] + (d[27] * 2.500000e+00f));
  data_pack_local[26] = (data_pack_local[26] + d[28]);
  data_pack_local[27] = 0.000000e+00f;
  data_pack_local[27] = (data_pack_local[27] + ((d[7] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[8] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[9] * 5.000000e-01f) * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[10] * 5.000000e-01f));
  data_pack_local[27] = (data_pack_local[27] + ((d[13] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[15] * -1.000000e+00f) * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[16] * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[19] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[20] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + ((d[21] * -5.000000e-01f) * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[22] * -5.000000e-01f));
  data_pack_local[27] = (data_pack_local[27] + (d[25] * -2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[26] * -1.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + (d[27] * 2.000000e+00f));
  data_pack_local[27] = (data_pack_local[27] + d[28]);
  data_pack_local[28] = 0.000000e+00f;
  data_pack_local[28] = (data_pack_local[28] + ((d[7] * 5.000000e-01f) * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[8] * 5.000000e-01f) * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[9] * 5.000000e-01f) * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[10] * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[13] * -1.000000e+00f) * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[14] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[15] * -1.000000e+00f) * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[16] * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[19] * -5.000000e-01f) * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + ((d[20] * -5.000000e-01f) * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + ((d[21] * -5.000000e-01f) * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[22] * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[25] * 5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + (d[26] * -1.000000e+00f));
  data_pack_local[28] = (data_pack_local[28] + (d[27] * -5.000000e-01f));
  data_pack_local[28] = (data_pack_local[28] + d[28]);
  data_pack_local[29] = 0.000000e+00f;
  data_pack_local[29] = (data_pack_local[29] + (d[7] * 5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + ((d[8] * 5.000000e-01f) * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[9] * 5.000000e-01f) * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[10] * 5.000000e-01f) * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[11] * 5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + (d[13] * -1.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[14] * -1.000000e+00f) * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[15] * -1.000000e+00f) * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[16] * -1.000000e+00f) * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[17] * -1.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[19] * -5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + ((d[20] * -5.000000e-01f) * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[21] * -5.000000e-01f) * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + ((d[22] * -5.000000e-01f) * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[23] * -5.000000e-01f));
  data_pack_local[29] = (data_pack_local[29] + d[25]);
  data_pack_local[29] = (data_pack_local[29] + (d[26] * -1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[27] * -2.000000e+00f));
  data_pack_local[29] = (data_pack_local[29] + (d[28] * 1.500000e+00f));
  data_pack_local[29] = (data_pack_local[29] + d[29]);
  data_pack_local[30] = 0.000000e+00f;
  data_pack_local[30] = (data_pack_local[30] + d[6]);
  data_pack_local[30] = (data_pack_local[30] + (d[7] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[8] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[9] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + d[10]);
  data_pack_local[30] = (data_pack_local[30] + (d[12] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[13] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[14] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[15] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[16] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[18] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[19] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[20] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[21] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[22] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[24] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[25] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[26] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + ((d[27] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[28] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + d[30]);
  data_pack_local[30] = (data_pack_local[30] + (d[31] * -1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[32] * -2.000000e+00f));
  data_pack_local[30] = (data_pack_local[30] + (d[33] * 1.500000e+00f));
  data_pack_local[30] = (data_pack_local[30] + d[34]);
  data_pack_local[31] = 0.000000e+00f;
  data_pack_local[31] = (data_pack_local[31] + d[7]);
  data_pack_local[31] = (data_pack_local[31] + (d[8] * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[9] * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + d[10]);
  data_pack_local[31] = (data_pack_local[31] + (d[13] * -1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[14] * -1.500000e+00f) * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[15] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + (d[16] * -1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[19] * -2.000000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[20] * -2.000000e+00f) * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[21] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + (d[22] * -2.000000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[25] * 1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[26] * 1.500000e+00f) * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + ((d[27] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + (d[28] * 1.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + d[31]);
  data_pack_local[31] = (data_pack_local[31] + (d[32] * -2.500000e+00f));
  data_pack_local[31] = (data_pack_local[31] + (d[33] * 5.000000e-01f));
  data_pack_local[31] = (data_pack_local[31] + d[34]);
  data_pack_local[32] = 0.000000e+00f;
  data_pack_local[32] = (data_pack_local[32] + (d[7] * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[8] * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + (d[9] * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + d[10]);
  data_pack_local[32] = (data_pack_local[32] + ((d[13] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[14] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + ((d[15] * -1.500000e+00f) * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[16] * -1.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[19] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[20] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + ((d[21] * -2.000000e+00f) * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[22] * -2.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[25] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + ((d[26] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + ((d[27] * 1.500000e+00f) * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[28] * 1.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[31] * -1.000000e+00f));
  data_pack_local[32] = (data_pack_local[32] + (d[32] * 5.000000e-01f));
  data_pack_local[32] = (data_pack_local[32] + (d[33] * 2.500000e+00f));
  data_pack_local[32] = (data_pack_local[32] + d[34]);
  data_pack_local[33] = 0.000000e+00f;
  data_pack_local[33] = (data_pack_local[33] + (d[7] * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[8] * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[9] * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + d[10]);
  data_pack_local[33] = (data_pack_local[33] + ((d[13] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[14] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[15] * -1.500000e+00f) * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[16] * -1.500000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[19] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[20] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[21] * -2.000000e+00f) * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[22] * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[25] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[26] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + ((d[27] * 1.500000e+00f) * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[28] * 1.500000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[31] * -2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[32] * -1.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + (d[33] * 2.000000e+00f));
  data_pack_local[33] = (data_pack_local[33] + d[34]);
  data_pack_local[34] = 0.000000e+00f;
  data_pack_local[34] = (data_pack_local[34] + (d[7] * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[8] * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + (d[9] * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + d[10]);
  data_pack_local[34] = (data_pack_local[34] + ((d[13] * -1.500000e+00f) * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + ((d[14] * -1.500000e+00f) * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[15] * -1.500000e+00f) * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[16] * -1.500000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[19] * -2.000000e+00f) * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + ((d[20] * -2.000000e+00f) * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[21] * -2.000000e+00f) * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[22] * -2.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[25] * 1.500000e+00f) * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + ((d[26] * 1.500000e+00f) * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + ((d[27] * 1.500000e+00f) * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[28] * 1.500000e+00f));
  data_pack_local[34] = (data_pack_local[34] + (d[31] * 5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + (d[32] * -1.000000e+00f));
  data_pack_local[34] = (data_pack_local[34] + (d[33] * -5.000000e-01f));
  data_pack_local[34] = (data_pack_local[34] + d[34]);
  data_pack_local[35] = 0.000000e+00f;
  data_pack_local[35] = (data_pack_local[35] + d[7]);
  data_pack_local[35] = (data_pack_local[35] + (d[8] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[9] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[10] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + d[11]);
  data_pack_local[35] = (data_pack_local[35] + (d[13] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[14] * -1.500000e+00f) * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[15] * -1.500000e+00f) * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[16] * -1.500000e+00f) * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[17] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[19] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[20] * -2.000000e+00f) * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[21] * -2.000000e+00f) * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[22] * -2.000000e+00f) * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[23] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[25] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[26] * 1.500000e+00f) * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[27] * 1.500000e+00f) * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + ((d[28] * 1.500000e+00f) * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[29] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + d[31]);
  data_pack_local[35] = (data_pack_local[35] + (d[32] * -1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[33] * -2.000000e+00f));
  data_pack_local[35] = (data_pack_local[35] + (d[34] * 1.500000e+00f));
  data_pack_local[35] = (data_pack_local[35] + d[35]);
  for (int eps_1 = 0; eps_1 < 6; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 6; ++nu_1) {
      data_pack[((((eps_1 * 75264) + (nu_1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 6) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[((eps * 4) + nu)] = (((((1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 2) + eps)) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 2) + eps) < 29)) && (1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2) + nu))) && ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2) + nu) < 29)) ? p0[((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 56) + (eps * 28)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2)) + nu) - 29)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[8] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[10] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + (d[1] * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[2]);
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.000000e+00f));
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + d[1]);
  data_pack_local[2] = (data_pack_local[2] + d[2]);
  data_pack_local[2] = (data_pack_local[2] + (d[9] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.000000e+00f));
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[3]);
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[11] * -1.000000e+00f));
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[4] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + d[8]);
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.000000e+00f));
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[6] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[9] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[10]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + (d[5] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[6] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[9]);
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[7] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + d[11]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + d[4]);
  data_pack_local[8] = (data_pack_local[8] + (d[6] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[8]);
  data_pack_local[8] = (data_pack_local[8] + (d[10] * -1.000000e+00f));
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[5] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[6]);
  data_pack_local[9] = (data_pack_local[9] + (d[9] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + d[5]);
  data_pack_local[10] = (data_pack_local[10] + d[6]);
  data_pack_local[10] = (data_pack_local[10] + d[9]);
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + (d[5] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[4] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[12]);
  data_pack_local[12] = (data_pack_local[12] + (d[14] * -1.000000e+00f));
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[6] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[14]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + (d[5] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[6] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[13]);
  data_pack_local[14] = (data_pack_local[14] + d[14]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[7] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[13] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[15]);
  for (int eps_1 = 0; eps_1 < 4; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 4; ++nu_1) {
      data_pack[((((eps_1 * 100352) + (nu_1 * 25088)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 4) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_2_kernel2(float* __restrict__ bgemm, float* __restrict__ T_add, float* __restrict__ p2) {
  float inverse[4];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7))]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 12544)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 25088)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 50176)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 100352)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 12544)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 25088)]);
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 37632)]);
  inverse[1] = (inverse[1] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)]);
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 87808)]);
  inverse[1] = (inverse[1] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 137984)]);
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 50176)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 100352)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 150528)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 163072)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 175616)]);
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + ((bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 87808)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 137984)]);
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 163072)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 175616)]);
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 188160)]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_add[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 7) * 28) + (ax2_inner * 14)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2)) + ax3_inner)] = (inverse[((ax2_inner * 2) + ax3_inner)] + p2[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 7) * 28) + (ax2_inner * 14)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2)) + ax3_inner)]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_add_nn_relu_3_kernel0(float* __restrict__ T_relu, float* __restrict__ p0, float* __restrict__ p1) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) < 49) {
    T_relu[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max((p0[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + p1[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 49)]), 0.000000e+00f);
  }
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_add_nn_relu_1_kernel0(float* __restrict__ T_relu, float* __restrict__ p0, float* __restrict__ p1) {
  T_relu[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max((p0[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + p1[(((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 4)) / 49)]), 0.000000e+00f);
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_1_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[((eps * 4) + nu)] = (((((1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 2) + eps)) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) % 98) / 7) * 2) + eps) < 29)) && (1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2) + nu))) && ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2) + nu) < 29)) ? p0[((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 56) + (eps * 28)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2)) + nu) - 29)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[8] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[10] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + (d[1] * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[2]);
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.000000e+00f));
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + d[1]);
  data_pack_local[2] = (data_pack_local[2] + d[2]);
  data_pack_local[2] = (data_pack_local[2] + (d[9] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.000000e+00f));
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[3]);
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[11] * -1.000000e+00f));
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[4] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + d[8]);
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.000000e+00f));
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[6] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[9] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[10]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + (d[5] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[6] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[9]);
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[7] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + d[11]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + d[4]);
  data_pack_local[8] = (data_pack_local[8] + (d[6] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[8]);
  data_pack_local[8] = (data_pack_local[8] + (d[10] * -1.000000e+00f));
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[5] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[6]);
  data_pack_local[9] = (data_pack_local[9] + (d[9] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + d[5]);
  data_pack_local[10] = (data_pack_local[10] + d[6]);
  data_pack_local[10] = (data_pack_local[10] + d[9]);
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + (d[5] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[4] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[12]);
  data_pack_local[12] = (data_pack_local[12] + (d[14] * -1.000000e+00f));
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[6] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[14]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + (d[5] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[6] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[13]);
  data_pack_local[14] = (data_pack_local[14] + d[14]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[7] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[13] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[15]);
  for (int eps_1 = 0; eps_1 < 4; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 4; ++nu_1) {
      data_pack[((((eps_1 * 100352) + (nu_1 * 25088)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 4) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_1_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ p2, float* __restrict__ p3) {
  float inverse[4];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)]);
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)]);
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320)]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_relu[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner)] = max(((inverse[((ax2_inner * 2) + ax3_inner)] + p2[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner)]) + p3[(((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2)) / 49)]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(196) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_1_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[16];
  __shared__ float p1_shared[512];
  __shared__ float data_pack_shared[1568];
  bgemm_local[0] = 0.000000e+00f;
  bgemm_local[4] = 0.000000e+00f;
  bgemm_local[8] = 0.000000e+00f;
  bgemm_local[12] = 0.000000e+00f;
  bgemm_local[2] = 0.000000e+00f;
  bgemm_local[6] = 0.000000e+00f;
  bgemm_local[10] = 0.000000e+00f;
  bgemm_local[14] = 0.000000e+00f;
  bgemm_local[1] = 0.000000e+00f;
  bgemm_local[5] = 0.000000e+00f;
  bgemm_local[9] = 0.000000e+00f;
  bgemm_local[13] = 0.000000e+00f;
  bgemm_local[3] = 0.000000e+00f;
  bgemm_local[7] = 0.000000e+00f;
  bgemm_local[11] = 0.000000e+00f;
  bgemm_local[15] = 0.000000e+00f;
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    p1_shared[((((int)threadIdx.y) * 49) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.y) * 17) + ((int)threadIdx.x)) & 31))];
    p1_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196)] = p1[(((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 17) + ((int)threadIdx.x)) + 4) & 31))];
    if (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) < 120) {
      if (((int)threadIdx.y) < 3) {
        p1_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392)] = p1[(((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 17) + ((int)threadIdx.x)) + 8) & 31))];
      }
    }
    data_pack_shared[((((int)threadIdx.y) * 49) + ((int)threadIdx.x))] = data_pack[((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x))];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 392)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 784)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 588)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 1176)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 784)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 1568)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 980)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 1960)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1176)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 2352)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1372)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 2744)];
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 16; ++ci_inner) {
      bgemm_local[0] = (bgemm_local[0] + (p1_shared[((ci_inner * 32) + (((int)threadIdx.y) * 2))] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[4] = (bgemm_local[4] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 8)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[8] = (bgemm_local[8] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 16)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[12] = (bgemm_local[12] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 24)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[2] = (bgemm_local[2] + (p1_shared[((ci_inner * 32) + (((int)threadIdx.y) * 2))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[6] = (bgemm_local[6] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 8)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[10] = (bgemm_local[10] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 16)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[14] = (bgemm_local[14] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 24)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[1] = (bgemm_local[1] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 1)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[5] = (bgemm_local[5] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 9)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[9] = (bgemm_local[9] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 17)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[13] = (bgemm_local[13] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 25)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[3] = (bgemm_local[3] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 1)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[7] = (bgemm_local[7] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 9)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[11] = (bgemm_local[11] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 17)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[15] = (bgemm_local[15] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 25)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
    }
  }
  bgemm[(((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x))] = bgemm_local[0];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1568)] = bgemm_local[4];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3136)] = bgemm_local[8];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4704)] = bgemm_local[12];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 49)] = bgemm_local[2];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1617)] = bgemm_local[6];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3185)] = bgemm_local[10];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4753)] = bgemm_local[14];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 196)] = bgemm_local[1];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1764)] = bgemm_local[5];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3332)] = bgemm_local[9];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4900)] = bgemm_local[13];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 245)] = bgemm_local[3];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1813)] = bgemm_local[7];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3381)] = bgemm_local[11];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4949)] = bgemm_local[15];
}

extern "C" __global__ void __launch_bounds__(196) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[16];
  __shared__ float p1_shared[512];
  __shared__ float data_pack_shared[1568];
  bgemm_local[0] = 0.000000e+00f;
  bgemm_local[4] = 0.000000e+00f;
  bgemm_local[8] = 0.000000e+00f;
  bgemm_local[12] = 0.000000e+00f;
  bgemm_local[2] = 0.000000e+00f;
  bgemm_local[6] = 0.000000e+00f;
  bgemm_local[10] = 0.000000e+00f;
  bgemm_local[14] = 0.000000e+00f;
  bgemm_local[1] = 0.000000e+00f;
  bgemm_local[5] = 0.000000e+00f;
  bgemm_local[9] = 0.000000e+00f;
  bgemm_local[13] = 0.000000e+00f;
  bgemm_local[3] = 0.000000e+00f;
  bgemm_local[7] = 0.000000e+00f;
  bgemm_local[11] = 0.000000e+00f;
  bgemm_local[15] = 0.000000e+00f;
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    p1_shared[((((int)threadIdx.y) * 49) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + ((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.y) * 17) + ((int)threadIdx.x)) & 31))];
    p1_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196)] = p1[(((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 17) + ((int)threadIdx.x)) + 4) & 31))];
    if (((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) < 120) {
      if (((int)threadIdx.y) < 3) {
        p1_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392)] = p1[(((((((int)blockIdx.z) * 16384) + (ci_outer * 2048)) + (((((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392) >> 5) * 128)) + (((int)blockIdx.y) * 32)) + ((((((int)threadIdx.y) * 17) + ((int)threadIdx.x)) + 8) & 31))];
      }
    }
    data_pack_shared[((((int)threadIdx.y) * 49) + ((int)threadIdx.x))] = data_pack[((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x))];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 196)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 392)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 392)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 784)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 588)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 1176)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 784)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 1568)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 980)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 1960)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1176)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 2352)];
    data_pack_shared[(((((int)threadIdx.y) * 49) + ((int)threadIdx.x)) + 1372)] = data_pack[(((((((((int)blockIdx.z) * 25088) + (ci_outer * 3136)) + ((((int)threadIdx.y) >> 1) * 196)) + (((int)blockIdx.x) * 98)) + ((((int)threadIdx.y) & 1) * 49)) + ((int)threadIdx.x)) + 2744)];
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 16; ++ci_inner) {
      bgemm_local[0] = (bgemm_local[0] + (p1_shared[((ci_inner * 32) + (((int)threadIdx.y) * 2))] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[4] = (bgemm_local[4] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 8)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[8] = (bgemm_local[8] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 16)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[12] = (bgemm_local[12] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 24)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[2] = (bgemm_local[2] + (p1_shared[((ci_inner * 32) + (((int)threadIdx.y) * 2))] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[6] = (bgemm_local[6] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 8)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[10] = (bgemm_local[10] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 16)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[14] = (bgemm_local[14] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 24)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[1] = (bgemm_local[1] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 1)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[5] = (bgemm_local[5] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 9)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[9] = (bgemm_local[9] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 17)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[13] = (bgemm_local[13] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 25)] * data_pack_shared[((ci_inner * 98) + ((int)threadIdx.x))]));
      bgemm_local[3] = (bgemm_local[3] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 1)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[7] = (bgemm_local[7] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 9)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[11] = (bgemm_local[11] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 17)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
      bgemm_local[15] = (bgemm_local[15] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 2)) + 25)] * data_pack_shared[(((ci_inner * 98) + ((int)threadIdx.x)) + 49)]));
    }
  }
  bgemm[(((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x))] = bgemm_local[0];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1568)] = bgemm_local[4];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3136)] = bgemm_local[8];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4704)] = bgemm_local[12];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 49)] = bgemm_local[2];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1617)] = bgemm_local[6];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3185)] = bgemm_local[10];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4753)] = bgemm_local[14];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 196)] = bgemm_local[1];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1764)] = bgemm_local[5];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3332)] = bgemm_local[9];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4900)] = bgemm_local[13];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 245)] = bgemm_local[3];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 1813)] = bgemm_local[7];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 3381)] = bgemm_local[11];
  bgemm[((((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 392)) + (((int)blockIdx.x) * 98)) + ((int)threadIdx.x)) + 4949)] = bgemm_local[15];
}

extern "C" __global__ void __launch_bounds__(224) tvmgen_default_fused_nn_conv2d_3_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[1];
  __shared__ float pad_temp_shared[208];
  __shared__ float p1_shared[512];
  conv2d_nchw_local[0] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 208) {
      pad_temp_shared[((((int)threadIdx.z) * 7) + ((int)threadIdx.x))] = p0[((((rc_outer * 3136) + ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 13) * 196)) + (((int)blockIdx.y) * 28)) + (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) % 13))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) >> 4) + ((int)threadIdx.z)) < 32) {
        if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 16) {
          p1_shared[(((((int)threadIdx.z) * 16) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = p1[(((((((int)blockIdx.z) * 8192) + (((int)threadIdx.z) * 256)) + (rc_outer * 16)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)];
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_inner * 13) + (((int)threadIdx.x) * 2))] * p1_shared[((((int)threadIdx.z) * 16) + rc_inner)]));
    }
  }
  conv2d_nchw[((((((int)blockIdx.z) * 1568) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x))] = conv2d_nchw_local[0];
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ p2, float* __restrict__ p3) {
  float inverse[16];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * -2.000000e+00f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 5.000000e-01f));
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f));
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 4.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 2.500000e-01f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f));
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 12544)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 37632)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 62720)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 1.250000e-01f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)]);
  inverse[4] = 0.000000e+00f;
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[4] = (inverse[4] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f));
  inverse[4] = (inverse[4] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f));
  inverse[5] = 0.000000e+00f;
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -2.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[5] = (inverse[5] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f) * -1.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f) * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f) * -2.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f) * -1.000000e+00f));
  inverse[5] = (inverse[5] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f) * 5.000000e-01f));
  inverse[5] = (inverse[5] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f) * -2.000000e+00f));
  inverse[6] = 0.000000e+00f;
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 2.500000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * 4.000000e+00f));
  inverse[6] = (inverse[6] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[6] = (inverse[6] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f) * 2.500000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f) * 4.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f));
  inverse[6] = (inverse[6] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f) * 2.500000e-01f));
  inverse[6] = (inverse[6] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f) * 4.000000e+00f));
  inverse[7] = 0.000000e+00f;
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 1.250000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -8.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)] * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[7] = (inverse[7] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[7] = (inverse[7] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 5.000000e-01f) * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 5.000000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 5.000000e-01f) * 1.250000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 5.000000e-01f) * -8.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)] * 5.000000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -2.000000e+00f) * -1.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -2.000000e+00f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -2.000000e+00f) * 1.250000e-01f));
  inverse[7] = (inverse[7] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -2.000000e+00f) * -8.000000e+00f));
  inverse[7] = (inverse[7] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)] * -2.000000e+00f));
  inverse[8] = 0.000000e+00f;
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[8] = (inverse[8] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f));
  inverse[8] = (inverse[8] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f));
  inverse[9] = 0.000000e+00f;
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[9] = (inverse[9] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 5.000000e-01f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -2.000000e+00f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[9] = (inverse[9] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f) * -1.000000e+00f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f) * 5.000000e-01f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f) * -2.000000e+00f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f) * -1.000000e+00f));
  inverse[9] = (inverse[9] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f) * 5.000000e-01f));
  inverse[9] = (inverse[9] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f) * -2.000000e+00f));
  inverse[10] = 0.000000e+00f;
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)]);
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[10] = (inverse[10] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f) * 2.500000e-01f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f) * 4.000000e+00f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f) * 2.500000e-01f));
  inverse[10] = (inverse[10] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f) * 4.000000e+00f));
  inverse[11] = 0.000000e+00f;
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * 1.250000e-01f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -8.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)]);
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[11] = (inverse[11] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 2.500000e-01f) * -1.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 2.500000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 2.500000e-01f) * 1.250000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 2.500000e-01f) * -8.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)] * 2.500000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * 4.000000e+00f) * -1.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * 4.000000e+00f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * 4.000000e+00f) * 1.250000e-01f));
  inverse[11] = (inverse[11] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * 4.000000e+00f) * -8.000000e+00f));
  inverse[11] = (inverse[11] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)] * 4.000000e+00f));
  inverse[12] = 0.000000e+00f;
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f));
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)]);
  inverse[12] = (inverse[12] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)]);
  inverse[13] = 0.000000e+00f;
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 5.000000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -2.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[13] = (inverse[13] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 5.000000e-01f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -2.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f) * -1.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f) * 5.000000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f) * -2.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f) * -1.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f) * 5.000000e-01f));
  inverse[13] = (inverse[13] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f) * -2.000000e+00f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)] * -1.000000e+00f));
  inverse[13] = (inverse[13] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)] * 5.000000e-01f));
  inverse[13] = (inverse[13] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)] * -2.000000e+00f));
  inverse[14] = 0.000000e+00f;
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 2.500000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * 4.000000e+00f));
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)]);
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 2.500000e-01f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * 4.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f) * 2.500000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f) * 4.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f) * 2.500000e-01f));
  inverse[14] = (inverse[14] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f) * 4.000000e+00f));
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)]);
  inverse[14] = (inverse[14] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)] * 2.500000e-01f));
  inverse[14] = (inverse[14] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)] * 4.000000e+00f));
  inverse[15] = 0.000000e+00f;
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 87808)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 112896)] * -1.000000e+00f) * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 137984)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 163072)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 188160)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 213248)]);
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 238336)] * 1.250000e-01f) * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 263424)] * 1.250000e-01f) * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)] * 1.250000e-01f) * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 288512)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 313600)] * -8.000000e+00f) * -1.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 338688)] * -8.000000e+00f) * 1.250000e-01f));
  inverse[15] = (inverse[15] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)] * -8.000000e+00f) * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 363776)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 388864)] * -1.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 401408)]);
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 413952)] * 1.250000e-01f));
  inverse[15] = (inverse[15] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 426496)] * -8.000000e+00f));
  inverse[15] = (inverse[15] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 439040)]);
  for (int ax2_inner = 0; ax2_inner < 4; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 4; ++ax3_inner) {
      T_relu[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 224) + (ax2_inner * 56)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4)) + ax3_inner)] = max(((inverse[((ax2_inner * 4) + ax3_inner)] + p2[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 224) + (ax2_inner * 56)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 4)) + ax3_inner)]) + p3[(((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2)) / 49)]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[8];
  __shared__ float p1_shared[1024];
  __shared__ float data_pack_shared[256];
  for (int co_c_init = 0; co_c_init < 4; ++co_c_init) {
    for (int p_c_init = 0; p_c_init < 2; ++p_c_init) {
      bgemm_local[((co_c_init * 2) + p_c_init)] = 0.000000e+00f;
    }
  }
  for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      p1_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = p1[(((((((((int)blockIdx.z) * 262144) + (ci_outer * 8192)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 1024)) + ((((int)threadIdx.y) >> 3) * 512)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.y) & 7) * 8)) + ((int)threadIdx.x))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1) {
      data_pack_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = data_pack[(((((((int)blockIdx.z) * 8192) + (ci_outer * 256)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 16; ++ci_inner) {
      for (int co_c = 0; co_c < 4; ++co_c) {
        for (int p_c = 0; p_c < 2; ++p_c) {
          bgemm_local[((co_c * 2) + p_c)] = (bgemm_local[((co_c * 2) + p_c)] + (p1_shared[(((ci_inner * 64) + (((int)threadIdx.y) * 4)) + co_c)] * data_pack_shared[(((ci_inner * 16) + (((int)threadIdx.x) * 2)) + p_c)]));
        }
      }
    }
  }
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 4; ++co_inner_inner_inner) {
    for (int p_inner_inner_inner = 0; p_inner_inner_inner < 2; ++p_inner_inner_inner) {
      bgemm[((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.y) * 64)) + (co_inner_inner_inner * 16)) + (((int)threadIdx.x) * 2)) + p_inner_inner_inner)] = bgemm_local[((co_inner_inner_inner * 2) + p_inner_inner_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ p2) {
  float inverse[4];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7))]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 12544)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 25088)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 50176)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 100352)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)]);
  inverse[0] = (inverse[0] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 12544)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 25088)]);
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 37632)]);
  inverse[1] = (inverse[1] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)]);
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 87808)]);
  inverse[1] = (inverse[1] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[1] = (inverse[1] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 137984)]);
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 50176)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 100352)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 150528)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 163072)]);
  inverse[2] = (inverse[2] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 175616)]);
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + ((bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 62720)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 75264)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 87808)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 112896)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 125440)]);
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 137984)]);
  inverse[3] = (inverse[3] + (bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 163072)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 175616)]);
  inverse[3] = (inverse[3] + bgemm[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 49) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 7)) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7)) + 188160)]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_relu[(((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 7) * 28) + (ax2_inner * 14)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2)) + ax3_inner)] = max((inverse[((ax2_inner * 2) + ax3_inner)] + p2[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49)]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_3_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[8];
  __shared__ float p1_shared[1024];
  __shared__ float data_pack_shared[256];
  for (int co_c_init = 0; co_c_init < 4; ++co_c_init) {
    for (int p_c_init = 0; p_c_init < 2; ++p_c_init) {
      bgemm_local[((co_c_init * 2) + p_c_init)] = 0.000000e+00f;
    }
  }
  for (int ci_outer = 0; ci_outer < 32; ++ci_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      p1_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = p1[(((((((((int)blockIdx.z) * 262144) + (ci_outer * 8192)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 1024)) + ((((int)threadIdx.y) >> 3) * 512)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.y) & 7) * 8)) + ((int)threadIdx.x))];
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1) {
      data_pack_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 128) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))] = data_pack[(((((((int)blockIdx.z) * 8192) + (ci_outer * 256)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 128)) + (((int)threadIdx.y) * 8)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 16; ++ci_inner) {
      for (int co_c = 0; co_c < 4; ++co_c) {
        for (int p_c = 0; p_c < 2; ++p_c) {
          bgemm_local[((co_c * 2) + p_c)] = (bgemm_local[((co_c * 2) + p_c)] + (p1_shared[(((ci_inner * 64) + (((int)threadIdx.y) * 4)) + co_c)] * data_pack_shared[(((ci_inner * 16) + (((int)threadIdx.x) * 2)) + p_c)]));
        }
      }
    }
  }
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 4; ++co_inner_inner_inner) {
    for (int p_inner_inner_inner = 0; p_inner_inner_inner < 2; ++p_inner_inner_inner) {
      bgemm[((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 1024)) + (((int)threadIdx.y) * 64)) + (co_inner_inner_inner * 16)) + (((int)threadIdx.x) * 2)) + p_inner_inner_inner)] = bgemm_local[((co_inner_inner_inner * 2) + p_inner_inner_inner)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(196) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[32];
  __shared__ float p1_shared[256];
  __shared__ float data_pack_shared[1568];
  for (int co_c_init = 0; co_c_init < 8; ++co_c_init) {
    bgemm_local[co_c_init] = 0.000000e+00f;
    bgemm_local[(co_c_init + 16)] = 0.000000e+00f;
    bgemm_local[(co_c_init + 8)] = 0.000000e+00f;
    bgemm_local[(co_c_init + 24)] = 0.000000e+00f;
  }
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + (((((int)threadIdx.y) * 49) + (((int)threadIdx.x) >> 1)) >> 1)) < 64) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 2) + ((int)threadIdx.y)) < 3) {
          p1_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.z) * 4096) + (ci_outer * 512)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + (((((int)threadIdx.y) * 49) + (((int)threadIdx.x) >> 1)) >> 1)) >> 3) * 64)) + (((int)blockIdx.y) * 32)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 4) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) & 31))];
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1) {
      data_pack_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x))] = data_pack[(((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 196)) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 8; ++ci_inner) {
      for (int co_c = 0; co_c < 8; ++co_c) {
        bgemm_local[co_c] = (bgemm_local[co_c] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c)] * data_pack_shared[((ci_inner * 196) + ((int)threadIdx.x))]));
        bgemm_local[(co_c + 16)] = (bgemm_local[(co_c + 16)] + (p1_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c) + 16)] * data_pack_shared[((ci_inner * 196) + ((int)threadIdx.x))]));
        bgemm_local[(co_c + 8)] = (bgemm_local[(co_c + 8)] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c)] * data_pack_shared[(((ci_inner * 196) + ((int)threadIdx.x)) + 98)]));
        bgemm_local[(co_c + 24)] = (bgemm_local[(co_c + 24)] + (p1_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c) + 16)] * data_pack_shared[(((ci_inner * 196) + ((int)threadIdx.x)) + 98)]));
      }
    }
  }
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 8; ++co_inner_inner_inner) {
    bgemm[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x))] = bgemm_local[co_inner_inner_inner];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 3136)] = bgemm_local[(co_inner_inner_inner + 16)];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 98)] = bgemm_local[(co_inner_inner_inner + 8)];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 3234)] = bgemm_local[(co_inner_inner_inner + 24)];
  }
}

extern "C" __global__ void __launch_bounds__(224) tvmgen_default_fused_nn_conv2d_add_nn_relu_2_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ T_relu, float* __restrict__ p2) {
  float conv2d_nchw[4];
  __shared__ float pad_temp_shared[580];
  __shared__ float p1_shared[1152];
  for (int ff_init = 0; ff_init < 2; ++ff_init) {
    for (int xx_init = 0; xx_init < 2; ++xx_init) {
      conv2d_nchw[((ff_init * 2) + xx_init)] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((((int)threadIdx.z) * 37) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 580) {
        if ((((((int)threadIdx.y) * 19) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 37) {
          if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 19) {
            pad_temp_shared[((((((int)threadIdx.z) * 37) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = (((1 <= ((((int)blockIdx.y) * 4) + ((((((((int)threadIdx.z) * 37) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 145) / 29))) && (1 <= (((((((int)threadIdx.y) * 19) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 29))) ? p0[((((((rc_outer * 3136) + ((((((((int)threadIdx.z) * 37) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 145) * 784)) + (((int)blockIdx.y) * 112)) + (((((((((int)threadIdx.z) * 37) + (((int)threadIdx.y) * 19)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 145) / 29) * 28)) + (((((((int)threadIdx.y) * 19) + (((int)threadIdx.z) * 8)) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 29)) - 29)] : 0.000000e+00f);
          }
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) {
      if (((((((int)threadIdx.x) / 6) + ((int)threadIdx.y)) >> 1) + ((int)threadIdx.z)) < 16) {
        if (((((((int)threadIdx.z) * 24) + (((int)threadIdx.y) * 12)) + (((int)threadIdx.x) * 2)) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 / 3)) < 384) {
          if (((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) < 1152) {
            if (((((int)threadIdx.x) / 6) + ((int)threadIdx.y)) < 2) {
              if (((int)threadIdx.x) < 6) {
                p1_shared[((((((int)threadIdx.z) * 72) + (((int)threadIdx.y) * 36)) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1)] = p1[((((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (((int)threadIdx.y) * 1152)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1)];
              }
            }
          }
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          for (int ff = 0; ff < 2; ++ff) {
            for (int xx = 0; xx < 2; ++xx) {
              conv2d_nchw[((ff * 2) + xx)] = (conv2d_nchw[((ff * 2) + xx)] + (pad_temp_shared[((((((rc_inner * 145) + (((int)threadIdx.y) * 58)) + (ry_inner * 29)) + (((int)threadIdx.x) * 4)) + (xx * 2)) + rx_inner)] * p1_shared[(((((((int)threadIdx.z) * 72) + (ff * 36)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
            }
          }
        }
      }
    }
  }
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 2; ++ax1_inner_inner_inner) {
    for (int ax3_inner_inner_inner = 0; ax3_inner_inner_inner < 2; ++ax3_inner_inner_inner) {
      T_relu[(((((((((int)blockIdx.z) * 6272) + (((int)threadIdx.z) * 392)) + (ax1_inner_inner_inner * 196)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + ax3_inner_inner_inner)] = max((conv2d_nchw[((ax1_inner_inner_inner * 2) + ax3_inner_inner_inner)] + p2[(((((int)blockIdx.z) * 32) + (((int)threadIdx.z) * 2)) + ax1_inner_inner_inner)]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ p2) {
  float inverse[4];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 25088)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 50176)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 75264)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)]);
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 100352)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 200704)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 301056)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)]);
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 125440)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 150528)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 175616)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 225792)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 250880)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 275968)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 326144)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 351232)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 376320)]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      T_relu[(((((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) / 7) * 56) + (ax2_inner * 28)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 14) * 2)) + ax3_inner)] = max((inverse[((ax2_inner * 2) + ax3_inner)] + p2[(((((int)blockIdx.x) * 32) + (((int)threadIdx.x) >> 2)) / 49)]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0(float* __restrict__ p0, float* __restrict__ data_pack) {
  float d[16];
  float data_pack_local[16];
  for (int eps = 0; eps < 4; ++eps) {
    for (int nu = 0; nu < 4; ++nu) {
      d[((eps * 4) + nu)] = (((((1 <= ((((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps)) && (((((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 2) + eps) < 15)) && (1 <= (((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2) + nu))) && ((((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2) + nu) < 15)) ? p0[(((((((((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) / 49) * 196) + (((((((int)blockIdx.x) * 30) + ((int)threadIdx.x)) % 49) / 7) * 28)) + (eps * 14)) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 7) * 2)) + nu) - 15)] : 0.000000e+00f);
    }
  }
  data_pack_local[0] = 0.000000e+00f;
  data_pack_local[0] = (data_pack_local[0] + d[0]);
  data_pack_local[0] = (data_pack_local[0] + (d[2] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + (d[8] * -1.000000e+00f));
  data_pack_local[0] = (data_pack_local[0] + ((d[10] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = 0.000000e+00f;
  data_pack_local[1] = (data_pack_local[1] + (d[1] * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + d[2]);
  data_pack_local[1] = (data_pack_local[1] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[1] = (data_pack_local[1] + (d[10] * -1.000000e+00f));
  data_pack_local[2] = 0.000000e+00f;
  data_pack_local[2] = (data_pack_local[2] + d[1]);
  data_pack_local[2] = (data_pack_local[2] + d[2]);
  data_pack_local[2] = (data_pack_local[2] + (d[9] * -1.000000e+00f));
  data_pack_local[2] = (data_pack_local[2] + (d[10] * -1.000000e+00f));
  data_pack_local[3] = 0.000000e+00f;
  data_pack_local[3] = (data_pack_local[3] + (d[1] * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + d[3]);
  data_pack_local[3] = (data_pack_local[3] + ((d[9] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[3] = (data_pack_local[3] + (d[11] * -1.000000e+00f));
  data_pack_local[4] = 0.000000e+00f;
  data_pack_local[4] = (data_pack_local[4] + (d[4] * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[4] = (data_pack_local[4] + d[8]);
  data_pack_local[4] = (data_pack_local[4] + (d[10] * -1.000000e+00f));
  data_pack_local[5] = 0.000000e+00f;
  data_pack_local[5] = (data_pack_local[5] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[6] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + (d[9] * -1.000000e+00f));
  data_pack_local[5] = (data_pack_local[5] + d[10]);
  data_pack_local[6] = 0.000000e+00f;
  data_pack_local[6] = (data_pack_local[6] + (d[5] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + (d[6] * -1.000000e+00f));
  data_pack_local[6] = (data_pack_local[6] + d[9]);
  data_pack_local[6] = (data_pack_local[6] + d[10]);
  data_pack_local[7] = 0.000000e+00f;
  data_pack_local[7] = (data_pack_local[7] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[7] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + (d[9] * -1.000000e+00f));
  data_pack_local[7] = (data_pack_local[7] + d[11]);
  data_pack_local[8] = 0.000000e+00f;
  data_pack_local[8] = (data_pack_local[8] + d[4]);
  data_pack_local[8] = (data_pack_local[8] + (d[6] * -1.000000e+00f));
  data_pack_local[8] = (data_pack_local[8] + d[8]);
  data_pack_local[8] = (data_pack_local[8] + (d[10] * -1.000000e+00f));
  data_pack_local[9] = 0.000000e+00f;
  data_pack_local[9] = (data_pack_local[9] + (d[5] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[6]);
  data_pack_local[9] = (data_pack_local[9] + (d[9] * -1.000000e+00f));
  data_pack_local[9] = (data_pack_local[9] + d[10]);
  data_pack_local[10] = 0.000000e+00f;
  data_pack_local[10] = (data_pack_local[10] + d[5]);
  data_pack_local[10] = (data_pack_local[10] + d[6]);
  data_pack_local[10] = (data_pack_local[10] + d[9]);
  data_pack_local[10] = (data_pack_local[10] + d[10]);
  data_pack_local[11] = 0.000000e+00f;
  data_pack_local[11] = (data_pack_local[11] + (d[5] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[7]);
  data_pack_local[11] = (data_pack_local[11] + (d[9] * -1.000000e+00f));
  data_pack_local[11] = (data_pack_local[11] + d[11]);
  data_pack_local[12] = 0.000000e+00f;
  data_pack_local[12] = (data_pack_local[12] + (d[4] * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + ((d[6] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[12] = (data_pack_local[12] + d[12]);
  data_pack_local[12] = (data_pack_local[12] + (d[14] * -1.000000e+00f));
  data_pack_local[13] = 0.000000e+00f;
  data_pack_local[13] = (data_pack_local[13] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[6] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + (d[13] * -1.000000e+00f));
  data_pack_local[13] = (data_pack_local[13] + d[14]);
  data_pack_local[14] = 0.000000e+00f;
  data_pack_local[14] = (data_pack_local[14] + (d[5] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + (d[6] * -1.000000e+00f));
  data_pack_local[14] = (data_pack_local[14] + d[13]);
  data_pack_local[14] = (data_pack_local[14] + d[14]);
  data_pack_local[15] = 0.000000e+00f;
  data_pack_local[15] = (data_pack_local[15] + ((d[5] * -1.000000e+00f) * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[7] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + (d[13] * -1.000000e+00f));
  data_pack_local[15] = (data_pack_local[15] + d[15]);
  for (int eps_1 = 0; eps_1 < 4; ++eps_1) {
    for (int nu_1 = 0; nu_1 < 4; ++nu_1) {
      data_pack[((((eps_1 * 50176) + (nu_1 * 12544)) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = data_pack_local[((eps_1 * 4) + nu_1)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(196) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_kernel1(float* __restrict__ p1, float* __restrict__ data_pack, float* __restrict__ bgemm) {
  float bgemm_local[32];
  __shared__ float p1_shared[256];
  __shared__ float data_pack_shared[1568];
  for (int co_c_init = 0; co_c_init < 8; ++co_c_init) {
    bgemm_local[co_c_init] = 0.000000e+00f;
    bgemm_local[(co_c_init + 16)] = 0.000000e+00f;
    bgemm_local[(co_c_init + 8)] = 0.000000e+00f;
    bgemm_local[(co_c_init + 24)] = 0.000000e+00f;
  }
  for (int ci_outer = 0; ci_outer < 8; ++ci_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + (((((int)threadIdx.y) * 49) + (((int)threadIdx.x) >> 1)) >> 1)) < 64) {
        if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 2) + ((int)threadIdx.y)) < 3) {
          p1_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.z) * 4096) + (ci_outer * 512)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 49) + (((((int)threadIdx.y) * 49) + (((int)threadIdx.x) >> 1)) >> 1)) >> 3) * 64)) + (((int)blockIdx.y) * 32)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 4) + (((int)threadIdx.y) * 2)) + ((int)threadIdx.x)) & 31))];
        }
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 < 8; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1) {
      data_pack_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 196) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x))] = data_pack[(((((((int)blockIdx.z) * 12544) + (ci_outer * 1568)) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer_1 * 196)) + (((int)threadIdx.y) * 98)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int ci_inner = 0; ci_inner < 8; ++ci_inner) {
      for (int co_c = 0; co_c < 8; ++co_c) {
        bgemm_local[co_c] = (bgemm_local[co_c] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c)] * data_pack_shared[((ci_inner * 196) + ((int)threadIdx.x))]));
        bgemm_local[(co_c + 16)] = (bgemm_local[(co_c + 16)] + (p1_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c) + 16)] * data_pack_shared[((ci_inner * 196) + ((int)threadIdx.x))]));
        bgemm_local[(co_c + 8)] = (bgemm_local[(co_c + 8)] + (p1_shared[(((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c)] * data_pack_shared[(((ci_inner * 196) + ((int)threadIdx.x)) + 98)]));
        bgemm_local[(co_c + 24)] = (bgemm_local[(co_c + 24)] + (p1_shared[((((ci_inner * 32) + (((int)threadIdx.y) * 8)) + co_c) + 16)] * data_pack_shared[(((ci_inner * 196) + ((int)threadIdx.x)) + 98)]));
      }
    }
  }
  for (int co_inner_inner_inner = 0; co_inner_inner_inner < 8; ++co_inner_inner_inner) {
    bgemm[(((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x))] = bgemm_local[co_inner_inner_inner];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 3136)] = bgemm_local[(co_inner_inner_inner + 16)];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 98)] = bgemm_local[(co_inner_inner_inner + 8)];
    bgemm[((((((((int)blockIdx.z) * 12544) + (((int)blockIdx.y) * 6272)) + (((int)threadIdx.y) * 1568)) + (co_inner_inner_inner * 196)) + ((int)threadIdx.x)) + 3234)] = bgemm_local[(co_inner_inner_inner + 24)];
  }
}

extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_contrib_conv2d_winograd_without_weight_transform_add_multiply_add_nn_re_a4e88f85cf7823fc__kernel2(float* __restrict__ bgemm, float* __restrict__ T_relu, float* __restrict__ p2, float* __restrict__ p3, float* __restrict__ p4) {
  float inverse[4];
  inverse[0] = 0.000000e+00f;
  inverse[0] = (inverse[0] + bgemm[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)]);
  inverse[0] = (inverse[0] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[1] = 0.000000e+00f;
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 8192)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 16384)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 24576)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344)]);
  inverse[1] = (inverse[1] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)] * -1.000000e+00f));
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[1] = (inverse[1] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112)]);
  inverse[2] = 0.000000e+00f;
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 32768)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)] * -1.000000e+00f));
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 65536)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 98304)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496)]);
  inverse[2] = (inverse[2] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688)]);
  inverse[3] = 0.000000e+00f;
  inverse[3] = (inverse[3] + ((bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 40960)] * -1.000000e+00f) * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 49152)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 57344)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 73728)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 81920)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 90112)]);
  inverse[3] = (inverse[3] + (bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 106496)] * -1.000000e+00f));
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 114688)]);
  inverse[3] = (inverse[3] + bgemm[(((((int)blockIdx.x) * 128) + ((int)threadIdx.x)) + 122880)]);
  for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
    for (int ax3_inner = 0; ax3_inner < 2; ++ax3_inner) {
      if (((((((int)threadIdx.x) & 15) >> 2) * 2) + ax2_inner) < 7) {
        if ((((((int)threadIdx.x) & 3) * 2) + ax3_inner) < 7) {
          T_relu[((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (ax2_inner * 7)) + ((((int)threadIdx.x) & 3) * 2)) + ax3_inner)] = max((((inverse[((ax2_inner * 2) + ax3_inner)] + p2[((((((((int)blockIdx.x) * 392) + ((((int)threadIdx.x) >> 4) * 49)) + (((((int)threadIdx.x) & 15) >> 2) * 14)) + (ax2_inner * 7)) + ((((int)threadIdx.x) & 3) * 2)) + ax3_inner)]) * p3[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4))]) + p4[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 4))]), 0.000000e+00f);
        }
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(512) tvmgen_default_fused_nn_batch_flatten_kernel0(float* __restrict__ tensor, float* __restrict__ p0) {
  tensor[((int)threadIdx.x)] = p0[((int)threadIdx.x)];
}

extern "C" __global__ void __launch_bounds__(64) tvmgen_default_fused_nn_dense_add_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ T_add, float* __restrict__ p2) {
  float T_matmul_NT_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_matmul_NT[1];
  T_matmul_NT_rf[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    T_matmul_NT_rf[0] = (T_matmul_NT_rf[0] + (p0[((k_outer * 64) + ((int)threadIdx.x))] * p1[(((((int)blockIdx.x) * 512) + (k_outer * 64)) + ((int)threadIdx.x))]));
  }
  __syncthreads();
  ((volatile float*)red_buf0)[((int)threadIdx.x)] = T_matmul_NT_rf[0];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 32)]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    float w_16_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 16)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_16_0;
    float w_8_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 8)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_8_0;
    float w_4_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 4)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_4_0;
    float w_2_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 2)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_2_0;
    float w_1_0 = (((volatile float*)red_buf0)[((int)threadIdx.x)] + ((volatile float*)red_buf0)[(((int)threadIdx.x) + 1)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_1_0;
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_matmul_NT[0] = ((volatile float*)red_buf0)[0];
  }
  if (((int)threadIdx.x) == 0) {
    T_add[((int)blockIdx.x)] = (T_matmul_NT[0] + p2[((int)blockIdx.x)]);
  }
}

