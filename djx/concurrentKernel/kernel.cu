__global__ void fused_nn_conv2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
    float compute[1];
    __shared__ float pad_temp_shared[180];
    __shared__ float placeholder_shared[576];
    compute[(0)] = 0.000000e+00f;
    for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
      __syncthreads();
      if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 180) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)))] = (((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) % 45) / 15))) && (1 <= (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) % 15))) ? placeholder[(((((((rc_outer * 784) + ((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) / 45) * 196)) + (((int)blockIdx.y) * 28)) + (((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) % 45) / 15) * 14)) + (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) % 15)) - 15))] : 0.000000e+00f);
        }
      }
      if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 179) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1))] = (((1 <= ((((int)blockIdx.y) * 2) + (((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1) % 45) / 15))) && (1 <= ((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1) % 15))) ? placeholder[(((((((rc_outer * 784) + (((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1) / 45) * 196)) + (((int)blockIdx.y) * 28)) + ((((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1) % 45) / 15) * 14)) + ((((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) + 1) % 15)) - 15))] : 0.000000e+00f);
        }
      }
      if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 16) {
        if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
          if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 192) {
            if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) < 576) {
              if (((int)threadIdx.x) < 6) {
                placeholder_shared[(((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)))] = placeholder1[(((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)))];
              }
            }
          }
        }
      }
      if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 16) {
        if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
          if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 192) {
            if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) < 575) {
              if (((int)threadIdx.x) < 6) {
                placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 1))];
              }
            }
          }
        }
      }
      if (((((int)threadIdx.x) / 6) + ((int)threadIdx.z)) < 16) {
        if (((((int)threadIdx.z) * 4) + ((((int)threadIdx.x) * 2) / 3)) < 64) {
          if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 192) {
            if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) < 574) {
              if (((int)threadIdx.x) < 6) {
                placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 2))] = placeholder1[((((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 2))];
              }
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 2) + 1) / 12) + ((int)threadIdx.z)) < 16) {
        if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
          if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 191) {
            if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) < 573) {
              if (((int)threadIdx.x) < 6) {
                placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 3))] = placeholder1[((((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 3))];
              }
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 2) + 1) / 12) + ((int)threadIdx.z)) < 16) {
        if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
          if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 191) {
            if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) < 572) {
              if (((int)threadIdx.x) < 6) {
                placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 4))] = placeholder1[((((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 4))];
              }
            }
          }
        }
      }
      if (((((((int)threadIdx.x) * 2) + 1) / 12) + ((int)threadIdx.z)) < 16) {
        if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + 1) / 3)) < 64) {
          if (((((int)threadIdx.z) * 12) + (((int)threadIdx.x) * 2)) < 191) {
            if (((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) < 571) {
              if (((int)threadIdx.x) < 6) {
                placeholder_shared[((((((int)threadIdx.z) * 36) + (((int)threadIdx.x) * 6)) + 5))] = placeholder1[((((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 36)) + (((int)threadIdx.x) * 6)) + 5))];
              }
            }
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) * 2))] * placeholder_shared[((((int)threadIdx.z) * 36))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 1))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 2))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 15))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 3))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 16))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 4))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 17))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 5))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 30))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 6))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 31))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 7))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 32))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 8))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 45))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 9))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 46))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 10))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 47))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 11))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 60))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 12))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 61))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 13))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 62))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 14))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 75))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 15))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 76))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 16))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 77))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 17))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 90))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 18))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 91))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 19))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 92))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 20))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 105))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 21))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 106))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 22))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 107))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 23))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 120))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 24))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 121))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 25))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 122))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 26))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 135))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 27))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 136))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 28))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 137))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 29))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 150))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 30))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 151))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 31))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 152))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 32))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 165))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 33))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 166))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 34))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.x) * 2) + 167))] * placeholder_shared[(((((int)threadIdx.z) * 36) + 35))]));
    }
    T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  }