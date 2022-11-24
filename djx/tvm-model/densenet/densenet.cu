extern "C" __global__ void fused_nn_conv2d_add_nn_relu_12_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[465];
  __shared__ float placeholder_shared[3];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 675))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 675))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 675))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[(((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 674))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 674))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 674))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 1))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 673))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 673))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 673))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 2))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 672))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 672))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 672))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 3))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 671))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 671))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 671))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 4))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 670))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 670))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 670))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 5))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 669))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 669))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 669))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 6))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 451))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 451))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 451))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 7))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 450))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 450))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 450))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 8))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 449))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 449))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 449))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 9))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 448))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 448))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 448))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 10))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 447))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 447))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 447))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 11))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 446))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 446))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 446))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 12))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 445))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 445))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 445))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 13))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 227))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 227))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 227))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 14))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 226))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 226))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 226))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 15))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 225))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 225))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 225))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 16))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 224))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 224))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 224))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 17))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 223))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 223))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 223))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 18))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 222))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 222))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 222))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 19))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 221))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 221))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 221))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 20))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 3))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 3))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 3))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 21))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 2))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 2))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 2))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 22))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) - 1))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) - 1))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) - 1))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 23))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 24))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 1))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 1))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 1))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 25))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 2))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 2))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 2))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 26))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 3))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 3))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 3))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 27))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 221))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 221))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 221))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 28))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 222))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 222))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 222))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 29))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 223))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 223))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 223))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 30))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 224))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 224))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 224))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 31))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 225))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 225))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 225))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 32))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 226))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 226))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 226))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 33))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 227))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 227))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 227))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 34))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 445))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 445))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 445))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 35))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 446))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 446))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 446))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 36))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 447))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 447))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 447))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 37))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 448))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 448))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 448))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 38))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 449))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 449))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 449))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 39))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 450))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 450))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 450))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 40))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 451))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 451))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 451))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 41))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 669))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 669))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 669))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 42))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 670))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 670))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 670))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 43))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 671))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 671))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 671))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 44))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 672))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 672))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 672))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 45))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 673))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 673))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 673))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 46))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 674))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 674))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 674))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 47))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 675))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 675))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 675))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 48))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49501))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49501))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49501))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 49))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49502))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49502))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49502))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 50))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49503))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49503))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49503))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 51))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49504))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49504))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49504))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 52))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49505))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49505))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49505))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 53))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49506))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49506))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49506))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 54))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49507))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49507))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49507))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 55))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49725))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49725))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49725))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 56))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49726))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49726))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49726))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 57))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49727))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49727))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49727))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 58))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49728))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49728))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49728))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 59))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49729))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49729))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49729))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 60))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49730))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49730))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49730))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 61))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49731))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49731))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49731))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 62))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49949))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49949))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49949))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 63))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49950))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49950))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49950))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 64))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49951))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49951))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49951))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 65))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49952))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49952))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49952))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 66))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49953))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49953))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49953))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 67))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49954))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49954))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49954))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 68))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 49955))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 49955))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 49955))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 69))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50173))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50173))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50173))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 70))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50174))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50174))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50174))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 71))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50175))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50175))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50175))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 72))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50176))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50176))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50176))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 73))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50177))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50177))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50177))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 74))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50178))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50178))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50178))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 75))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50179))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50179))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50179))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 76))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50397))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50397))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50397))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 77))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50398))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50398))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50398))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 78))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50399))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50399))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50399))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 79))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50400))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50400))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50400))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 80))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50401))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50401))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50401))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 81))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50402))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50402))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50402))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 82))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50403))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50403))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50403))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 83))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50621))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50621))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50621))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 84))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50622))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50622))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50622))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 85))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50623))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50623))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50623))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 86))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50624))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50624))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50624))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 87))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50625))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50625))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50625))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 88))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50626))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50626))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50626))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 89))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50627))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50627))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50627))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 90))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50845))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50845))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50845))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 91))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50846))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50846))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50846))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 92))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50847))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50847))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50847))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 93))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50848))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50848))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50848))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 94))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50849))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50849))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50849))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 95))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50850))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50850))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50850))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 96))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 50851))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 50851))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 50851))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 97))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99677))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99677))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99677))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 98))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99678))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99678))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99678))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 99))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99679))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99679))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99679))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 100))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99680))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99680))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99680))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 101))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99681))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99681))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99681))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 102))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99682))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99682))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99682))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 103))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99683))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99683))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((3 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99683))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 104))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99901))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99901))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99901))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 105))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99902))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99902))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99902))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 106))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99903))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99903))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99903))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 107))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99904))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99904))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99904))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 108))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99905))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99905))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99905))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 109))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99906))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99906))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99906))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 110))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 99907))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 99907))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((2 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 99907))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 111))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100125))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100125))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100125))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 112))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100126))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100126))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100126))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 113))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100127))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100127))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100127))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 114))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100128))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100128))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100128))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 115))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100129))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100129))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100129))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 116))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100130))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100130))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100130))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 117))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31))) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100131))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100131))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((1 <= (((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31))) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100131))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 118))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100349))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100349))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100349))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 119))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100350))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100350))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100350))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 120))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100351))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100351))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100351))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 121))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100352))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100352))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100352))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 122))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100353))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100353))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100353))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 123))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100354))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100354))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100354))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 124))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100355))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100355))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100355))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 125))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100573))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100573))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100573))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 126))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100574))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100574))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100574))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 127))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100575))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100575))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100575))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 128))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100576))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100576))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100576))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 129))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100577))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100577))];
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100577))];
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 130))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100578))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100578))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100578))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 131))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100579))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100579))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100579))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 132))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100797))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100797))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100797))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 133))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100798))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100798))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100798))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 134))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100799))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100799))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100799))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 135))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100800))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100800))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100800))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 136))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100801))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100801))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100801))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 137))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100802))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100802))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100802))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 138))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 222) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 100803))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 100803))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 222) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 100803))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 139))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (3 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 101021))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 101021))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (3 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 101021))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 140))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (2 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 101022))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 101022))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (2 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 101022))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 141))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (1 <= ((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 101023))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 101023))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (1 <= ((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)))) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 101023))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 142))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 101024))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 101024))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 101024))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 143))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 101025))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 101025))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = (((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 101025))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 144))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 101026))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 101026))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 222)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 101026))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 145))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  __syncthreads();
  if (((((int)threadIdx.z) * 5) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 465) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 155) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31)) < 221) && (((((int)blockIdx.x) * 32) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) / 31) * 224)) + (((int)blockIdx.x) * 32)) + (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) % 31)) + 101027))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 464) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 154) {
        if (((int)threadIdx.x) < 7) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 1))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 1) % 31)) + 101027))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.z) * 5) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 15) {
    if ((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) < 463) {
      if (((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) < 153) {
        if (((int)threadIdx.x) < 6) {
          pad_temp_shared[(((((((int)threadIdx.z) * 155) + (((int)threadIdx.y) * 20)) + (((int)threadIdx.x) * 3)) + 2))] = ((((((((int)blockIdx.y) * 16) + (((int)threadIdx.z) * 5)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31)) < 221) && (((((int)blockIdx.x) * 32) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) < 221)) ? placeholder[(((((((((int)blockIdx.y) * 3584) + (((int)threadIdx.z) * 1120)) + (((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) / 31) * 224)) + (((int)blockIdx.x) * 32)) + ((((((int)threadIdx.y) * 20) + (((int)threadIdx.x) * 3)) + 2) % 31)) + 101027))] : 0.000000e+00f);
        }
      }
    }
  }
  if (((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)) < 3) {
    if ((((int)threadIdx.x) + ((int)threadIdx.y)) < 1) {
      if (((int)threadIdx.x) < 1) {
        placeholder_shared[(((((int)threadIdx.x) + ((int)threadIdx.y)) + ((int)threadIdx.z)))] = placeholder1[((((((((int)blockIdx.z) * 441) + (((int)threadIdx.x) * 147)) + (((int)threadIdx.y) * 147)) + (((int)threadIdx.z) * 147)) + 146))];
      }
    }
  }
  __syncthreads();
  compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)))] * placeholder_shared[(((int)threadIdx.z))]));
  compute[(1)] = (compute[(1)] + (pad_temp_shared[((((((int)threadIdx.y) * 62) + (((int)threadIdx.x) * 4)) + 2))] * placeholder_shared[(((int)threadIdx.z))]));
  T_relu[(((((((((int)blockIdx.z) * 37632) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.y) * 896)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 2)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 3) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[((((((((((int)blockIdx.z) * 37632) + (((int)threadIdx.z) * 12544)) + (((int)blockIdx.y) * 896)) + (((int)threadIdx.y) * 112)) + (((int)blockIdx.x) * 16)) + (((int)threadIdx.x) * 2)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 3) + ((int)threadIdx.z)))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_8_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[280];
  __shared__ float placeholder_shared[160];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 4; ++yy_init) {
      compute[(((ff_init * 4) + yy_init))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 26; ++rc_outer) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 35) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 280) {
        if (((((int)threadIdx.x) * 3) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 35) {
          pad_temp_shared[((((((int)threadIdx.z) * 35) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 3920) + (((((((int)threadIdx.z) * 35) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 56) * 784)) + (((int)blockIdx.y) * 112)) + ((((((((int)threadIdx.z) * 35) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 56) / 14) * 28)) + (((int)blockIdx.x) * 14)) + ((((((int)threadIdx.z) * 35) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 14)))];
        }
      }
    }
    #pragma unroll
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((((int)threadIdx.z) * 4) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 5)) < 32) {
        if ((((((int)threadIdx.z) * 20) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 160) {
          if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) < 20) {
            placeholder_shared[((((((int)threadIdx.z) * 20) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1))] = placeholder1[(((((((int)threadIdx.z) * 520) + ((((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) / 5) * 130)) + (rc_outer * 5)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) % 5)))];
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int rc_inner = 0; rc_inner < 5; ++rc_inner) {
      #pragma unroll
      for (int ff = 0; ff < 4; ++ff) {
        #pragma unroll
        for (int yy = 0; yy < 4; ++yy) {
          compute[(((ff * 4) + yy))] = (compute[(((ff * 4) + yy))] + (pad_temp_shared[((((rc_inner * 56) + (yy * 14)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 20) + (ff * 5)) + rc_inner))]));
        }
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 4; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 4; ++ax2_inner_inner_inner) {
      T_relu[(((((((((int)threadIdx.z) * 3136) + (ax1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)))] = max((compute[(((ax1_inner_inner_inner * 4) + ax2_inner_inner_inner))] + placeholder2[(((((int)threadIdx.z) * 4) + ax1_inner_inner_inner))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_4_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[8];
  __shared__ float pad_temp_shared[28];
  __shared__ float placeholder_shared[32];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 2; ++yy_init) {
      compute[(((ff_init * 2) + yy_init))] = 0.000000e+00f;
    }
  }
  if (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) < 28) {
    if (((int)threadIdx.x) < 4) {
      pad_temp_shared[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))] = placeholder[((((((int)blockIdx.y) * 28) + (((int)threadIdx.z) * 4)) + ((int)threadIdx.x)))];
    }
  }
  if (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) < 32) {
    if (((int)threadIdx.x) < 4) {
      placeholder_shared[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))] = placeholder1[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ff = 0; ff < 4; ++ff) {
    #pragma unroll
    for (int yy = 0; yy < 2; ++yy) {
      compute[(((ff * 2) + yy))] = (compute[(((ff * 2) + yy))] + (pad_temp_shared[(((yy * 14) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 4) + ff))]));
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 4; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
      T_relu[((((((((int)threadIdx.z) * 784) + (ax1_inner_inner_inner * 196)) + (((int)blockIdx.y) * 28)) + (ax2_inner_inner_inner * 14)) + ((int)threadIdx.x)))] = max((compute[(((ax1_inner_inner_inner * 2) + ax2_inner_inner_inner))] + placeholder2[(((((int)threadIdx.z) * 4) + ax1_inner_inner_inner))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[28];
  __shared__ float pad_temp_shared[98];
  __shared__ float placeholder_shared[32];
  compute[(0)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 320; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 25) + (((int)threadIdx.y) * 4)) < 98) {
      pad_temp_shared[(((((int)threadIdx.z) * 25) + (((int)threadIdx.y) * 4)))] = placeholder[((((rc_outer * 98) + (((int)threadIdx.z) * 25)) + (((int)threadIdx.y) * 4)))];
    }
    if (((((int)threadIdx.z) * 25) + (((int)threadIdx.y) * 4)) < 97) {
      if (((int)threadIdx.y) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 25) + (((int)threadIdx.y) * 4)) + 1))] = placeholder[(((((rc_outer * 98) + (((int)threadIdx.z) * 25)) + (((int)threadIdx.y) * 4)) + 1))];
      }
    }
    if (((((int)threadIdx.z) * 25) + (((int)threadIdx.y) * 4)) < 96) {
      if (((int)threadIdx.y) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 25) + (((int)threadIdx.y) * 4)) + 2))] = placeholder[(((((rc_outer * 98) + (((int)threadIdx.z) * 25)) + (((int)threadIdx.y) * 4)) + 2))];
      }
    }
    if (((((int)threadIdx.z) * 25) + (((int)threadIdx.y) * 4)) < 95) {
      if (((int)threadIdx.y) < 6) {
        pad_temp_shared[((((((int)threadIdx.z) * 25) + (((int)threadIdx.y) * 4)) + 3))] = placeholder[(((((rc_outer * 98) + (((int)threadIdx.z) * 25)) + (((int)threadIdx.y) * 4)) + 3))];
      }
    }
    if (((((int)threadIdx.z) * 4) + ((int)threadIdx.y)) < 16) {
      if (((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) < 32) {
        if (((int)threadIdx.y) < 4) {
          placeholder_shared[(((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)))] = placeholder1[(((((((int)blockIdx.z) * 10240) + (((int)threadIdx.z) * 2560)) + (((int)threadIdx.y) * 640)) + (rc_outer * 2)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + ((int)threadIdx.y)) < 16) {
      if (((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) < 31) {
        if (((int)threadIdx.y) < 4) {
          placeholder_shared[((((((int)threadIdx.z) * 8) + (((int)threadIdx.y) * 2)) + 1))] = placeholder1[((((((((int)blockIdx.z) * 10240) + (((int)threadIdx.z) * 2560)) + (((int)threadIdx.y) * 640)) + (rc_outer * 2)) + 1))];
        }
      }
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.y) * 7))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.y) * 7))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 8))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.y) * 7))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.y) * 7))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 24))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 1))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 8))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 1))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 24))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 2))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 8))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 2))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 24))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 3))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 8))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 3))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 24))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 4))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 8))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 4))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 24))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 5))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 8))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 5))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 24))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 6))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 8))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 6))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 24))]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
    compute[(7)] = (compute[(7)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 9))]));
    compute[(14)] = (compute[(14)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
    compute[(21)] = (compute[(21)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 49))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 25))]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
    compute[(8)] = (compute[(8)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 9))]));
    compute[(15)] = (compute[(15)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
    compute[(22)] = (compute[(22)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 50))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 25))]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
    compute[(9)] = (compute[(9)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 9))]));
    compute[(16)] = (compute[(16)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
    compute[(23)] = (compute[(23)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 51))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 25))]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
    compute[(10)] = (compute[(10)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 9))]));
    compute[(17)] = (compute[(17)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
    compute[(24)] = (compute[(24)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 52))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 25))]));
    compute[(4)] = (compute[(4)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 53))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
    compute[(11)] = (compute[(11)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 53))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 9))]));
    compute[(18)] = (compute[(18)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 53))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
    compute[(25)] = (compute[(25)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 53))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 25))]));
    compute[(5)] = (compute[(5)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
    compute[(12)] = (compute[(12)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 9))]));
    compute[(19)] = (compute[(19)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
    compute[(26)] = (compute[(26)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 54))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 25))]));
    compute[(6)] = (compute[(6)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
    compute[(13)] = (compute[(13)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 9))]));
    compute[(20)] = (compute[(20)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
    compute[(27)] = (compute[(27)] + (pad_temp_shared[(((((int)threadIdx.y) * 7) + 55))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 25))]));
  }
  T_relu[((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)))] = max((compute[(0)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 196))] = max((compute[(7)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 4))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 392))] = max((compute[(14)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 588))] = max((compute[(21)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 12))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 1))] = max((compute[(1)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 197))] = max((compute[(8)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 4))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 393))] = max((compute[(15)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 589))] = max((compute[(22)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 12))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 2))] = max((compute[(2)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 198))] = max((compute[(9)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 4))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 394))] = max((compute[(16)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 590))] = max((compute[(23)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 12))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 3))] = max((compute[(3)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 199))] = max((compute[(10)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 4))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 395))] = max((compute[(17)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 591))] = max((compute[(24)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 12))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 4))] = max((compute[(4)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 200))] = max((compute[(11)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 4))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 396))] = max((compute[(18)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 592))] = max((compute[(25)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 12))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 5))] = max((compute[(5)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 201))] = max((compute[(12)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 4))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 397))] = max((compute[(19)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 593))] = max((compute[(26)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 12))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 6))] = max((compute[(6)] + placeholder2[(((((int)blockIdx.z) * 16) + ((int)threadIdx.z)))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 202))] = max((compute[(13)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 4))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 398))] = max((compute[(20)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
  T_relu[(((((((int)blockIdx.z) * 784) + (((int)threadIdx.z) * 49)) + (((int)threadIdx.y) * 7)) + 594))] = max((compute[(27)] + placeholder2[((((((int)blockIdx.z) * 16) + ((int)threadIdx.z)) + 12))]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[9];
  __shared__ float placeholder_shared[3];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 5) {
      pad_temp_shared[((((int)threadIdx.x) * 2))] = ((((1 <= ((int)blockIdx.y)) && (1 <= ((int)threadIdx.x))) && (((int)threadIdx.x) < 4)) ? placeholder[(((((rc_outer * 49) + (((int)blockIdx.y) * 7)) + (((int)threadIdx.x) * 2)) - 8))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 4) {
      pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] = ((1 <= ((int)blockIdx.y)) ? placeholder[(((((rc_outer * 49) + (((int)blockIdx.y) * 7)) + (((int)threadIdx.x) * 2)) - 7))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[(((rc_outer * 9) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    __syncthreads();
    if (((int)threadIdx.x) < 5) {
      pad_temp_shared[((((int)threadIdx.x) * 2))] = (((1 <= ((int)threadIdx.x)) && (((int)threadIdx.x) < 4)) ? placeholder[(((((rc_outer * 49) + (((int)blockIdx.y) * 7)) + (((int)threadIdx.x) * 2)) - 1))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 4) {
      pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] = placeholder[((((rc_outer * 49) + (((int)blockIdx.y) * 7)) + (((int)threadIdx.x) * 2)))];
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((rc_outer * 9) + ((int)threadIdx.x)) + 3))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    __syncthreads();
    if (((int)threadIdx.x) < 5) {
      pad_temp_shared[((((int)threadIdx.x) * 2))] = ((((((int)blockIdx.y) < 6) && (1 <= ((int)threadIdx.x))) && (((int)threadIdx.x) < 4)) ? placeholder[(((((rc_outer * 49) + (((int)blockIdx.y) * 7)) + (((int)threadIdx.x) * 2)) + 6))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 4) {
      pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] = ((((int)blockIdx.y) < 6) ? placeholder[(((((rc_outer * 49) + (((int)blockIdx.y) * 7)) + (((int)threadIdx.x) * 2)) + 7))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((rc_outer * 9) + ((int)threadIdx.x)) + 6))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
  }
  T_relu[(((((int)blockIdx.y) * 7) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(0)]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_avg_pool2d_kernel0(float* __restrict__ placeholder, float* __restrict__ tensor) {
  float tensor1[1];
  tensor1[(0)] = 0.000000e+00f;
  for (int dh = 0; dh < 7; ++dh) {
    for (int dw = 0; dw < 7; ++dw) {
      if (((int)threadIdx.x) < 1) {
        tensor1[(0)] = (tensor1[(0)] + placeholder[((((((int)threadIdx.x) * 49) + (dh * 7)) + dw))]);
      }
    }
  }
  if (((int)threadIdx.x) < 1) {
    tensor[(((int)threadIdx.x))] = (tensor1[(0)] * 2.040816e-02f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_11_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[336];
  __shared__ float placeholder_shared[96];
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
  for (int rc_outer = 0; rc_outer < 23; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 336) {
        if (((((int)threadIdx.x) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 42) {
          pad_temp_shared[((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = placeholder[(((((((rc_outer * 9408) + (((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 112) * 3136)) + (((int)blockIdx.y) * 224)) + ((((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 112) / 28) * 56)) + (((int)blockIdx.x) * 28)) + ((((((int)threadIdx.z) * 42) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 28)))];
        }
      }
    }
    if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 32) {
      if (((((int)threadIdx.z) * 12) + ((int)threadIdx.x)) < 96) {
        if (((int)threadIdx.x) < 12) {
          placeholder_shared[(((((int)threadIdx.z) * 12) + ((int)threadIdx.x)))] = placeholder1[(((((((int)threadIdx.z) * 276) + ((((int)threadIdx.x) / 3) * 69)) + (rc_outer * 3)) + (((int)threadIdx.x) % 3)))];
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
      for (int yy = 0; yy < 2; ++yy) {
        compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 3) + rc_inner))]));
        compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 24))]));
        compute[((yy + 8))] = (compute[((yy + 8))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 48))]));
        compute[((yy + 12))] = (compute[((yy + 12))] + (pad_temp_shared[((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 72))]));
        compute[((yy + 2))] = (compute[((yy + 2))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 3) + rc_inner))]));
        compute[((yy + 6))] = (compute[((yy + 6))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 24))]));
        compute[((yy + 10))] = (compute[((yy + 10))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 48))]));
        compute[((yy + 14))] = (compute[((yy + 14))] + (pad_temp_shared[(((((rc_inner * 112) + (yy * 28)) + ((int)threadIdx.x)) + 56))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 72))]));
      }
    }
  }
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    T_relu[((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)))] = max((compute[(ax2_inner_inner_inner)] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 25088))] = max((compute[((ax2_inner_inner_inner + 4))] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 50176))] = max((compute[((ax2_inner_inner_inner + 8))] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 75264))] = max((compute[((ax2_inner_inner_inner + 12))] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 112))] = max((compute[((ax2_inner_inner_inner + 2))] + placeholder2[(((int)threadIdx.z))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 25200))] = max((compute[((ax2_inner_inner_inner + 6))] + placeholder2[((((int)threadIdx.z) + 8))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 50288))] = max((compute[((ax2_inner_inner_inner + 10))] + placeholder2[((((int)threadIdx.z) + 16))]), 0.000000e+00f);
    T_relu[(((((((((int)threadIdx.z) * 3136) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 28)) + ((int)threadIdx.x)) + 75376))] = max((compute[((ax2_inner_inner_inner + 14))] + placeholder2[((((int)threadIdx.z) + 24))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_10_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[56];
  __shared__ float placeholder_shared[32];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 4; ++yy_init) {
      compute[(((ff_init * 4) + yy_init))] = 0.000000e+00f;
    }
  }
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 56) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = placeholder[(((((((int)blockIdx.y) * 224) + ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 14) * 56)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) % 14)))];
    }
  }
  if (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) < 32) {
    if (((int)threadIdx.x) < 4) {
      placeholder_shared[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))] = placeholder1[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ff = 0; ff < 4; ++ff) {
    #pragma unroll
    for (int yy = 0; yy < 4; ++yy) {
      compute[(((ff * 4) + yy))] = (compute[(((ff * 4) + yy))] + (pad_temp_shared[(((yy * 14) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 4) + ff))]));
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 4; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 4; ++ax2_inner_inner_inner) {
      T_relu[(((((((((int)threadIdx.z) * 12544) + (ax1_inner_inner_inner * 3136)) + (((int)blockIdx.y) * 224)) + (ax2_inner_inner_inner * 56)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)))] = max((compute[(((ax1_inner_inner_inner * 4) + ax2_inner_inner_inner))] + placeholder2[(((((int)threadIdx.z) * 4) + ax1_inner_inner_inner))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_avg_pool2d_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
  float tensor[1];
  tensor[(0)] = 0.000000e+00f;
  for (int dh = 0; dh < 2; ++dh) {
    for (int dw = 0; dw < 2; ++dw) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 50372) {
        tensor[(0)] = (tensor[(0)] + placeholder[((((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 14) * 56) + (dh * 28)) + ((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 14) * 2)) + dw))]);
      }
    }
  }
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 50372) {
    T_relu[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = max(((tensor[(0)] * 2.500000e-01f) + placeholder1[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 196))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_batch_flatten_kernel0(float* __restrict__ tensor, float* __restrict__ placeholder) {
  tensor[(0)] = placeholder[(0)];
}

extern "C" __global__ void fused_nn_conv2d_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[56];
  __shared__ float pad_temp_shared[98];
  __shared__ float placeholder_shared[64];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(14)] = 0.000000e+00f;
  compute_local[(28)] = 0.000000e+00f;
  compute_local[(42)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(15)] = 0.000000e+00f;
  compute_local[(29)] = 0.000000e+00f;
  compute_local[(43)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(16)] = 0.000000e+00f;
  compute_local[(30)] = 0.000000e+00f;
  compute_local[(44)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(17)] = 0.000000e+00f;
  compute_local[(31)] = 0.000000e+00f;
  compute_local[(45)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(18)] = 0.000000e+00f;
  compute_local[(32)] = 0.000000e+00f;
  compute_local[(46)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(19)] = 0.000000e+00f;
  compute_local[(33)] = 0.000000e+00f;
  compute_local[(47)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(20)] = 0.000000e+00f;
  compute_local[(34)] = 0.000000e+00f;
  compute_local[(48)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  compute_local[(21)] = 0.000000e+00f;
  compute_local[(35)] = 0.000000e+00f;
  compute_local[(49)] = 0.000000e+00f;
  compute_local[(8)] = 0.000000e+00f;
  compute_local[(22)] = 0.000000e+00f;
  compute_local[(36)] = 0.000000e+00f;
  compute_local[(50)] = 0.000000e+00f;
  compute_local[(9)] = 0.000000e+00f;
  compute_local[(23)] = 0.000000e+00f;
  compute_local[(37)] = 0.000000e+00f;
  compute_local[(51)] = 0.000000e+00f;
  compute_local[(10)] = 0.000000e+00f;
  compute_local[(24)] = 0.000000e+00f;
  compute_local[(38)] = 0.000000e+00f;
  compute_local[(52)] = 0.000000e+00f;
  compute_local[(11)] = 0.000000e+00f;
  compute_local[(25)] = 0.000000e+00f;
  compute_local[(39)] = 0.000000e+00f;
  compute_local[(53)] = 0.000000e+00f;
  compute_local[(12)] = 0.000000e+00f;
  compute_local[(26)] = 0.000000e+00f;
  compute_local[(40)] = 0.000000e+00f;
  compute_local[(54)] = 0.000000e+00f;
  compute_local[(13)] = 0.000000e+00f;
  compute_local[(27)] = 0.000000e+00f;
  compute_local[(41)] = 0.000000e+00f;
  compute_local[(55)] = 0.000000e+00f;
  if (((((int)threadIdx.z) * 13) + ((int)threadIdx.x)) < 98) {
    if (((int)threadIdx.x) < 13) {
      pad_temp_shared[(((((int)threadIdx.z) * 13) + ((int)threadIdx.x)))] = placeholder[((((((int)blockIdx.y) * 98) + (((int)threadIdx.z) * 13)) + ((int)threadIdx.x)))];
    }
  }
  if (((((int)threadIdx.z) * 8) + ((int)threadIdx.x)) < 64) {
    if (((int)threadIdx.x) < 8) {
      placeholder_shared[(((((int)threadIdx.z) * 8) + ((int)threadIdx.x)))] = placeholder1[((((((int)blockIdx.z) * 64) + (((int)threadIdx.z) * 8)) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute_local[(14)] = (compute_local[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
  compute_local[(28)] = (compute_local[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 32))]));
  compute_local[(42)] = (compute_local[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 48))]));
  compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute_local[(15)] = (compute_local[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
  compute_local[(29)] = (compute_local[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 32))]));
  compute_local[(43)] = (compute_local[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 48))]));
  compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute_local[(16)] = (compute_local[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
  compute_local[(30)] = (compute_local[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 32))]));
  compute_local[(44)] = (compute_local[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 48))]));
  compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute_local[(17)] = (compute_local[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
  compute_local[(31)] = (compute_local[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 32))]));
  compute_local[(45)] = (compute_local[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 48))]));
  compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute_local[(18)] = (compute_local[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
  compute_local[(32)] = (compute_local[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 32))]));
  compute_local[(46)] = (compute_local[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 48))]));
  compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute_local[(19)] = (compute_local[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
  compute_local[(33)] = (compute_local[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 32))]));
  compute_local[(47)] = (compute_local[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 48))]));
  compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[((((int)threadIdx.z) * 2))]));
  compute_local[(20)] = (compute_local[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 16))]));
  compute_local[(34)] = (compute_local[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 32))]));
  compute_local[(48)] = (compute_local[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 48))]));
  compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute_local[(21)] = (compute_local[(21)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
  compute_local[(35)] = (compute_local[(35)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 33))]));
  compute_local[(49)] = (compute_local[(49)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 49))]));
  compute_local[(8)] = (compute_local[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute_local[(22)] = (compute_local[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
  compute_local[(36)] = (compute_local[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 33))]));
  compute_local[(50)] = (compute_local[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 49))]));
  compute_local[(9)] = (compute_local[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute_local[(23)] = (compute_local[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
  compute_local[(37)] = (compute_local[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 33))]));
  compute_local[(51)] = (compute_local[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 49))]));
  compute_local[(10)] = (compute_local[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute_local[(24)] = (compute_local[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
  compute_local[(38)] = (compute_local[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 33))]));
  compute_local[(52)] = (compute_local[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 49))]));
  compute_local[(11)] = (compute_local[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute_local[(25)] = (compute_local[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
  compute_local[(39)] = (compute_local[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 33))]));
  compute_local[(53)] = (compute_local[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 49))]));
  compute_local[(12)] = (compute_local[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute_local[(26)] = (compute_local[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
  compute_local[(40)] = (compute_local[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 33))]));
  compute_local[(54)] = (compute_local[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 49))]));
  compute_local[(13)] = (compute_local[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 1))]));
  compute_local[(27)] = (compute_local[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 17))]));
  compute_local[(41)] = (compute_local[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 33))]));
  compute_local[(55)] = (compute_local[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 2) + 49))]));
  compute[(((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3136))] = compute_local[(14)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6272))] = compute_local[(28)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9408))] = compute_local[(42)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 14))] = compute_local[(1)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3150))] = compute_local[(15)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6286))] = compute_local[(29)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9422))] = compute_local[(43)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 28))] = compute_local[(2)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3164))] = compute_local[(16)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6300))] = compute_local[(30)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9436))] = compute_local[(44)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 42))] = compute_local[(3)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3178))] = compute_local[(17)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6314))] = compute_local[(31)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9450))] = compute_local[(45)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 56))] = compute_local[(4)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3192))] = compute_local[(18)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6328))] = compute_local[(32)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9464))] = compute_local[(46)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 70))] = compute_local[(5)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3206))] = compute_local[(19)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6342))] = compute_local[(33)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9478))] = compute_local[(47)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 84))] = compute_local[(6)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3220))] = compute_local[(20)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6356))] = compute_local[(34)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9492))] = compute_local[(48)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 196))] = compute_local[(7)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3332))] = compute_local[(21)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6468))] = compute_local[(35)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9604))] = compute_local[(49)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 210))] = compute_local[(8)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3346))] = compute_local[(22)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6482))] = compute_local[(36)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9618))] = compute_local[(50)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 224))] = compute_local[(9)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3360))] = compute_local[(23)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6496))] = compute_local[(37)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9632))] = compute_local[(51)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 238))] = compute_local[(10)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3374))] = compute_local[(24)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6510))] = compute_local[(38)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9646))] = compute_local[(52)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 252))] = compute_local[(11)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3388))] = compute_local[(25)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6524))] = compute_local[(39)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9660))] = compute_local[(53)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 266))] = compute_local[(12)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3402))] = compute_local[(26)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6538))] = compute_local[(40)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9674))] = compute_local[(54)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 280))] = compute_local[(13)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 3416))] = compute_local[(27)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 6552))] = compute_local[(41)];
  compute[((((((((int)blockIdx.z) * 12544) + (((int)threadIdx.z) * 392)) + (((int)blockIdx.y) * 98)) + ((int)threadIdx.x)) + 9688))] = compute_local[(55)];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_6_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[2];
  __shared__ float pad_temp_shared[30];
  __shared__ float placeholder_shared[3];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 10) {
      pad_temp_shared[((((int)threadIdx.x) * 3))] = (((1 <= ((int)blockIdx.y)) && (1 <= ((int)threadIdx.x))) ? placeholder[(((((rc_outer * 784) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 3)) - 29))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 10) {
      pad_temp_shared[(((((int)threadIdx.x) * 3) + 1))] = ((1 <= ((int)blockIdx.y)) ? placeholder[(((((rc_outer * 784) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 3)) - 28))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 10) {
      pad_temp_shared[(((((int)threadIdx.x) * 3) + 2))] = (((1 <= ((int)blockIdx.y)) && (((int)threadIdx.x) < 9)) ? placeholder[(((((rc_outer * 784) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 3)) - 27))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[(((rc_outer * 9) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 15))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 16))] * placeholder_shared[(2)]));
    __syncthreads();
    if (((int)threadIdx.x) < 10) {
      pad_temp_shared[((((int)threadIdx.x) * 3))] = ((1 <= ((int)threadIdx.x)) ? placeholder[(((((rc_outer * 784) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 3)) - 1))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 10) {
      pad_temp_shared[(((((int)threadIdx.x) * 3) + 1))] = placeholder[((((rc_outer * 784) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 3)))];
    }
    if (((int)threadIdx.x) < 10) {
      pad_temp_shared[(((((int)threadIdx.x) * 3) + 2))] = ((((int)threadIdx.x) < 9) ? placeholder[(((((rc_outer * 784) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 3)) + 1))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((rc_outer * 9) + ((int)threadIdx.x)) + 3))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 15))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 16))] * placeholder_shared[(2)]));
    __syncthreads();
    if (((int)threadIdx.x) < 10) {
      pad_temp_shared[((((int)threadIdx.x) * 3))] = (((((int)blockIdx.y) < 27) && (1 <= ((int)threadIdx.x))) ? placeholder[(((((rc_outer * 784) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 3)) + 27))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 10) {
      pad_temp_shared[(((((int)threadIdx.x) * 3) + 1))] = ((((int)blockIdx.y) < 27) ? placeholder[(((((rc_outer * 784) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 3)) + 28))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 10) {
      pad_temp_shared[(((((int)threadIdx.x) * 3) + 2))] = (((((int)blockIdx.y) < 27) && (((int)threadIdx.x) < 9)) ? placeholder[(((((rc_outer * 784) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.x) * 3)) + 29))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((rc_outer * 9) + ((int)threadIdx.x)) + 6))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 15))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 16))] * placeholder_shared[(2)]));
  }
  T_relu[(((((int)blockIdx.y) * 28) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(0)]), 0.000000e+00f);
  T_relu[((((((int)blockIdx.y) * 28) + ((int)threadIdx.x)) + 14))] = max((compute[(1)] + placeholder2[(0)]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_2_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[8];
  __shared__ float pad_temp_shared[56];
  __shared__ float placeholder_shared[2];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(2)] = 0.000000e+00f;
  compute_local[(4)] = 0.000000e+00f;
  compute_local[(6)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  compute_local[(3)] = 0.000000e+00f;
  compute_local[(5)] = 0.000000e+00f;
  compute_local[(7)] = 0.000000e+00f;
  pad_temp_shared[((((int)threadIdx.x) * 4))] = placeholder[(((((int)blockIdx.y) * 56) + (((int)threadIdx.x) * 4)))];
  pad_temp_shared[(((((int)threadIdx.x) * 4) + 1))] = placeholder[((((((int)blockIdx.y) * 56) + (((int)threadIdx.x) * 4)) + 1))];
  pad_temp_shared[(((((int)threadIdx.x) * 4) + 2))] = placeholder[((((((int)blockIdx.y) * 56) + (((int)threadIdx.x) * 4)) + 2))];
  pad_temp_shared[(((((int)threadIdx.x) * 4) + 3))] = placeholder[((((((int)blockIdx.y) * 56) + (((int)threadIdx.x) * 4)) + 3))];
  if (((int)threadIdx.x) < 2) {
    placeholder_shared[(((int)threadIdx.x))] = placeholder1[(((((int)blockIdx.z) * 2) + ((int)threadIdx.x)))];
  }
  __syncthreads();
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
  compute_local[(2)] = (compute_local[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(0)]));
  compute_local[(4)] = (compute_local[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(0)]));
  compute_local[(6)] = (compute_local[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(0)]));
  compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(1)]));
  compute_local[(3)] = (compute_local[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(1)]));
  compute_local[(5)] = (compute_local[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(1)]));
  compute_local[(7)] = (compute_local[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(1)]));
  compute[((((((int)blockIdx.z) * 6272) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((int)blockIdx.z) * 6272) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 14))] = compute_local[(2)];
  compute[(((((((int)blockIdx.z) * 6272) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 28))] = compute_local[(4)];
  compute[(((((((int)blockIdx.z) * 6272) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 42))] = compute_local[(6)];
  compute[(((((((int)blockIdx.z) * 6272) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 3136))] = compute_local[(1)];
  compute[(((((((int)blockIdx.z) * 6272) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 3150))] = compute_local[(3)];
  compute[(((((((int)blockIdx.z) * 6272) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 3164))] = compute_local[(5)];
  compute[(((((((int)blockIdx.z) * 6272) + (((int)blockIdx.y) * 56)) + ((int)threadIdx.x)) + 3178))] = compute_local[(7)];
}

extern "C" __global__ void fused_nn_dense_add_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float T_dense_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float T_dense[1];
  T_dense_rf[(0)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 1) {
    T_dense_rf[(0)] = (T_dense_rf[(0)] + (placeholder[(((int)threadIdx.x))] * placeholder1[((((int)blockIdx.x) + ((int)threadIdx.x)))]));
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

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_5_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[8];
  __shared__ float pad_temp_shared[28];
  __shared__ float placeholder_shared[32];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 2; ++yy_init) {
      compute[(((ff_init * 2) + yy_init))] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 257; ++rc_outer) {
    __syncthreads();
    if (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) < 28) {
      if (((int)threadIdx.x) < 4) {
        pad_temp_shared[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))] = placeholder[(((((rc_outer * 196) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 4)) + ((int)threadIdx.x)))];
      }
    }
    if (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) < 32) {
      if (((int)threadIdx.x) < 4) {
        placeholder_shared[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))] = placeholder1[((((((int)threadIdx.z) * 1028) + (((int)threadIdx.x) * 257)) + rc_outer))];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int ff = 0; ff < 4; ++ff) {
      #pragma unroll
      for (int yy = 0; yy < 2; ++yy) {
        compute[(((ff * 2) + yy))] = (compute[(((ff * 2) + yy))] + (pad_temp_shared[(((yy * 14) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 4) + ff))]));
      }
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 4; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
      T_relu[((((((((int)threadIdx.z) * 784) + (ax1_inner_inner_inner * 196)) + (((int)blockIdx.y) * 28)) + (ax2_inner_inner_inner * 14)) + ((int)threadIdx.x)))] = max((compute[(((ax1_inner_inner_inner * 2) + ax2_inner_inner_inner))] + placeholder2[(((((int)threadIdx.z) * 4) + ax1_inner_inner_inner))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_9_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[58];
  __shared__ float placeholder_shared[3];
  compute[(0)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[((((int)threadIdx.x) * 5))] = (((1 <= ((int)blockIdx.y)) && (1 <= ((int)threadIdx.x))) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) - 57))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 1))] = ((1 <= ((int)blockIdx.y)) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) - 56))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 2))] = (((1 <= ((int)blockIdx.y)) && (((int)threadIdx.x) < 11)) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) - 55))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 11) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 3))] = ((1 <= ((int)blockIdx.y)) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) - 54))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 11) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 4))] = ((1 <= ((int)blockIdx.y)) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) - 53))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[(((rc_outer * 9) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(0)]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(0)]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 15))] * placeholder_shared[(1)]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 29))] * placeholder_shared[(1)]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 43))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 16))] * placeholder_shared[(2)]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 30))] * placeholder_shared[(2)]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[(2)]));
    __syncthreads();
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[((((int)threadIdx.x) * 5))] = ((1 <= ((int)threadIdx.x)) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) - 1))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 1))] = placeholder[((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)))];
    }
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 2))] = ((((int)threadIdx.x) < 11) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) + 1))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 11) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 3))] = placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) + 2))];
    }
    if (((int)threadIdx.x) < 11) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 4))] = placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) + 3))];
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((rc_outer * 9) + ((int)threadIdx.x)) + 3))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(0)]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(0)]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 15))] * placeholder_shared[(1)]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 29))] * placeholder_shared[(1)]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 43))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 16))] * placeholder_shared[(2)]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 30))] * placeholder_shared[(2)]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[(2)]));
    __syncthreads();
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[((((int)threadIdx.x) * 5))] = (((((int)blockIdx.y) < 55) && (1 <= ((int)threadIdx.x))) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) + 55))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 1))] = ((((int)blockIdx.y) < 55) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) + 56))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 12) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 2))] = (((((int)blockIdx.y) < 55) && (((int)threadIdx.x) < 11)) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) + 57))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 11) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 3))] = ((((int)blockIdx.y) < 55) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) + 58))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 11) {
      pad_temp_shared[(((((int)threadIdx.x) * 5) + 4))] = ((((int)blockIdx.y) < 55) ? placeholder[(((((rc_outer * 3136) + (((int)blockIdx.y) * 56)) + (((int)threadIdx.x) * 5)) + 59))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((rc_outer * 9) + ((int)threadIdx.x)) + 6))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(0)]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(0)]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 15))] * placeholder_shared[(1)]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 29))] * placeholder_shared[(1)]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 43))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 16))] * placeholder_shared[(2)]));
    compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 30))] * placeholder_shared[(2)]));
    compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 44))] * placeholder_shared[(2)]));
  }
  T_relu[(((((int)blockIdx.y) * 56) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(0)]), 0.000000e+00f);
  T_relu[((((((int)blockIdx.y) * 56) + ((int)threadIdx.x)) + 14))] = max((compute[(1)] + placeholder2[(0)]), 0.000000e+00f);
  T_relu[((((((int)blockIdx.y) * 56) + ((int)threadIdx.x)) + 28))] = max((compute[(2)] + placeholder2[(0)]), 0.000000e+00f);
  T_relu[((((((int)blockIdx.y) * 56) + ((int)threadIdx.x)) + 42))] = max((compute[(3)] + placeholder2[(0)]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_conv2d_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ compute) {
  float compute_local[2];
  __shared__ float pad_temp_shared[28];
  __shared__ float placeholder_shared[1];
  compute_local[(0)] = 0.000000e+00f;
  compute_local[(1)] = 0.000000e+00f;
  pad_temp_shared[((((int)threadIdx.x) * 2))] = placeholder[(((((int)blockIdx.y) * 28) + (((int)threadIdx.x) * 2)))];
  pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] = placeholder[((((((int)blockIdx.y) * 28) + (((int)threadIdx.x) * 2)) + 1))];
  if (((int)threadIdx.x) < 1) {
    placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((int)threadIdx.x) + ((int)blockIdx.z)))];
  }
  __syncthreads();
  compute_local[(0)] = (compute_local[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
  compute_local[(1)] = (compute_local[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(0)]));
  compute[((((((int)blockIdx.z) * 784) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)))] = compute_local[(0)];
  compute[(((((((int)blockIdx.z) * 784) + (((int)blockIdx.y) * 28)) + ((int)threadIdx.x)) + 14))] = compute_local[(1)];
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_1_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[4];
  __shared__ float pad_temp_shared[7];
  __shared__ float placeholder_shared[32];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    compute[(ff_init)] = 0.000000e+00f;
  }
  if ((((int)threadIdx.x) + ((int)threadIdx.z)) < 7) {
    if (((int)threadIdx.x) < 1) {
      pad_temp_shared[((((int)threadIdx.x) + ((int)threadIdx.z)))] = placeholder[((((((int)blockIdx.y) * 7) + ((int)threadIdx.x)) + ((int)threadIdx.z)))];
    }
  }
  if (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) < 32) {
    if (((int)threadIdx.x) < 4) {
      placeholder_shared[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))] = placeholder1[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ff = 0; ff < 4; ++ff) {
    compute[(ff)] = (compute[(ff)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 4) + ff))]));
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 4; ++ax1_inner_inner_inner) {
    T_relu[(((((((int)threadIdx.z) * 196) + (ax1_inner_inner_inner * 49)) + (((int)blockIdx.y) * 7)) + ((int)threadIdx.x)))] = max((compute[(ax1_inner_inner_inner)] + placeholder2[(((((int)threadIdx.z) * 4) + ax1_inner_inner_inner))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_7_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[16];
  __shared__ float pad_temp_shared[56];
  __shared__ float placeholder_shared[32];
  #pragma unroll
  for (int ff_init = 0; ff_init < 4; ++ff_init) {
    #pragma unroll
    for (int yy_init = 0; yy_init < 4; ++yy_init) {
      compute[(((ff_init * 4) + yy_init))] = 0.000000e+00f;
    }
  }
  if (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) < 56) {
    if (((int)threadIdx.x) < 7) {
      pad_temp_shared[(((((int)threadIdx.z) * 7) + ((int)threadIdx.x)))] = placeholder[(((((((int)blockIdx.y) * 112) + ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 14) * 28)) + (((int)blockIdx.x) * 14)) + (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) % 14)))];
    }
  }
  if (((((int)threadIdx.z) * 4) + ((int)threadIdx.x)) < 32) {
    if (((int)threadIdx.x) < 4) {
      placeholder_shared[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))] = placeholder1[(((((int)threadIdx.z) * 4) + ((int)threadIdx.x)))];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ff = 0; ff < 4; ++ff) {
    #pragma unroll
    for (int yy = 0; yy < 4; ++yy) {
      compute[(((ff * 4) + yy))] = (compute[(((ff * 4) + yy))] + (pad_temp_shared[(((yy * 14) + ((int)threadIdx.x)))] * placeholder_shared[(((((int)threadIdx.z) * 4) + ff))]));
    }
  }
  #pragma unroll
  for (int ax1_inner_inner_inner = 0; ax1_inner_inner_inner < 4; ++ax1_inner_inner_inner) {
    #pragma unroll
    for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 4; ++ax2_inner_inner_inner) {
      T_relu[(((((((((int)threadIdx.z) * 3136) + (ax1_inner_inner_inner * 784)) + (((int)blockIdx.y) * 112)) + (ax2_inner_inner_inner * 28)) + (((int)blockIdx.x) * 14)) + ((int)threadIdx.x)))] = max((compute[(((ax1_inner_inner_inner * 4) + ax2_inner_inner_inner))] + placeholder2[(((((int)threadIdx.z) * 4) + ax1_inner_inner_inner))]), 0.000000e+00f);
    }
  }
}

extern "C" __global__ void fused_nn_max_pool2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
  float tensor[1];
  tensor[(0)] = -3.402823e+38f;
  for (int dh = 0; dh < 3; ++dh) {
    for (int dw = 0; dw < 3; ++dw) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 216384) {
        tensor[(0)] = max(tensor[(0)], (((1 <= ((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 3136) / 56) * 2) + dh)) && (1 <= (((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 56) * 2) + dw))) ? placeholder[(((((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 56) * 224) + (dh * 112)) + ((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 56) * 2)) + dw) - 113))] : -3.402823e+38f));
      }
    }
  }
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 216384) {
    T_relu[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = max((tensor[(0)] + placeholder1[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 3136))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_avg_pool2d_add_nn_relu_2_kernel0(float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
  float tensor[1];
  tensor[(0)] = 0.000000e+00f;
  for (int dh = 0; dh < 2; ++dh) {
    for (int dw = 0; dw < 2; ++dw) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 101920) {
        tensor[(0)] = (tensor[(0)] + placeholder[((((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 28) * 112) + (dh * 56)) + ((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 28) * 2)) + dw))]);
      }
    }
  }
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 101920) {
    T_relu[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = max(((tensor[(0)] * 2.500000e-01f) + placeholder1[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 784))]), 0.000000e+00f);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_3_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
  float compute[1];
  __shared__ float pad_temp_shared[16];
  __shared__ float placeholder_shared[3];
  compute[(0)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 8) {
      pad_temp_shared[((((int)threadIdx.x) * 2))] = (((1 <= ((int)blockIdx.y)) && (1 <= ((int)threadIdx.x))) ? placeholder[(((((rc_outer * 196) + (((int)blockIdx.y) * 14)) + (((int)threadIdx.x) * 2)) - 15))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 8) {
      pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] = (((1 <= ((int)blockIdx.y)) && (((int)threadIdx.x) < 7)) ? placeholder[(((((rc_outer * 196) + (((int)blockIdx.y) * 14)) + (((int)threadIdx.x) * 2)) - 14))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[(((rc_outer * 9) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    __syncthreads();
    if (((int)threadIdx.x) < 8) {
      pad_temp_shared[((((int)threadIdx.x) * 2))] = ((1 <= ((int)threadIdx.x)) ? placeholder[(((((rc_outer * 196) + (((int)blockIdx.y) * 14)) + (((int)threadIdx.x) * 2)) - 1))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 8) {
      pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] = ((((int)threadIdx.x) < 7) ? placeholder[((((rc_outer * 196) + (((int)blockIdx.y) * 14)) + (((int)threadIdx.x) * 2)))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((rc_outer * 9) + ((int)threadIdx.x)) + 3))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
    __syncthreads();
    if (((int)threadIdx.x) < 8) {
      pad_temp_shared[((((int)threadIdx.x) * 2))] = (((((int)blockIdx.y) < 13) && (1 <= ((int)threadIdx.x))) ? placeholder[(((((rc_outer * 196) + (((int)blockIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 13))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 8) {
      pad_temp_shared[(((((int)threadIdx.x) * 2) + 1))] = (((((int)blockIdx.y) < 13) && (((int)threadIdx.x) < 7)) ? placeholder[(((((rc_outer * 196) + (((int)blockIdx.y) * 14)) + (((int)threadIdx.x) * 2)) + 14))] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 3) {
      placeholder_shared[(((int)threadIdx.x))] = placeholder1[((((rc_outer * 9) + ((int)threadIdx.x)) + 6))];
    }
    __syncthreads();
    compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(0)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 1))] * placeholder_shared[(1)]));
    compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 2))] * placeholder_shared[(2)]));
  }
  T_relu[(((((int)blockIdx.y) * 14) + ((int)threadIdx.x)))] = max((compute[(0)] + placeholder2[(0)]), 0.000000e+00f);
}

extern "C" __global__ void fused_nn_avg_pool2d_add_nn_relu_kernel0(float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
  float tensor[1];
  tensor[(0)] = 0.000000e+00f;
  for (int dh = 0; dh < 2; ++dh) {
    for (int dw = 0; dw < 2; ++dw) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 31360) {
        tensor[(0)] = (tensor[(0)] + placeholder[((((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 7) * 28) + (dh * 14)) + ((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 7) * 2)) + dw))]);
      }
    }
  }
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 31360) {
    T_relu[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)))] = max(((tensor[(0)] * 2.500000e-01f) + placeholder1[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 49))]), 0.000000e+00f);
  }
}

