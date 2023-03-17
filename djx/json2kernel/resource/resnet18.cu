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
    float compute[56];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(28)] = 0.000000e+00f;
  compute[(35)] = 0.000000e+00f;
  compute[(42)] = 0.000000e+00f;
  compute[(49)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(29)] = 0.000000e+00f;
  compute[(36)] = 0.000000e+00f;
  compute[(43)] = 0.000000e+00f;
  compute[(50)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(30)] = 0.000000e+00f;
  compute[(37)] = 0.000000e+00f;
  compute[(44)] = 0.000000e+00f;
  compute[(51)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(31)] = 0.000000e+00f;
  compute[(38)] = 0.000000e+00f;
  compute[(45)] = 0.000000e+00f;
  compute[(52)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(32)] = 0.000000e+00f;
  compute[(39)] = 0.000000e+00f;
  compute[(46)] = 0.000000e+00f;
  compute[(53)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(33)] = 0.000000e+00f;
  compute[(40)] = 0.000000e+00f;
  compute[(47)] = 0.000000e+00f;
  compute[(54)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  compute[(34)] = 0.000000e+00f;
  compute[(41)] = 0.000000e+00f;
  compute[(48)] = 0.000000e+00f;
  compute[(55)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)))] = ((((1 <= (((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer)) && ((((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer) < 15)) && (1 <= (((int)threadIdx.x) & 1))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 1))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 1) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 2))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 2) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 3))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 3) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 12))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 4))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 4) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 11))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 5))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 5) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 10))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 6))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 6) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 9))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 5) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 512) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 5) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 5) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 1) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 511) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 1))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 1) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 1) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 2) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 510) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 2))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 2) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 2) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 3) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 509) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 3))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 3) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 3) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 4) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 508) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 4))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 4) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 4) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)))] = (((1 <= (((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer)) && ((((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 1))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 2))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 12))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 3))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 11))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 4))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 10))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 5))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 9))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 6))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 8))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 5) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 512) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 5) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 5) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 1) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 511) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 1))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 1) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 1) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 2) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 510) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 2))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 2) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 2) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 3) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 509) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 3))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 3) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 3) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 4) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 508) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 4))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 4) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 4) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)))] = (((1 <= (((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer)) && ((((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 1))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 1) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 12))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 2))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 2) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 11))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 3))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 3) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 10))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 4))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 4) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 9))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 5))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 5) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 8))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 6))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 6) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 7))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 5) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 512) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 5) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 5) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 1) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 511) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 1))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 1) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 1) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 2) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 510) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 2))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 2) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 2) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 3) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 509) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 3))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 3) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 3) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 4) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 508) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 4))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 4) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 4) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
    }
  }
  T_add[(((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)))] = (compute[(0)] + placeholder2[(((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1568))] = (compute[(7)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1568))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3136))] = (compute[(14)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3136))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4704))] = (compute[(21)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4704))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6272))] = (compute[(28)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6272))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7840))] = (compute[(35)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7840))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9408))] = (compute[(42)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9408))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 10976))] = (compute[(49)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 10976))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 14))] = (compute[(1)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 14))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1582))] = (compute[(8)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1582))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3150))] = (compute[(15)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3150))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4718))] = (compute[(22)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4718))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6286))] = (compute[(29)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6286))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7854))] = (compute[(36)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7854))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9422))] = (compute[(43)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9422))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 10990))] = (compute[(50)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 10990))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 28))] = (compute[(2)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 28))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1596))] = (compute[(9)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1596))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3164))] = (compute[(16)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3164))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4732))] = (compute[(23)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4732))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6300))] = (compute[(30)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6300))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7868))] = (compute[(37)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7868))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9436))] = (compute[(44)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9436))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11004))] = (compute[(51)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11004))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 42))] = (compute[(3)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 42))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1610))] = (compute[(10)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1610))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3178))] = (compute[(17)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3178))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4746))] = (compute[(24)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4746))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6314))] = (compute[(31)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6314))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7882))] = (compute[(38)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7882))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9450))] = (compute[(45)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9450))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11018))] = (compute[(52)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11018))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 56))] = (compute[(4)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 56))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1624))] = (compute[(11)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1624))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3192))] = (compute[(18)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3192))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4760))] = (compute[(25)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4760))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6328))] = (compute[(32)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6328))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7896))] = (compute[(39)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7896))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9464))] = (compute[(46)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9464))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11032))] = (compute[(53)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11032))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 70))] = (compute[(5)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 70))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1638))] = (compute[(12)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1638))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3206))] = (compute[(19)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3206))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4774))] = (compute[(26)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4774))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6342))] = (compute[(33)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6342))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7910))] = (compute[(40)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7910))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9478))] = (compute[(47)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9478))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11046))] = (compute[(54)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11046))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 84))] = (compute[(6)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 84))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1652))] = (compute[(13)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1652))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3220))] = (compute[(20)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3220))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4788))] = (compute[(27)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4788))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6356))] = (compute[(34)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6356))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7924))] = (compute[(41)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7924))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9492))] = (compute[(48)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9492))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11060))] = (compute[(55)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11060))]);
    offset+=blocksize[0];
    }
  }
  else
  {
  float compute[56];
  __shared__ float pad_temp_shared[784];
  __shared__ float placeholder_shared[512];
  compute[(0)] = 0.000000e+00f;
  compute[(7)] = 0.000000e+00f;
  compute[(14)] = 0.000000e+00f;
  compute[(21)] = 0.000000e+00f;
  compute[(28)] = 0.000000e+00f;
  compute[(35)] = 0.000000e+00f;
  compute[(42)] = 0.000000e+00f;
  compute[(49)] = 0.000000e+00f;
  compute[(1)] = 0.000000e+00f;
  compute[(8)] = 0.000000e+00f;
  compute[(15)] = 0.000000e+00f;
  compute[(22)] = 0.000000e+00f;
  compute[(29)] = 0.000000e+00f;
  compute[(36)] = 0.000000e+00f;
  compute[(43)] = 0.000000e+00f;
  compute[(50)] = 0.000000e+00f;
  compute[(2)] = 0.000000e+00f;
  compute[(9)] = 0.000000e+00f;
  compute[(16)] = 0.000000e+00f;
  compute[(23)] = 0.000000e+00f;
  compute[(30)] = 0.000000e+00f;
  compute[(37)] = 0.000000e+00f;
  compute[(44)] = 0.000000e+00f;
  compute[(51)] = 0.000000e+00f;
  compute[(3)] = 0.000000e+00f;
  compute[(10)] = 0.000000e+00f;
  compute[(17)] = 0.000000e+00f;
  compute[(24)] = 0.000000e+00f;
  compute[(31)] = 0.000000e+00f;
  compute[(38)] = 0.000000e+00f;
  compute[(45)] = 0.000000e+00f;
  compute[(52)] = 0.000000e+00f;
  compute[(4)] = 0.000000e+00f;
  compute[(11)] = 0.000000e+00f;
  compute[(18)] = 0.000000e+00f;
  compute[(25)] = 0.000000e+00f;
  compute[(32)] = 0.000000e+00f;
  compute[(39)] = 0.000000e+00f;
  compute[(46)] = 0.000000e+00f;
  compute[(53)] = 0.000000e+00f;
  compute[(5)] = 0.000000e+00f;
  compute[(12)] = 0.000000e+00f;
  compute[(19)] = 0.000000e+00f;
  compute[(26)] = 0.000000e+00f;
  compute[(33)] = 0.000000e+00f;
  compute[(40)] = 0.000000e+00f;
  compute[(47)] = 0.000000e+00f;
  compute[(54)] = 0.000000e+00f;
  compute[(6)] = 0.000000e+00f;
  compute[(13)] = 0.000000e+00f;
  compute[(20)] = 0.000000e+00f;
  compute[(27)] = 0.000000e+00f;
  compute[(34)] = 0.000000e+00f;
  compute[(41)] = 0.000000e+00f;
  compute[(48)] = 0.000000e+00f;
  compute[(55)] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)))] = ((((1 <= (((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer)) && ((((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer) < 15)) && (1 <= (((int)threadIdx.x) & 1))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 15))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 1))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 1) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 2))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 2) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 3))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 3) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 12))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 4))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 4) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 11))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 5))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 5) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 10))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 6))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer) < 15)) && (1 <= (((((int)threadIdx.x) * 7) + 6) % 14))) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 9))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 5) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 512) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 5) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 5) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 1) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 511) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 1))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 1) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 1) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 2) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 510) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 2))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 2) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 2) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 3) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 509) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 3))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 3) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 3) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 4) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 508) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 4))] = placeholder1[((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 4) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 4) & 7) * 9)) + (ry_outer * 3)))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)))] = (((1 <= (((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer)) && ((((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 14))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 1))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 2))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 12))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 3))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 11))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 4))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 10))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 5))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 9))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 6))] = (((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 8))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 5) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 512) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 5) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 5) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 1) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 511) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 1))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 1) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 1) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 2) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 510) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 2))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 2) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 2) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 3) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 509) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 3))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 3) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 3) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 4) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 508) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 4))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 4) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 4) & 7) * 9)) + (ry_outer * 3)) + 1))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      __syncthreads();
      pad_temp_shared[(((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)))] = (((1 <= (((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer)) && ((((((int)vy) * 7) + (((int)threadIdx.x) >> 1)) + ry_outer) < 15)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 13))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 1))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 1) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 1) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 12))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 2))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 2) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 2) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 11))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 3))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 3) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 3) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 10))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 4))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 4) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 4) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 9))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 5))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 5) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 5) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 8))] : 0.000000e+00f);
      pad_temp_shared[((((((int)threadIdx.z) * 98) + (((int)threadIdx.x) * 7)) + 6))] = ((((1 <= (((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer)) && ((((((int)vy) * 7) + (((((int)threadIdx.x) * 7) + 6) / 14)) + ry_outer) < 15)) && ((((((int)threadIdx.x) * 7) + 6) % 14) < 13)) ? placeholder[(((((((((((int)vz) >> 2) * 50176) + (rc_outer * 1568)) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + (ry_outer * 14)) + (((int)threadIdx.x) * 7)) - 7))] : 0.000000e+00f);
      if (((((int)threadIdx.z) * 8) + ((((int)threadIdx.x) * 5) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 512) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[(((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + (((((int)threadIdx.x) * 5) >> 3) * 2304)) + (rc_outer * 72)) + (((((int)threadIdx.x) * 5) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 1) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 511) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 1))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 1) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 1) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 2) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 510) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 2))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 2) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 2) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 3) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 509) {
          if (((int)threadIdx.x) < 13) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 3))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 3) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 3) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      if (((((int)threadIdx.z) * 8) + (((((int)threadIdx.x) * 5) + 4) >> 3)) < 64) {
        if (((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) < 508) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[((((((int)threadIdx.z) * 64) + (((int)threadIdx.x) * 5)) + 4))] = placeholder1[(((((((((((int)vz) & 3) * 147456) + (((int)threadIdx.z) * 18432)) + ((((((int)threadIdx.x) * 5) + 4) >> 3) * 2304)) + (rc_outer * 72)) + ((((((int)threadIdx.x) * 5) + 4) & 7) * 9)) + (ry_outer * 3)) + 2))];
          }
        }
      }
      __syncthreads();
      compute[(0)] = (compute[(0)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[(((int)threadIdx.x))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 14))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 28))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 42))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 56))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 70))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[((((int)threadIdx.z) * 8))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 64))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 128))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 192))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 256))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 320))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 384))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 84))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 448))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 98))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 112))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 126))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 140))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 154))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 168))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 1))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 65))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 129))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 193))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 257))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 321))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 385))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 182))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 449))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 196))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 210))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 224))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 238))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 252))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 266))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 2))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 66))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 130))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 194))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 258))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 322))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 386))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 280))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 450))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 294))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 308))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 322))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 336))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 350))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 364))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 3))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 67))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 131))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 195))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 259))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 323))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 387))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 378))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 451))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 392))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 406))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 420))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 434))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 448))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 462))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 4))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 68))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 132))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 196))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 260))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 324))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 388))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 476))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 452))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 490))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 504))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 518))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 532))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 546))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 560))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 5))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 69))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 133))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 197))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 261))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 325))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 389))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 574))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 453))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 588))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 602))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 616))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 630))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 644))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 658))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 6))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 70))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 134))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 198))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 262))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 326))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 390))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 672))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 454))]));
      compute[(0)] = (compute[(0)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(7)] = (compute[(7)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(14)] = (compute[(14)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(21)] = (compute[(21)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(28)] = (compute[(28)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(35)] = (compute[(35)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(42)] = (compute[(42)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(49)] = (compute[(49)] + (pad_temp_shared[((((int)threadIdx.x) + 686))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(1)] = (compute[(1)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(8)] = (compute[(8)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(15)] = (compute[(15)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(22)] = (compute[(22)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(29)] = (compute[(29)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(36)] = (compute[(36)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(43)] = (compute[(43)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(50)] = (compute[(50)] + (pad_temp_shared[((((int)threadIdx.x) + 700))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(2)] = (compute[(2)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(9)] = (compute[(9)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(16)] = (compute[(16)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(23)] = (compute[(23)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(30)] = (compute[(30)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(37)] = (compute[(37)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(44)] = (compute[(44)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(51)] = (compute[(51)] + (pad_temp_shared[((((int)threadIdx.x) + 714))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(3)] = (compute[(3)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(10)] = (compute[(10)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(17)] = (compute[(17)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(24)] = (compute[(24)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(31)] = (compute[(31)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(38)] = (compute[(38)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(45)] = (compute[(45)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(52)] = (compute[(52)] + (pad_temp_shared[((((int)threadIdx.x) + 728))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(4)] = (compute[(4)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(11)] = (compute[(11)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(18)] = (compute[(18)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(25)] = (compute[(25)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(32)] = (compute[(32)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(39)] = (compute[(39)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(46)] = (compute[(46)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(53)] = (compute[(53)] + (pad_temp_shared[((((int)threadIdx.x) + 742))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(5)] = (compute[(5)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(12)] = (compute[(12)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(19)] = (compute[(19)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(26)] = (compute[(26)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(33)] = (compute[(33)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(40)] = (compute[(40)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(47)] = (compute[(47)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(54)] = (compute[(54)] + (pad_temp_shared[((((int)threadIdx.x) + 756))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
      compute[(6)] = (compute[(6)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 7))]));
      compute[(13)] = (compute[(13)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 71))]));
      compute[(20)] = (compute[(20)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 135))]));
      compute[(27)] = (compute[(27)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 199))]));
      compute[(34)] = (compute[(34)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 263))]));
      compute[(41)] = (compute[(41)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 327))]));
      compute[(48)] = (compute[(48)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 391))]));
      compute[(55)] = (compute[(55)] + (pad_temp_shared[((((int)threadIdx.x) + 770))] * placeholder_shared[(((((int)threadIdx.z) * 8) + 455))]));
    }
  }
  T_add[(((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)))] = (compute[(0)] + placeholder2[(((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1568))] = (compute[(7)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1568))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3136))] = (compute[(14)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3136))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4704))] = (compute[(21)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4704))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6272))] = (compute[(28)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6272))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7840))] = (compute[(35)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7840))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9408))] = (compute[(42)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9408))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 10976))] = (compute[(49)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 10976))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 14))] = (compute[(1)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 14))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1582))] = (compute[(8)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1582))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3150))] = (compute[(15)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3150))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4718))] = (compute[(22)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4718))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6286))] = (compute[(29)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6286))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7854))] = (compute[(36)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7854))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9422))] = (compute[(43)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9422))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 10990))] = (compute[(50)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 10990))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 28))] = (compute[(2)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 28))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1596))] = (compute[(9)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1596))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3164))] = (compute[(16)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3164))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4732))] = (compute[(23)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4732))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6300))] = (compute[(30)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6300))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7868))] = (compute[(37)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7868))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9436))] = (compute[(44)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9436))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11004))] = (compute[(51)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11004))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 42))] = (compute[(3)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 42))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1610))] = (compute[(10)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1610))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3178))] = (compute[(17)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3178))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4746))] = (compute[(24)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4746))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6314))] = (compute[(31)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6314))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7882))] = (compute[(38)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7882))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9450))] = (compute[(45)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9450))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11018))] = (compute[(52)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11018))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 56))] = (compute[(4)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 56))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1624))] = (compute[(11)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1624))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3192))] = (compute[(18)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3192))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4760))] = (compute[(25)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4760))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6328))] = (compute[(32)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6328))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7896))] = (compute[(39)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7896))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9464))] = (compute[(46)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9464))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11032))] = (compute[(53)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11032))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 70))] = (compute[(5)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 70))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1638))] = (compute[(12)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1638))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3206))] = (compute[(19)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3206))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4774))] = (compute[(26)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4774))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6342))] = (compute[(33)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6342))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7910))] = (compute[(40)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7910))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9478))] = (compute[(47)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9478))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11046))] = (compute[(54)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11046))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 84))] = (compute[(6)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 84))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1652))] = (compute[(13)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 1652))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3220))] = (compute[(20)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 3220))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4788))] = (compute[(27)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 4788))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6356))] = (compute[(34)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 6356))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7924))] = (compute[(41)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 7924))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9492))] = (compute[(48)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 9492))]);
  T_add[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11060))] = (compute[(55)] + placeholder2[((((((((int)vz) * 12544) + (((int)threadIdx.z) * 196)) + (((int)vy) * 98)) + ((int)threadIdx.x)) + 11060))]);
  }
}

extern "C" __global__ void fused_nn_conv2d_add_nn_relu_7_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_relu, float* __restrict__ placeholder2) {
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
    float compute[16];
  __shared__ float pad_temp_shared[1155];
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
  for (int ry_outer = 0; ry_outer < 7; ++ry_outer) {
    for (int rx_outer = 0; rx_outer < 7; ++rx_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
        if ((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 1155) {
          if (((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 145) {
            pad_temp_shared[((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((3 <= (((((int)vy) * 8) + (((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 385) / 55)) + ry_outer)) && ((((((int)vy) * 8) + (((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 385) / 55)) + ry_outer) < 227)) && (3 <= (((((int)vx) * 56) + rx_outer) + ((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 55)))) && ((((((int)vx) * 56) + rx_outer) + ((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 55)) < 227)) ? placeholder[(((((((((((((int)vz) >> 1) * 150528) + (((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 385) * 50176)) + (((int)vy) * 1792)) + ((((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 385) / 55) * 224)) + (ry_outer * 224)) + (((int)vx) * 56)) + rx_outer) + ((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 55)) - 675))] : 0.000000e+00f);
          }
        }
      }
      if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 32) {
        if (((((int)threadIdx.z) * 12) + ((int)threadIdx.x)) < 96) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[(((((int)threadIdx.z) * 12) + ((int)threadIdx.x)))] = placeholder1[(((((((((int)vz) & 1) * 4704) + (((int)threadIdx.z) * 588)) + (((int)threadIdx.x) * 49)) + (ry_outer * 7)) + rx_outer))];
          }
        }
      }
      __syncthreads();
      for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
        for (int yy = 0; yy < 2; ++yy) {
          compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[(((((int)threadIdx.z) * 3) + rc_inner))]));
          compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 24))]));
          compute[((yy + 8))] = (compute[((yy + 8))] + (pad_temp_shared[((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 48))]));
          compute[((yy + 12))] = (compute[((yy + 12))] + (pad_temp_shared[((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 72))]));
          compute[((yy + 2))] = (compute[((yy + 2))] + (pad_temp_shared[(((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 3) + rc_inner))]));
          compute[((yy + 6))] = (compute[((yy + 6))] + (pad_temp_shared[(((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)) + 220))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 24))]));
          compute[((yy + 10))] = (compute[((yy + 10))] + (pad_temp_shared[(((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)) + 220))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 48))]));
          compute[((yy + 14))] = (compute[((yy + 14))] + (pad_temp_shared[(((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)) + 220))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 72))]));
        }
      }
    }
  }
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    T_relu[(((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)))] = max((compute[(ax2_inner_inner_inner)] + placeholder2[((((((int)vz) & 1) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 100352))] = max((compute[((ax2_inner_inner_inner + 4))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 200704))] = max((compute[((ax2_inner_inner_inner + 8))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 301056))] = max((compute[((ax2_inner_inner_inner + 12))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 224))] = max((compute[((ax2_inner_inner_inner + 2))] + placeholder2[((((((int)vz) & 1) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 100576))] = max((compute[((ax2_inner_inner_inner + 6))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 200928))] = max((compute[((ax2_inner_inner_inner + 10))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 301280))] = max((compute[((ax2_inner_inner_inner + 14))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
  }
    offset+=blocksize[0];
    }
  }
  else
  {
    float compute[16];
  __shared__ float pad_temp_shared[1155];
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
  for (int ry_outer = 0; ry_outer < 7; ++ry_outer) {
    for (int rx_outer = 0; rx_outer < 7; ++rx_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
        if ((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 1155) {
          if (((((int)threadIdx.x) * 6) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) < 145) {
            pad_temp_shared[((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner))] = (((((3 <= (((((int)vy) * 8) + (((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 385) / 55)) + ry_outer)) && ((((((int)vy) * 8) + (((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 385) / 55)) + ry_outer) < 227)) && (3 <= (((((int)vx) * 56) + rx_outer) + ((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 55)))) && ((((((int)vx) * 56) + rx_outer) + ((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 55)) < 227)) ? placeholder[(((((((((((((int)vz) >> 1) * 150528) + (((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 385) * 50176)) + (((int)vy) * 1792)) + ((((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 385) / 55) * 224)) + (ry_outer * 224)) + (((int)vx) * 56)) + rx_outer) + ((((((int)threadIdx.z) * 145) + (((int)threadIdx.x) * 6)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 55)) - 675))] : 0.000000e+00f);
          }
        }
      }
      if (((((int)threadIdx.z) * 4) + (((int)threadIdx.x) / 3)) < 32) {
        if (((((int)threadIdx.z) * 12) + ((int)threadIdx.x)) < 96) {
          if (((int)threadIdx.x) < 12) {
            placeholder_shared[(((((int)threadIdx.z) * 12) + ((int)threadIdx.x)))] = placeholder1[(((((((((int)vz) & 1) * 4704) + (((int)threadIdx.z) * 588)) + (((int)threadIdx.x) * 49)) + (ry_outer * 7)) + rx_outer))];
          }
        }
      }
      __syncthreads();
      for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
        for (int yy = 0; yy < 2; ++yy) {
          compute[(yy)] = (compute[(yy)] + (pad_temp_shared[((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[(((((int)threadIdx.z) * 3) + rc_inner))]));
          compute[((yy + 4))] = (compute[((yy + 4))] + (pad_temp_shared[((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 24))]));
          compute[((yy + 8))] = (compute[((yy + 8))] + (pad_temp_shared[((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 48))]));
          compute[((yy + 12))] = (compute[((yy + 12))] + (pad_temp_shared[((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 72))]));
          compute[((yy + 2))] = (compute[((yy + 2))] + (pad_temp_shared[(((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)) + 220))] * placeholder_shared[(((((int)threadIdx.z) * 3) + rc_inner))]));
          compute[((yy + 6))] = (compute[((yy + 6))] + (pad_temp_shared[(((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)) + 220))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 24))]));
          compute[((yy + 10))] = (compute[((yy + 10))] + (pad_temp_shared[(((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)) + 220))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 48))]));
          compute[((yy + 14))] = (compute[((yy + 14))] + (pad_temp_shared[(((((rc_inner * 385) + (yy * 110)) + (((int)threadIdx.x) * 2)) + 220))] * placeholder_shared[((((((int)threadIdx.z) * 3) + rc_inner) + 72))]));
        }
      }
    }
  }
  for (int ax2_inner_inner_inner = 0; ax2_inner_inner_inner < 2; ++ax2_inner_inner_inner) {
    T_relu[(((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)))] = max((compute[(ax2_inner_inner_inner)] + placeholder2[((((((int)vz) & 1) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 100352))] = max((compute[((ax2_inner_inner_inner + 4))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 200704))] = max((compute[((ax2_inner_inner_inner + 8))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 301056))] = max((compute[((ax2_inner_inner_inner + 12))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 224))] = max((compute[((ax2_inner_inner_inner + 2))] + placeholder2[((((((int)vz) & 1) * 32) + ((int)threadIdx.z)))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 100576))] = max((compute[((ax2_inner_inner_inner + 6))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 8))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 200928))] = max((compute[((ax2_inner_inner_inner + 10))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 16))]), 0.000000e+00f);
    T_relu[((((((((((int)vz) * 401408) + (((int)threadIdx.z) * 12544)) + (((int)vy) * 448)) + (ax2_inner_inner_inner * 112)) + (((int)vx) * 28)) + ((int)threadIdx.x)) + 301280))] = max((compute[((ax2_inner_inner_inner + 14))] + placeholder2[(((((((int)vz) & 1) * 32) + ((int)threadIdx.z)) + 24))]), 0.000000e+00f);
  }
  }
}
extern "C" __global__ void fused_nn_max_pool2d_add_nn_relu_kernel0(int* flag,int* blocknum,int* blocksize,float* __restrict__ placeholder, float* __restrict__ T_relu, float* __restrict__ placeholder1) {
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
    float tensor[1];
  tensor[(0)] = -3.402823e+38f;
  for (int dh = 0; dh < 3; ++dh) {
    for (int dw = 0; dw < 3; ++dw) {
      tensor[(0)] = max(tensor[(0)], (((1 <= ((((((((int)vx) * 1024) + ((int)threadIdx.x)) % 3136) / 56) * 2) + dh)) && (1 <= (((((((int)vx) * 1024) + ((int)threadIdx.x)) % 56) * 2) + dw))) ? placeholder[(((((((((((int)vx) * 1024) + ((int)threadIdx.x)) / 56) * 224) + (dh * 112)) + ((((((int)vx) * 1024) + ((int)threadIdx.x)) % 56) * 2)) + dw) - 113))] : -3.402823e+38f));
    }
  }
  T_relu[(((((int)vx) * 1024) + ((int)threadIdx.x)))] = max((tensor[(0)] + placeholder1[(((((((int)vx) * 1024) + ((int)threadIdx.x)) % 200704) / 3136))]), 0.000000e+00f);
    offset+=blocksize[0];
    }
  }
  else
  {
    float tensor[1];
  tensor[(0)] = -3.402823e+38f;
  for (int dh = 0; dh < 3; ++dh) {
    for (int dw = 0; dw < 3; ++dw) {
      tensor[(0)] = max(tensor[(0)], (((1 <= ((((((((int)vx) * 1024) + ((int)threadIdx.x)) % 3136) / 56) * 2) + dh)) && (1 <= (((((((int)vx) * 1024) + ((int)threadIdx.x)) % 56) * 2) + dw))) ? placeholder[(((((((((((int)vx) * 1024) + ((int)threadIdx.x)) / 56) * 224) + (dh * 112)) + ((((((int)vx) * 1024) + ((int)threadIdx.x)) % 56) * 2)) + dw) - 113))] : -3.402823e+38f));
    }
  }
  T_relu[(((((int)vx) * 1024) + ((int)threadIdx.x)))] = max((tensor[(0)] + placeholder1[(((((((int)vx) * 1024) + ((int)threadIdx.x)) % 200704) / 3136))]), 0.000000e+00f);
  }
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

























