extern "C" __device__ uint get_smid(void) {

    uint ret;
  
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
  
    return ret;
  
}
extern "C" __global__ void matrixMulCpu(int *sm)
{   sm[blockIdx.z] = get_smid();
    float sum = 0.0f;
    for (int time = 0; time <= 100000000; ++time) {
        for(float i = 0.0f; i < 10000000.0f; i+=1.0f)
        {
            for(float j = 0.0f; j < 10000000.0f; j+=1.0f)
            {
                for(float l = 0.0f; l < 1000000.0f; l+=1.0f)
                {
                    sum += (i * j + l) * (i * l + j) * blockIdx.z * blockIdx.z;
                    sum -= (i * j) * (i * l) * blockIdx.z * blockIdx.z;
                }
                sum = 0.0f;
            }
        }
    }
}