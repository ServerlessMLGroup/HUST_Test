#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
#define BLOCKX 9
#define BLOCKY 9
#define THREADX 9
#define THREADY 9
#define COREX 9
#define COREY 9
#define RESIZEBLOCKX 20
#define RESIZETHREADX 30
#define ITERATION ((BLOCKX*BLOCKY*THREADX*THREADY-1)/(RESIZEBLOCKX*RESIZETHREADX)+1)
#define LEFT (BLOCKX*BLOCKY*THREADX*THREADY - (ITERATION-1)*RESIZEBLOCKX*RESIZETHREADX)
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}
void check(cudaError_t err)
    {
    const char * errorStr;
    errorStr = cudaGetErrorString(err);
    //printf("checkCudaErrors()  error = %04d %s\n",err, errorStr);
    printf("checkCudaErrors()  error =  %s\n", errorStr);
    }
//this code is to test the ptb way yy suggests

#define GPU_RETURN_STATUS(cmd) \
{ \
    CUresult result = cmd; \
    if (result != CUDA_SUCCESS) { \
        std::cout << #cmd " error, return code:" << result << " | " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;

    errorStr = cudaGetErrorString(err);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}

//diy concolution kernel
__global__ void convolutionkernel(float** photo,float**** temp,float** convolutioncore,float** result) {
    //confirm the element
    int newx = blockIdx.x;
    int newy = blockIdx.y;

    for(){
    //get the data based on the threadIdx.x and threadIdx.y
    int thx = threadIdx.x;
    int thy = threadIdx.y;

    //caculate(COREX * COREY thread respectively by each thread)
    temp[newy][newx][thy][thx] = photo[newy + thy][newx + thx] * convolutioncore[thy][thx];
    __syncthreads();

    //get the final result by one thread
    if (thx == 0 && thy == 0)
    for(int i = 0;i < COREY;i++){
        for(int j = 0;j < COREX;j++){
            result[newy][newx] +=temp[newy][newx][i][j];
            }
        }
    }
    }
}

__global__ void resizeconvolutionkernel(float** photo,float**** temp,float** convolutioncore,float** result) {
//preparation
    //get the blockidx and threadidx
    int oldx = blockIdx.x;
    int oldthx = threadIdx.x;

    //calculate the offset of the theread in the grid
    int offset = oldx*RESIZETHREADX + oldthx;

    //create index to get the virtual offset in the virtual grid fuether
    int index = offset;

    //new y ,new x ,thy ,thx represent the virtual blockidx.y blockidx.x threadx.y threadx.x
    int newy = 0;
    int newx = 0;
    int thy = 0;
    int thx = 0;

    for(int i=0;i<ITERATION;i++)
    {
        //the last time should be limited because (RESIZEBLOCKX*RESIZETHREADX)may not able to be
        //divided exactly by BLOCKX*BLOCKY*THREADX*THREADY
        if(i!=(ITERATION-1))
        {
            //get the new y ,new x ,thy ,thx
            index = i*RESIZETHREADX*RESIZEBLOCKX +offset;
            newy = index / (BLOCKX*COREX*COREY);
            newx = (index - newy * (BLOCKX*COREX*COREY))/(COREX*COREY);
            thy = (index - newy*(BLOCKX*COREX*COREY) -newx*(COREX*COREY))/COREX;
            thx = index - newy*(BLOCKX*COREX*COREY) -newx*(COREX*COREY) - thy*COREX;

            //caculate(COREX * COREY thread respectively by each thread)
            temp[newy][newx][thy][thx] = photo[newy + thy][newx + thx] * convolutioncore[thy][thx];
        }
        else
        {
            if(offset<LEFT){
                //should be limited within the index
                index = i*RESIZETHREADX*RESIZEBLOCKX +oldx*RESIZETHREADX + oldthx;
                newy = index/(BLOCKX*COREX*COREY);
                newx = (index-newy*(BLOCKX*COREX*COREY))/(COREX*COREY);
                thy = (index - newy*(BLOCKX*COREX*COREY) -newx*(COREX*COREY))/COREX;
                thx = index - newy*(BLOCKX*COREX*COREY) -newx*(COREX*COREY) - thy*COREY;
                //caculate(COREX * COREY thread respectively by each thread)
                temp[newy][newx][thy][thx] = photo[newy + thy][newx + thx] * convolutioncore[thy][thx];
            }
        }
    }

    //syncronize the threads
    __syncthreads();

    //get the result
    for(int i=0;i<ITERATION;i++)
    {
        if(i!=(ITERATION-1))
        {
            //get the final result by one thread
            index = i*RESIZETHREADX*RESIZEBLOCKX +oldx*RESIZETHREADX + oldthx;
            newy = index/(BLOCKX*COREX*COREY);
            newx = (index-newy*(BLOCKX*COREX*COREY))/(COREX*COREY);
            thy = (index - newy*(BLOCKX*COREX*COREY) -newx*(COREX*COREY))/COREX;
            thx = index - newy*(BLOCKX*COREX*COREY) -newx*(COREX*COREY) - thy*COREY;
            //get the final result by thread 0
            if (thx == 0 && thy == 0){
            for(int i = 0;i < COREY;i++){
                for(int j = 0;j < COREX;j++){
                    result[newy][newx] +=temp[newy][newx][i][j];
                    }
                }
            }
        }
        else
        {
            if(offset<LEFT){
                index = i*RESIZETHREADX*RESIZEBLOCKX +oldx*RESIZETHREADX + oldthx;
                newy = index/(BLOCKX*COREX*COREY);
                newx = (index-newy*(BLOCKX*COREX*COREY))/(COREX*COREY);
                thy = (index - newy*(BLOCKX*COREX*COREY) -newx*(COREX*COREY))/COREX;
                thx = index - newy*(BLOCKX*COREX*COREY) -newx*(COREX*COREY) - thy*COREY;
                //get the final result by one thread
                if (thx == 0 && thy == 0){
                for(int i = 0;i < COREY;i++){
                    for(int j = 0;j < COREX;j++){
                        result[newy][newx] +=temp[newy][newx][i][j];
                        }
                    }
                }
            }
        }
    }
}




void run_kernel() {
    //device variable  dphoto:the old matrix  dconvolutioncore:as the name
    //dtemp:four dimesional ,for each node in the photo, save the temporory result of concolution
    //dresult:save the result
    float **dphoto2 = NULL;
    float *dphoto1 = NULL;
    float **dconvolutioncore2 = NULL;
    float *dconvolutioncore1 = NULL;
    float **dresult2 = NULL;
    float *dresult1 = NULL;
    float ****dtemp4 = NULL;
    float ***dtemp3 = NULL;
    float **dtemp2 = NULL;
    float *dtemp1 = NULL;

    //Host variable ,just as above
    float **hphoto2 = NULL;
    float *hphoto1 = NULL;
    float **hconvolutioncore2 = NULL;
    float *hconvolutioncore1 = NULL;
    float **hresult2 = NULL;
    float *hresult1 = NULL;
    float*** htemp4[BLOCKY];
    float** htemp3[BLOCKY][BLOCKX];
    float* htemp2[BLOCKY][BLOCKX][COREY];
    float htemp1[BLOCKY][BLOCKX][COREY][COREX];

	cudaError_t res;

    //test ITERATION and LEFT
    int ite = ITERATION;
    int lef = LEFT;
    printf("Iteration:%d \n",ite);
    printf("Left:%d \n",lef);

    //manage dphoto
	res = cudaMalloc((void**)(&dphoto2), (BLOCKY+COREY-1)*sizeof(float*));CHECK(res)
	res = cudaMalloc((void**)(&dphoto1), (BLOCKY+COREY-1)*(BLOCKX+COREX-1)*sizeof(float));CHECK(res)
	hphoto2 = (float**)malloc((BLOCKY+COREY-1)*(BLOCKX+COREX-1)*sizeof(float*));
	hphoto1 = (float*)malloc((BLOCKY+COREY-1)*(BLOCKX+COREX-1)*sizeof(float));
	//manage two dimesion array
	for (int r = 0; r < (BLOCKY+COREY-1) ; r++)
	{
		hphoto2[r] = dphoto1 + r * (BLOCKX+COREX-1);
	}
	//assign the data to the photo
	for (int r = 0; r < ((BLOCKY+COREY-1)*(BLOCKX+COREX-1)); r++)
	{
		hphoto1[r] = 2.0 ;
	}
	res = cudaMemcpy((void*)(dphoto2), (void*)(hphoto2), (BLOCKY+COREY-1)*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res)
    res = cudaMemcpy((void*)(dphoto1), (void*)(hphoto1), ((BLOCKY+COREY-1)*(BLOCKX+COREX-1))*sizeof(float), cudaMemcpyHostToDevice);CHECK(res)

    //manage dconvolutioncore
	res = cudaMalloc((void**)(&dconvolutioncore2), COREY*sizeof(float*));CHECK(res)
	res = cudaMalloc((void**)(&dconvolutioncore1), COREY*COREX*sizeof(float));CHECK(res)
	hconvolutioncore2 = (float**)malloc(COREY*sizeof(float*));
	hconvolutioncore1 = (float*)malloc(COREY*COREX*sizeof(float));
	//manage two dimesion array
	for (int r = 0; r < COREY; r++)
	{
		hconvolutioncore2[r] = dconvolutioncore1 + r * COREX;
	}
	//assign the data to the core
	for (int r = 0; r < COREY*COREX; r++)
	{
		hconvolutioncore1[r] = 3.0;
	}
	res = cudaMemcpy((void*)(dconvolutioncore2), (void*)(hconvolutioncore2), COREY*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res)
    res = cudaMemcpy((void*)(dconvolutioncore1), (void*)(hconvolutioncore1), COREY*COREX*sizeof(float), cudaMemcpyHostToDevice);CHECK(res)

    //manage dresult
	res = cudaMalloc((void**)(&dresult2), BLOCKY*sizeof(float*));CHECK(res)
	res = cudaMalloc((void**)(&dresult1), BLOCKY*BLOCKX*sizeof(float));CHECK(res)
	hresult2 = (float**)malloc(BLOCKY*sizeof(float*));
	hresult1 = (float*)malloc(BLOCKY*BLOCKX*sizeof(float));
	//manage two dimesion array
	for (int r = 0; r < BLOCKY; r++)
	{
		hresult2[r] = dresult1 + r * BLOCKX;
	}
	//assign the data to the result
	for (int r = 0; r < BLOCKY*BLOCKX; r++)
	{
		hresult1[r] = 0.0;
	}
	res = cudaMemcpy((void*)(dresult2), (void*)(hresult2), BLOCKY*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res)
    res = cudaMemcpy((void*)(dresult1), (void*)(hresult1), BLOCKY*BLOCKX*sizeof(float), cudaMemcpyHostToDevice);CHECK(res)

    //manage dtemp
	res = cudaMalloc((void**)(&dtemp1), BLOCKY*BLOCKX*COREY*COREX*sizeof(float));CHECK(res)
	res = cudaMalloc((void**)(&dtemp2), BLOCKY*BLOCKX*COREY*sizeof(float*));CHECK(res)
	res = cudaMalloc((void**)(&dtemp3), BLOCKY*BLOCKX*sizeof(float**));CHECK(res)
	res = cudaMalloc((void**)(&dtemp4), BLOCKY*sizeof(float***));CHECK(res)
    //manage four dimesion array,malloc is fininshed in line 190
	for(int h=0;h<BLOCKY;h++)
	{
	    htemp4[h] = dtemp3 + h*BLOCKX;
	        for(int i=0;i<BLOCKX;i++)
	        {
	        htemp3[h][i] = dtemp2 + h*BLOCKX*COREY + i*COREY;
	            for(int j=0;j<COREY;j++){
	                htemp2[h][i][j] = dtemp1+ h*BLOCKX*COREY*COREX + i*COREY*COREX+j* COREX;
                }
            }
    }
	res = cudaMemcpy((void*)(dtemp4), (void*)(htemp4), BLOCKY*sizeof(float***), cudaMemcpyHostToDevice);CHECK(res)
	res = cudaMemcpy((void*)(dtemp3), (void*)(htemp3), BLOCKY*BLOCKX*sizeof(float**), cudaMemcpyHostToDevice);CHECK(res)
	res = cudaMemcpy((void*)(dtemp2), (void*)(htemp2), BLOCKY*BLOCKX*COREY*sizeof(float*), cudaMemcpyHostToDevice);CHECK(res)

    //normal way
	dim3 dimBlock(COREX,COREY);
	dim3 dimGrid(BLOCKX,BLOCKY);
    convolutionkernel<<<dimGrid, dimBlock>>>(dphoto2,dtemp4,dconvolutioncore2,dresult2);

    //ptb way
    //resizeconvolutionkernel<<<RESIZEBLOCKX, RESIZETHREADX>>>(dphoto2,dtemp4,dconvolutioncore2,dresult2);

    //Get the result
	res = cudaMemcpy((void*)(hphoto1), (void*)(dresult1), BLOCKY*BLOCKX*sizeof(float), cudaMemcpyDeviceToHost);
	check(res);
	for (int r = 0; r < BLOCKY; r++)
	{
		printf("\ncolum %d ",r);
		for (int c = 0; c < BLOCKX; c++)
		{
			printf("%f ", hphoto1[r*BLOCKX+c]);
		}
	}
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("args num error! argc:%d", argc);
        exit(1);
    }
    int gpu_no = atoi(argv[1]);
    checkCudaErrors(cudaSetDevice(gpu_no));
	run_kernel();
	return 0;
}

