#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <thread>
#include <random>
#include <ctime>
using namespace std;

enum class Unit{
    Byte, KB, MB, GB, TB, PB, EB
};


double convert(double size, Unit unit)
{
    double result = size;
    switch (unit)
    {
    case Unit::EB:
        result /= 1024;     // flow through
    case Unit::PB:
        result /= 1024;     // flow through
    case Unit::TB:
        result /= 1024;     // flow through
    case Unit::GB:
        result /= 1024;     // flow through
    case Unit::MB:
        result /= 1024;     // flow through
    case Unit::KB:
        result /= 1024;     // flow through
    case Unit::Byte:
        result /= 1;
    default:
        break;
    }
    return result;
}

void getMem() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Free mem = %.4f MB, Total = %.4f MB \n", convert(free, Unit::MB), convert(total, Unit::MB));
}

void getMembycu() {
    size_t free, total;
    int err = cuMemGetInfo(&free, &total);
    if (err) {
        cout<<"getMembycu error:"<<err<<endl;
    }
    else {
        printf("Free mem = %.4f MB, Total = %.4f MB \n", convert(free, Unit::MB), convert(total, Unit::MB));
    }
}

void getLimit() {
    size_t value;
    int err = cuCtxGetLimit(&value, CU_LIMIT_STACK_SIZE);
    if(err) {
        printf("[getLimit]:[CU_LIMIT_STACK_SIZE]:%d", err);
        exit(1);
    }
    //printf("getLimit:stack_size = %.4f MB\n", convert(value, Unit::MB));

    err = cuCtxGetLimit(&value, CU_LIMIT_PRINTF_FIFO_SIZE);
    if(err) {
        printf("[getLimit]:[CU_LIMIT_PRINTF_FIFO_SIZE]:%d", err);
        exit(1);
    }
    //printf("getLimit:printf_fifo_size = %.4f MB\n", convert(value, Unit::MB));

    err = cuCtxGetLimit(&value, CU_LIMIT_MALLOC_HEAP_SIZE);
    if(err) {
        printf("[getLimit]:[CU_LIMIT_MALLOC_HEAP_SIZE]:%d", err);
        exit(1);
    }
    //printf("getLimit:malloc_heap_size = %.4f MB\n", convert(value, Unit::MB));
}


int main(int argc，char** argv)
{
    if (argc < 2)
        printf("args num error! argc:%d", argc);
    cudaSetDevice(argv[1]);
    while(1) {
        getMem();
        std::this_thread::sleep_for(std::chrono:: milliseconds (50));
    }

}