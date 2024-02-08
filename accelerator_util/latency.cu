/*
Compilation
1)
sourcer Pierrick environment (HPC/AI soft stack):
source /home/users/ppochelu/program/miniconda/scripts/env.sh

2) commamd
nvcc nvlink.cu -L/home/users/ppochelu/program/nccl/install/nccl_2.17.1-1+cuda12.0_x86_64//lib/ -I/home/users/ppochelu/program/nccl/install/nccl_2.17.1-1+cuda12.0_x86_64//include/ -lnccl

Code inspired from: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
 */


#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"

// chrono
#include <stdio.h>
#include <time.h>



#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{
  ncclComm_t comms[4];


  const int nDev = 2;
  int size = 1;
  int devs[4] = { 0, 1, 2, 3 };


  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }


  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));



 double cumul_time=0;
 int num_pings=10000
 double worst_ping=0;


 for (int p=0;p<num_pings;p++){
  

   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
	 

    // Start chrono
    clock_t start, end;
    double cpu_time_used;
    double through;
    start = clock();

  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));

  NCCLCHECK(ncclGroupEnd());


  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  // End chrono
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    if(i>0){//the first ping is slower, thus the first iteration is a warmup
    cumul_time+=cpu_time_used;
    if(cpu_time_used>worst_ping){
     worst_ping=cpu_time_used;
    }
    }
    //printf(" Ping = %.9f sec \n", cpu_time_used );
 }
    
    printf(" Mean ping = %.9f sec \n", cumul_time/num_pings );
    printf(" Worst ping = %.9f sec \n", worst_ping );


  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }


  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
