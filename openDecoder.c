/*******************************************************************************
** GPU LogMap Decoder V1.0
** Contains all functions to calculate Metrics/Hard Decisions and Wrapper-Funcs
**
** Jan Krämer Dezember 2013
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#include "openDecoderConfig.h"
#include "openDecoder.h"
#include <unistd.h>

#define tic clock_gettime(CLOCK_REALTIME, &timestampA);
#define toc clock_gettime(CLOCK_REALTIME, &timestampB);

void openDecoder(int runs)
{
	FILE   *fp;
  char   openClSource[MAX_SOURCE_SIZE];  	
  size_t fileSize = 0;

  /* Open logMAPKernels.cl and write Content to memory */
  fp = fopen("logMAPKernels.cl", "r");
  if(fp == NULL)
  {
    fprintf(stderr, "Failed to load logMAPKernels.cl\nExiting OpenDecoder\n");
    exit(EXIT_FAILURE);
  }


  fileSize = fread(openClSource, 1, MAX_SOURCE_SIZE, fp);
  if(!fileSize)
  {
    fprintf(stderr, "Failed to read FILE fp\nExiting OpenDecoder\n");
    fclose(fp);
    exit(EXIT_FAILURE);
  } 
  fclose(fp);
  
  /* Get Platform and DeviceIDs */
  cl_int err;
  cl_platform_id platformID;
  cl_device_id deviceID;
  err = clGetPlatformIDs(1, &platformID, NULL);
  if(err != CL_SUCCESS)
  {
    fprintf(stderr, "Failed to get a valid OpenCL Platform\nExiting OpenDecoder\n");
    exit(EXIT_FAILURE);
  }

  err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
  if(err != CL_SUCCESS)
  {
    fprintf(stderr, "Failed to get a valid OpenCL Device\nExiting OpenDecoder\n");
    exit(EXIT_FAILURE);
  }

  /* Create OpenCL GPU Context */
  cl_context_properties properties[] = 
  {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platformID,
    0    
  };

  cl_context gpuContext = clCreateContext(0, 1,  &deviceID, NULL, NULL, &err);
  if(err != CL_SUCCESS)
  {
    fprintf(stderr, "Failed to create a valid OpenCL GPU Context\nExiting OpenDecoder\n");
    exit(EXIT_FAILURE);
  }

  /* Create OpenCL command Queue */
  cl_command_queue cmdQueue = clCreateCommandQueue(gpuContext, deviceID, CL_QUEUE_PROFILING_ENABLE, NULL);
  if(!cmdQueue)
  {
    fprintf(stderr, "Failed to create a valid OpenCL Commandqueue\nExiting OpenDecoder\n");
    exit(EXIT_FAILURE);
  }
  const char *src = openClSource;
  cl_program openClProgram = clCreateProgramWithSource(gpuContext, 1, 
                             (const char**) &src, NULL, NULL);
  if(!openClProgram)
  {
    fprintf(stderr, "Failed to create an OpenCL program with Source\nExiting Decoder\n");
    exit(err);    
  }
  
  const char buildOptions[] = "-cl-mad-enable -cl-nv-verbose ";
  err = clBuildProgram(openClProgram, 0, NULL, "-cl-nv-verbose -cl-mad-enable -cl-nv-maxrregcount=112", NULL, NULL);
  if(err != CL_SUCCESS)
  {
    printf("ErrorCode %d\n", err);
    /*Get Build Log*/
    char buildLog[1000 * 1024];
    clGetProgramBuildInfo(openClProgram, deviceID, CL_PROGRAM_BUILD_LOG, 
                          sizeof(char)*1000*1024, buildLog, NULL);
    fprintf(stderr, "Compilation failed du to following reasons\n\n%s\n\n", buildLog);
    clReleaseProgram(openClProgram);
    exit(err);
  }

  char buildLog[1000 * 1024];
  clGetProgramBuildInfo(openClProgram, deviceID, CL_PROGRAM_BUILD_LOG, 
                          sizeof(char)*1000*1024, buildLog, NULL);
  
  
  FILE *fileP = fopen("build.log", "w+");
  if(fileP != NULL)
  {
    fputs(buildLog, fileP);
    fclose(fileP);
  }  
  else
  {
    fprintf(stderr, "Could not write BuildLog\n");
  }

    /*Benchmark Initialisation*/
  struct timespec timestampA;
  struct timespec timestampB;
  double timerStart = 0.0;
  double timerStop  = 0.0;
  double timerDiff  = 0.0;
  unsigned int run;
  cl_event timerEventA;
  cl_event timerEventB;
  cl_ulong timeStart, timeStop;

/******************************************************************************/
  /*Initialisation of LLRIN and LLROUT*/
  float *llrIn_H;
  float *llrOut_H;
  float llrIn_Buf[_N];// = {3.24376,  4.82663,  2.30248,  -1.56122, 0.84069,  -3.92230, 4.06308,  3.79653,  3.17760,  -2.39272, 0.94356,  -4.77487, -0.74740, -1.87281, -3.38515, -3.21233, -0.77114, -4.05770, 0.98523,  -0.29075, 1.95949,  1.99887,  1.38530,  -4.663961,  -4.31193, -1.80400, 0.30864,  1.54445,  -0.92380, 3.19981,  2.18358,  4.68649};
  srand(time(NULL));
  for(unsigned int k = 0; k < _N; k++)
  { 
      llrIn_Buf[k] = 10*(float)(rand()/(float)RAND_MAX-0.5);
  }

/******************************************************************************/
  /*Initialisation of alpha and beta matrices*/
  
  float alphaBuf[(_K+1)*_NUMSTATES];
  float betaBuf[(_K+1)*_NUMSTATES];
  float *alpha_H;
  float *beta_H;
  float *gamma_H;
  for(int s = 0; s < _NUMSTATES; s++)
  {
      alphaBuf[s]              = -INF;
      betaBuf[_K*_NUMSTATES + s] = -log(_NUMSTATES);
  }
  alphaBuf[0] = 0;

  for(int s = 1; s < _NUMSUBS; s++)
  {
      alphaBuf[(s*_LSUB) *_NUMSTATES + s] = -log(_NUMSTATES);
      betaBuf[(s*_LSUB)*_NUMSTATES + s] = -log(_NUMSTATES);
  }

/******************************************************************************/
  /****************************************************************************
  ** ALLOCATE MEMORY ON DEVICE
  *****************************************************************************/
  cl_mem pinnedLLRIN_H;
  cl_mem pinnedGamma_H;
  cl_mem pinnedAlpha_H;
  cl_mem pinnedBeta_H;
  cl_mem pinnedLLROUT_H;
  cl_mem llrIn_G;
  cl_mem gamma_G;
  cl_mem alpha_G;
  cl_mem beta_G;
  cl_mem llrOut_G;

  size_t globalWorkSize[1];
  size_t localWorkSize[1];

  pinnedLLRIN_H = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, 
                                  sizeof(float)*_N, NULL, &err);
  pinnedGamma_H = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 
                                  sizeof(float)*_K*_NUMTRANS, NULL, &err);

  llrIn_G = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY,
                                  sizeof(float) * _N, NULL, NULL);
  gamma_G = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE,
                                  sizeof(float) * _NUMTRANS * _K, NULL, NULL);
  
  pinnedAlpha_H = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, 
                                  sizeof(float)*(_K+1)*_NUMSTATES, NULL, &err);
  pinnedBeta_H = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 
                                  sizeof(float)*(_K+1)*_NUMSTATES, NULL, &err);

  alpha_G = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE,
                                  sizeof(float)*(_K+1)*_NUMSTATES, NULL, NULL);
  beta_G = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE,
                                  sizeof(float)*(_K+1)*_NUMSTATES, NULL, NULL);

  pinnedLLROUT_H = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 
                                  sizeof(float)*_N, NULL, &err);
  llrOut_G = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE,
                                  sizeof(float)*_N, NULL, NULL);


  /* MAP the the Data pointer llrIn_H to pinned memory*/
  llrIn_H = (float*) clEnqueueMapBuffer(cmdQueue, pinnedLLRIN_H, CL_TRUE, CL_MAP_WRITE,
                                        0, sizeof(float) * _N, 0, NULL, NULL, &err);
  if(err != CL_SUCCESS)
  {
    fprintf(stderr, "Error Mapping Data Buffer to pinned Host Memory\n");
    exit(err);
  }

  gamma_H = (float*) clEnqueueMapBuffer(cmdQueue, pinnedGamma_H, CL_TRUE, CL_MAP_WRITE,
                                        0, sizeof(float)*_K*_NUMTRANS, 0, NULL, NULL, &err);
  if(err != CL_SUCCESS)
  {
    fprintf(stderr, "Error Mapping Data Buffer to pinned Host Memory\n");
    exit(err);
  }
  
  /* Initialization LLRIN Databuffer*/
  for(int i = 0; i < _K; i++)
  {
    llrIn_H[i<<1]      = llrIn_Buf[i<<1];
    llrIn_H[(i<<1) + 1] = llrIn_Buf[(i<<1) + 1];
  }
  
  /* MAP ALPHA AND BETA MATRICES TO PINNED HOST MEMORY*/
  alpha_H = (float*) clEnqueueMapBuffer(cmdQueue, pinnedAlpha_H, CL_TRUE, CL_MAP_WRITE,
                                        0, sizeof(float)*(_K+1)*_NUMSTATES, 0, NULL, NULL, &err);
  if(err != CL_SUCCESS)
  {
    fprintf(stderr, "Error Mapping Data Buffer to pinned Host Memory\n");
    exit(err);
  }

  beta_H = (float*) clEnqueueMapBuffer(cmdQueue, pinnedBeta_H, CL_TRUE, CL_MAP_WRITE,
                                        0, sizeof(float)*(_K+1)*_NUMSTATES, 0, NULL, NULL, &err);
  if(err != CL_SUCCESS)
  {
    fprintf(stderr, "Error Mapping Data Buffer to pinned Host Memory\n");
    exit(err);
  }
  
/*Initialisation of beta and alpha matrices*/
  for(int i = 0; i < (_K+1)*_NUMSTATES; i++)
  { 
    alpha_H[i] = alphaBuf[i];
    beta_H[i] = betaBuf[i];
  }

/* PreCopy Alpha and Beta Matrices to GPU Memory*/
  clEnqueueWriteBuffer(cmdQueue, alpha_G, CL_TRUE, 0, sizeof(float)*(_K+1)*_NUMSTATES,
                   alphaBuf, 0, NULL, NULL);
  clEnqueueWriteBuffer(cmdQueue, beta_G, CL_TRUE, 0, sizeof(float)*(_K+1)*_NUMSTATES,
                   betaBuf, 0, NULL, NULL);

/* MAP Pinned Memory for LLROUT*/
  llrOut_H = (float*) clEnqueueMapBuffer(cmdQueue, pinnedLLROUT_H, CL_TRUE, CL_MAP_WRITE,
                                        0, sizeof(float) * _N, 0, NULL, NULL, &err);
  if(err != CL_SUCCESS)
  {
    fprintf(stderr, "Error Mapping Data Buffer to pinned Host Memory\n");
    exit(err);
  }  

  printf("Allocated Memory on GPU\nNow starting Decoding of Datastream\n");

/******************************************************************************/
for(run = 0; run < runs; run++)
  {
    
    //printf("\r%3.0f%%",(float)run/runs*100);
    
    /**************************************************
    ** LogMap                                        **
    **************************************************/
    globalWorkSize[0] = _K * 8;
    localWorkSize[0] = 8;
    tic;
    clEnqueueWriteBuffer(cmdQueue, llrIn_G, CL_TRUE, 0, sizeof(float) * _N,
                         llrIn_H, 0, NULL, NULL);
    /*Create the OpenCL Kernel and set kernel arguments for Gamma*/
    cl_kernel createGamma_G = clCreateKernel(openClProgram, "creategamma", &err);
    if(err !=  CL_SUCCESS)
    {
      fprintf(stderr, "Error in Create Gamma found\n");
      exit(err);
    }     
    clSetKernelArg(createGamma_G, 0, sizeof(cl_mem),     (void*) &llrIn_G);
    clSetKernelArg(createGamma_G, 1, sizeof(cl_mem),     (void*) &gamma_G);
    //clSetKernelArg(createGamma_G, 2, sizeof(cl_float)*(localWorkSize[0]<<1), NULL);
       
    err = clEnqueueNDRangeKernel(cmdQueue, createGamma_G, 1, 0, globalWorkSize, localWorkSize,
                          0, NULL, NULL);
    if(err != CL_SUCCESS)
    {
      fprintf(stderr, "Could not execute Kernel Gamma on GPU device Error: %d \n", err);
      exit(err);    
    }    
    /*err = clEnqueueReadBuffer(cmdQueue, gamma_G, CL_FALSE, 0, sizeof(float)*_K*_NUMTRANS,
                        gamma_H, 0, NULL, NULL);
    
    if(err != CL_SUCCESS)
    {
      fprintf(stderr, "Could not copy Gamma from GPU device\n");
    }

    clFinish(cmdQueue);
    for (int i = 0; i < _K*_NUMTRANS; ++i)
    {
      printf("Gamma_H[%d][%d] = %f\n", i%_NUMTRANS, (int) i/_NUMTRANS, gamma_H[i]);
    }

    /*Create OpenCL Kernel to create the Matrices alpha/beta*/
    cl_kernel createMatrices_G = clCreateKernel(openClProgram, "matricesSubDecoder", NULL);
    clSetKernelArg(createMatrices_G, 0, sizeof(cl_mem), (void*) &alpha_G);
    clSetKernelArg(createMatrices_G, 1, sizeof(cl_mem), (void*) &beta_G);
    clSetKernelArg(createMatrices_G, 2, sizeof(cl_mem), (void*) &gamma_G);

    globalWorkSize[0] = _NUMSUBS*8;
    localWorkSize[0] = 8;    
    err = clEnqueueNDRangeKernel(cmdQueue, createMatrices_G, 1, 0, globalWorkSize, localWorkSize,
                         0, NULL, NULL);
    if(err != CL_SUCCESS)
    {
      fprintf(stderr, "Could not execute Kernel Matrices on GPU device Error: %d\n", err);
      exit(err);    
    }       
   /* err = clEnqueueReadBuffer(cmdQueue, alpha_G, CL_FALSE, 0, sizeof(float)*(_K+1)*_NUMSTATES,
                        alpha_H, 0, NULL, NULL);
    
    clFinish(cmdQueue);
    if(err != CL_SUCCESS)
    {
      fprintf(stderr, "Could not copy Beta from GPU device\n");
    }
    /*for (int i = 0; i < (_K+1)*_NUMSTATES; ++i)
    {
      printf("alpha_H[%d][%d] = %f\n", i%_NUMSTATES, (int) i/_NUMSTATES, alpha_H[i]);
    }*/

    /*for (int i = 0; i < _NUMSUBS; ++i)
    {
      for (int s = 0; s < _NUMSTATES; ++s)
      {
        printf("beta_H[%d][%d] = %f\n", s, i*_LSUB+_LSUB, beta_H[(i*_LSUB+_LSUB)*_NUMSTATES+s]);
      }
    }


    /*LLROUT*/
    cl_kernel decodeLLR_G = clCreateKernel(openClProgram, "llrOut", NULL);
    clSetKernelArg(decodeLLR_G, 0, sizeof(cl_mem), (void*) &alpha_G);
    clSetKernelArg(decodeLLR_G, 1, sizeof(cl_mem), (void*) &beta_G);
    clSetKernelArg(decodeLLR_G, 2, sizeof(cl_mem), (void*) &gamma_G); 
    clSetKernelArg(decodeLLR_G, 3, sizeof(cl_mem), (void*) &llrOut_G);
    globalWorkSize[0] = _K;
    localWorkSize[0] = 256;
    err = clEnqueueNDRangeKernel(cmdQueue, decodeLLR_G, 1, 0, globalWorkSize, localWorkSize,
                           0, NULL, NULL);
    if(err != CL_SUCCESS)
    {
      fprintf(stderr, "Could not execute Kernel LLROUT on GPU device Error: %d\n", err);
      exit(err);    
    }    

    err = clEnqueueReadBuffer(cmdQueue, llrOut_G, CL_TRUE, 0, sizeof(float)*_N,
                        llrOut_H, 0, NULL, NULL);
    toc;
    if(err != CL_SUCCESS)
    {
      fprintf(stderr, "Could not copy LLR from GPU device Error: %d\n", err);
    }
    //toc;

   /* for (int i = 0; i < _K; ++i)
    {
      int index = i;
      printf("LLR_H[%d] = %f\n", index<<1, llrOut_H[i] - llrIn_H[i<<1]);
      printf("LLR_H[%d] = %f\n", (index<<1) + 1, llrOut_H[i + _K] - llrIn_H[(i<<1) + 1]);
    }
    /*DATAOUT*/
    /*cl_mem dataOUT_G = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE,
                                     sizeof(float)*_K, NULL, NULL);
    cl_kernel decodeData_G = clCreateKernel(openClProgram, "dataOut", NULL);
    clSetKernelArg(decodeData_G, 0, sizeof(cl_mem), (void*) &alpha_G);
    clSetKernelArg(decodeData_G, 1, sizeof(cl_mem), (void*) &beta_G);
    clSetKernelArg(decodeData_G, 2, sizeof(cl_mem), (void*) &gamma_G); 
    clSetKernelArg(decodeData_G, 3, sizeof(cl_mem), (void*) &dataOUT_G);
    globalWorkSize[0] = 16;
    clEnqueueNDRangeKernel(cmdQueue, decodeData_G, 1, 0, globalWorkSize, localWorkSize,
                           0, NULL, NULL);
    
    float dataOUT_H[_K];
    clEnqueueReadBuffer(cmdQueue, dataOUT_G, CL_TRUE, 0, sizeof(float)*_K,
                        dataOUT_H, 0, NULL, NULL);*/
    timerStart = 1e9*timestampA.tv_sec + timestampA.tv_nsec;
    timerStop  = 1e9*timestampB.tv_sec + timestampB.tv_nsec;

    timerDiff += (timerStop - timerStart)/runs;

    clReleaseKernel      (createMatrices_G);
    clReleaseKernel      (createGamma_G);
    //clReleaseKernel      (decodeData_G);
    //clReleaseMemObject   (dataOUT_G);
    clReleaseKernel      (decodeLLR_G);
    
  }
  clReleaseMemObject   (pinnedLLROUT_H);
  clReleaseMemObject   (llrOut_G);
  clReleaseMemObject   (pinnedBeta_H);
  clReleaseMemObject   (pinnedAlpha_H);
  clReleaseMemObject   (alpha_G);
  clReleaseMemObject   (beta_G);
  clReleaseMemObject   (gamma_G);
  clReleaseMemObject   (llrIn_G);
  clReleaseMemObject   (pinnedGamma_H);
  clReleaseMemObject   (pinnedLLRIN_H);
  clReleaseProgram(openClProgram);
  clReleaseCommandQueue(cmdQueue);
  clReleaseContext(gpuContext);

 // benchmarkSave(timerDiff);
  printf(" --> Execution Time for LLR decoding: %3.6f us\n",timerDiff/1000.0/_N);
  printf(" --> Data Rate for LLR:      %3.6f kbps\n", _N*1000000.0/timerDiff);
}

/*Saving benchmark to file*/
void benchmarkSave(double time)
{
  
  FILE *fp;
  double meanDelay = INFINITY;
  double meanSymbolTime = INFINITY;
  double meanSymbolRate = 0;
  int guardInterval = _GUARDINTTERVAL;
  int lSub = _LSUB;
  const char mode[12] = "MultiKernel";
  int cCount = 0;

  meanDelay = time/1000;
  meanSymbolTime = time/(_N*1000);
  meanSymbolRate = _N*1e6/time;

  /* Open logMAPKernels.cl and write Content to memory */
  fp = fopen("speedLog.dat", "a");
  if(fp == NULL)
  {
    fprintf(stderr, "Failed to load speedLog.dat\nUnable to save results\n");
    exit(EXIT_FAILURE);
  }  

    cCount = fprintf(fp, "%-19.2f %-17.2f %-13.2f %-25d %-23d %s\n", meanSymbolRate, meanSymbolTime, meanDelay, lSub, guardInterval, mode);
    if(cCount == 0)
    {
      fprintf(stderr, "Error: Could not write to file\n->Benchmark results lost");
      fclose(fp);      
      exit(EXIT_FAILURE);
    }

    fprintf(stdout, "Benchmark results written to speedLog.dat\n");
    fclose(fp);
}

