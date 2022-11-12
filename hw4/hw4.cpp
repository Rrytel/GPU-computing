#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <hip/hip_runtime.h>
#define NUM_BINS 1024


__global__ void histogram_smem_atomics(const float *in, int width, int height, float *out)
{
    // pixel coordinates
    int binSize = 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // linear thread index within 2D block
    int t = threadIdx.x + threadIdx.y * blockDim.x; 
    //int t = threadIdx.x + blockDim.x*blockIdx.x;
    
    // initialize temporary accumulation array in shared memory
    __shared__ unsigned int smem[NUM_BINS];
    __shared__ unsigned int numThreads;
    if(t<NUM_BINS)
    {
       smem[t] = 0;
    }
    numThreads = 0;
    
    __syncthreads();
    atomicAdd(&numThreads,1);
    
    // process pixels
    // updates our block's partial histogram in shared memory
    unsigned int rgb;
    //rgb = (unsigned int)(in[y * width + x]); //Numbers between 0 and 1024
    rgb = (unsigned int)(in[t]);
    atomicAdd(&smem[rgb/binSize], 1);

    __syncthreads();
    
    // write partial histogram into the global memory
    //  out += g * NUM_BINS;
    if(t<NUM_BINS)
    {
       //atomicAdd(&out[t], smem[t]);
       atomicAdd(&out[t],1);
    } 
    //atomicAdd(&out[t], numThreads);
}


__global__ void LoadArrayDataKernel(float *f1)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x; 
	f1[tid] = tid;
        
}

__global__ void ShmemReduceKernelSum(float * dOut, float *dIn)
{
    // sdata is allocated in the kernel call: via dynamic shared memeory
    extern __shared__ float sData[];
    
    int myId = threadIdx.x + blockDim.x*blockIdx.x;
    int tId = threadIdx.x;
    
    //load shared mem from global mem
    sData[tId] = dIn[myId];
    __syncthreads(); // always sync before using sdata
    
    //do reduction over shared memory
    for(int s = blockDim.x/2; s>0; s >>=1)
    {
        if(tId < s)
        {
            sData[tId] += sData[tId + s];
        }
        __syncthreads(); //make sure all additions are finished
    }
    
    //only tid 0 writes out result!
    if(tId == 0)
    {
            dOut[blockIdx.x] = sData[0];
    }
}

__global__ void ShmemReduceKernelMaxMin(float * dOut, const float *dIn, const bool isMax)
{
    // sdata is allocated in the kernel call: via dynamic shared memeory
    extern __shared__ float sData[];
    
    int myId = threadIdx.x + blockDim.x*blockIdx.x;
    int tId = threadIdx.x;
    
    //load shared mem from global mem
    sData[tId] = dIn[myId];
    __syncthreads(); // always sync before using sdata
    
    //do reduction over shared memory
    for(int s = blockDim.x/2; s>0; s >>=1)
    {
        if(tId < s)
        {
            if(isMax)
            {
               sData[tId] = max(sData[tId + s],sData[tId]);
            }
            else
            {
               sData[tId] = min(sData[tId + s],sData[tId]);
            }
        }
        __syncthreads(); //make sure all additions are finished
    }
    
    //only tid 0 writes out result!
    if(tId == 0)
    {
            dOut[blockIdx.x] = sData[0];
    }
}

float MmmPi(int n)
{
    //Initialization
    float value;
    float minValue;
    float maxValue;
    float histoSum;
    
    
    float *dData;
    
    float *dReduc; 
    unsigned int *dReducI; 
    size_t original = n*sizeof(float);
    size_t reduc = n/(1024)*sizeof(float);
    size_t reducI = n/(1024)*sizeof(unsigned int);
    
    //Allocation    
    hipMalloc(&dData, original);
    hipMalloc(&dReduc, reduc);
    hipMalloc(&dReducI, reducI);
    
    float* ptr = (float*) malloc(sizeof(float)*n);
    
    //Kernel Parameters
    dim3 blockDim(1024, 1, 1);
    dim3 gridDim(n/blockDim.x, 1, 1);

    size_t size = blockDim.x*sizeof(float);
    size_t sizeI = blockDim.x*sizeof(unsigned int);

    //for(int j = 0; j < n; j++)
    //{
	//ptr[j] = j;
    //}
    //float Max = 0;
    //for(int i = 0; i < n; i++)
    //{ 
	//Max = max(ptr[i],Max);
    //}
    //hipMemcpy(dData, ptr, sizeof(float)*n,hipMemcpyHostToDevice);
    LoadArrayDataKernel<<<gridDim, blockDim,size>>>(dData);


    //hipMemcpy(dData, ptr, sizeof(float)*n,hipMemcpyHostToDevice);
    //std::cout << "serial max: " << Max << std::endl;
    ShmemReduceKernelMaxMin<<<gridDim,blockDim,size>>>(dReduc,dData,true);
    ShmemReduceKernelMaxMin<<<1,blockDim,size>>>(dReduc, dReduc,true);    

    hipMemcpy(&value, dReduc, sizeof(float), hipMemcpyDeviceToHost); 
    maxValue = value;

    //LoadArrayDataKernel<<<gridDim, blockDim,size>>>(dData);
    ShmemReduceKernelMaxMin<<<gridDim,blockDim,size>>>(dReduc,dData,false);
    ShmemReduceKernelMaxMin<<<1,blockDim,size>>>(dReduc, dReduc,false);
    hipMemcpy(&value, dReduc, sizeof(float), hipMemcpyDeviceToHost);
    minValue = value;   


    //Histo
    float *hHisto = new float[NUM_BINS];
    float *dHisto;
    hipMalloc((void**)&dHisto, NUM_BINS*sizeof(float)); 

	//hipMemcpy(ptr, dData, sizeof(float)*n, hipMemcpyDeviceToHost);
    	//for(int i =0; i<n; i++)
    	//{
		//std::cout << "data "<< i<<": " << ptr[i] << std::endl;
    	//}

    
    histogram_smem_atomics<<<gridDim, blockDim>>>(dData, 2048, 0, dHisto);
    hipMemcpy(hHisto, dHisto, NUM_BINS*sizeof(float), hipMemcpyDeviceToHost);
    float temp = 0;
    for(int i = 0; i<1024; i++)
    {
    	std::cout << "bin "<< i<<": " << hHisto[i] << std::endl;
        temp += i;
        if(true)
        {
                //std::cout << "bin "<< i<<": " << i/2<< std::endl;
        }
        else
	{
		//std::cout << "bin "<< i<<": " << i<< std::endl;
	}
        
    }

    //Sum histo // 524544
    
    //ShmemReduceKernelSum<<<gridDim,blockDim,size>>>(dReduc,dData);
    //ShmemReduceKernelSum<<<1,blockDim,size>>>(dReduc, dReduc);
    //hipMemcpy(&value, dReduc, sizeof(float), hipMemcpyDeviceToHost);
    std::cout << "Histo sum: " << value << std::endl; 
    std::cout << "Max: " << maxValue << std::endl;
    std::cout << "Min: " << minValue << std::endl;  
     //std::cout << n/1024 << std::endl;
    
    //Free memory
    hipFree(dReduc);
    hipFree(dData);
    hipFree(dReducI);
    return value;
}

/* Driver for the computation of pi. */
int main()
{
        int n = pow(2,4);
        //n = 1024;
        n = 2048;
        float pi = MmmPi(n);
        std::cout<<" Pi = "<< pi <<std::endl;
        
        //unsigned int *hHisto = new unsigned int[NUM_BINS];
        //unsigned int *dHisto;
        //hipMalloc((void**)&dHisto, NUM_BINS*sizeof(unsigned int));
}

