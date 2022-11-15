#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <hip/hip_runtime.h>
#include <vector>
#include <iterator>
#include <fstream>

#define NUM_BINS 1024

__global__ void normalize(float min, float *out)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x; 
	out[tid] -= min;
        //divide max-min
        //(int) *1023
}

__global__ void histogram_smem_atomics(const float *in, float range, float min, float *out)
{
    
    extern __shared__ unsigned int smem[];
    
    // linear thread index within linear block
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    out[t] = 0;
    __syncthreads();

    if(tid< NUM_BINS)
    {
	    smem[tid] = 0;
    }
    __syncthreads();
    int bin = static_cast<int>(((in[t]-min)/(range))*1023);
    if(blockIdx.x == 0) out[tid] = bin;
/*    atomicAdd(&smem[bin],1);

    __syncthreads();
    
    if(tid<NUM_BINS)
    {
        atomicAdd(&(out[tid]), smem[tid]);
    } */
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

    int numElements = 0;
    std::ifstream inputStream("data.txt");
    std::vector<float> numbers;
    std::string line;
    float element;
    if (inputStream)
    {
	while(std::getline(inputStream,line))
        {
        	numbers.push_back(std::stof(line));
		numElements++;
        }
	
    }
    n= numElements;


    float minTemp = numbers[0];
    float maxTemp = numbers[0];
    for(int i = 0; i < numElements; i++)
    {
        minTemp = (minTemp <= numbers[i]) ? minTemp : numbers[i];
        maxTemp = (maxTemp >= numbers[i]) ? maxTemp : numbers[i];
    }
    std::cout<<"Min " << minTemp << '\t' << "Max " << maxTemp << std::endl;
    float value;
    float minValue;
    float maxValue;
    float histoSum;
    
    float *dData;
    float *dReduc; 
     
    size_t original = n*sizeof(float);
    size_t reduc = n/(1024)*sizeof(float);
    size_t reducI = n/(1024)*sizeof(unsigned int);
    
    //Allocation    
    hipMalloc(&dData, original);
    hipMalloc(&dReduc, reduc);
    
    float* ptr = (float*) malloc(sizeof(float)*n);
    
    //Kernel Parameters
    dim3 blockDim(1024, 1, 1);
    dim3 gridDim(n/blockDim.x, 1, 1);

    size_t size = blockDim.x*sizeof(float);
    size_t sizeI = blockDim.x*sizeof(unsigned int);

    std::cout<< "Number of array elements: " << numElements << std::endl;
    //Load array data to gpu
    hipMemcpy(dData, numbers.data(), size, hipMemcpyHostToDevice);

    //LoadArrayDataKernel<<<gridDim, blockDim>>>(dData);

    //Get Max
    ShmemReduceKernelMaxMin<<<gridDim,blockDim,size>>>(dReduc,dData,true);
    ShmemReduceKernelMaxMin<<<1,gridDim,size>>>(dReduc, dReduc,true);    
    hipMemcpy(&value, dReduc, sizeof(float), hipMemcpyDeviceToHost); 
    maxValue = value;

    //Get Min
    ShmemReduceKernelMaxMin<<<gridDim,blockDim,size>>>(dReduc,dData,false);
    ShmemReduceKernelMaxMin<<<1,gridDim,size>>>(dReduc, dReduc,false);
    hipMemcpy(&value, dReduc, sizeof(float), hipMemcpyDeviceToHost);
    minValue = value;   

    //normalize<<<gridDim,blockDim>>>(minValue,dData);
    

    //Histo
    float *hHisto = new float[NUM_BINS];
    float *dHisto;
    hipMalloc((void**)&dHisto, NUM_BINS*sizeof(float)); 

	//hipMemcpy(ptr, dData, sizeof(float)*n, hipMemcpyDeviceToHost);
    	//for(int i =0; i<n; i++)
    	//{
		//std::cout << "data "<< i<<": " << ptr[i] << std::endl;
    	//}

    dim3 blockDimHisto(1024);
    dim3 gridDimHisto((n + blockDim.x -1)/blockDim.x);    

//    histogram_smem_atomics<<<gridDimHisto, blockDimHisto,size>>>(dData, (maxValue-minValue), minValue, dHisto);
    histogram_smem_atomics<<<gridDimHisto, blockDimHisto,size>>>(dData, (maxTemp-minTemp), minTemp, dHisto);

    hipMemcpy(hHisto, dHisto, NUM_BINS*sizeof(float), hipMemcpyDeviceToHost);
    float temp = 0;
    float serialSum = 0;
    for(int i = 0; i<1024; i++)
    {
    	std::cout << "bin "<< i<<": " << hHisto[i] << std::endl;
        serialSum += hHisto[i];
        temp += i;
    }

    
    ShmemReduceKernelSum<<<gridDim,blockDim,size>>>(dReduc,dHisto);
    //ShmemReduceKernelSum<<<1,blockDim,size>>>(dReduc, dReduc);
    hipMemcpy(&histoSum, dReduc, sizeof(float), hipMemcpyDeviceToHost);
    std::cout << "Histo sum: " << histoSum << std::endl; 
    std::cout << "Serial sum: " << serialSum << std::endl;
    std::cout << "Max: " << maxValue << std::endl;
    std::cout << "Min: " << minValue << std::endl;  
    std::cout << "Math: " << (int)(1.880974126480706/((maxValue-minValue)/NUM_BINS)) << std::endl; 
    
    //Free memory
    hipFree(dReduc);
    hipFree(dData);

    return value;
}

/* Driver for the computation of pi. */
int main()
{
        int n = pow(2,4);
        n = 2048;
        float pi = MmmPi(n);
        
}

