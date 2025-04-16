#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <hip/hip_runtime.h>
#include <vector>
#include <iterator>
#include <fstream>
#include <cmath>

#define NUM_BINS 1024

__device__ float Normal(float x, float mu, float sig) {
  float temp1 = 1.0f / (std::sqrt(2.0f * 3.14) * sig);
  float temp2 = 0.5f * (x - mu) * (x - mu) / (sig * sig);
  float val = temp1 * std::exp(-temp2);
  return val;
}

__global__ void DataGen(float *data, float *x, float xBeg, float deltaX,
                        float mu, float sig, int n) {
  // generate x and data
  // use x for plotting later
  int tId = threadIdx.x + blockDim.x * blockIdx.x;
  if (tId < 1 || tId > n) {
    return;
  }
  float xTemp[2];
  xTemp[0] = xBeg + tId * deltaX;
  xTemp[1] = xBeg + (tId - 1) * deltaX;
  data[tId - 1] =(Normal(xTemp[0], mu, sig) + Normal(xTemp[1], mu, sig)) * deltaX * 0.5f;
  x[tId - 1] = xTemp[0];
}


__global__ void BlScan(float *oData, const float *iData, int n) 
{
  extern __shared__ float temp[]; // allocated on invocation
  int tId = threadIdx.x;
  int offset = 1;
  temp[2 * tId] = iData[2 * tId]; // load input into shared memory
  temp[2 * tId + 1] = iData[2 * tId + 1];
  for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (tId < d) {
      int aI = offset * (2 * tId + 1) - 1;
      int bI = offset * (2 * tId + 2) - 1;
      temp[bI] += temp[aI];
    }
    offset *= 2;
  }
  if (tId == 0) {
    temp[n - 1] = 0;
  }                              // clear the last element
  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (tId < d) {
      int aI = offset * (2 * tId + 1) - 1;
      int bI = offset * (2 * tId + 2) - 1;
      float t = temp[aI];
      temp[aI] = temp[bI];
      temp[bI] += t;
    }
  }
  __syncthreads();
  oData[2 * tId] = temp[2 * tId]; // write results to device memory
  oData[2 * tId + 1] = temp[2 * tId + 1];
}

__global__ void Ex2In(float *scan, float *iData, int n) {
  extern __shared__ float temp[]; // allocated via kernel config
  int tId = threadIdx.x;
  if (tId >= n)
    return;
  temp[tId] = scan[tId]; // load scan data;
  __syncthreads();

  if (tId > 0)
    scan[tId - 1] = temp[tId];

  if (tId == n - 1)
    scan[tId] = temp[tId] + iData[tId]; // last element clean up!
}
__global__ void PDF(float histoSum,float *histo, float *outPDF)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x; 
	outPDF[tid] = histo[tid]/histoSum;
}

__global__ void histogram_smem_atomics(const float *in, float range, float min, float *out)
{
    // initialize temporary accumulation array in shared memory
    extern __shared__ unsigned int smem[];
    
    // linear thread index within linear block
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    if(tid< NUM_BINS)
    {
	smem[tid] = 0;
    }
    
    __syncthreads();

    int bin = static_cast<int>(((in[t]-min)/(range))*1023);
    atomicAdd(&smem[bin],1);

    __syncthreads();
    
    if(tid<NUM_BINS)
    {
	atomicAdd(&(out[tid]), smem[tid]);
    }
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

void Histo()
{

    std::vector<float> x(NUM_BINS, 0.f);
    //Initialization
    int n;
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

    //std::cout<< "Number of array elements: " << numElements << std::endl;
    //Load array data to gpu
    hipMemcpy(dData, numbers.data(), original, hipMemcpyHostToDevice);

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

    
    //Histo
    float *hHisto = new float[NUM_BINS];
    float *dHisto;
    hipMalloc((void**)&dHisto, NUM_BINS*sizeof(float)); 

    dim3 blockDimHisto(1024);
    dim3 gridDimHisto(n/blockDim.x);    

    histogram_smem_atomics<<<gridDimHisto, blockDimHisto,size>>>(dData, (maxValue-minValue), minValue, dHisto);
    

    hipMemcpy(hHisto, dHisto, NUM_BINS*sizeof(float), hipMemcpyDeviceToHost);
    
    float temp = 0;
    //for(int i = 0; i<1024; i++)
    //{
    	//std::cout << "bin "<< i<<": " << hHisto[i] << std::endl;
        //temp += i;
    //}

    
    ShmemReduceKernelSum<<<gridDim,blockDim,size>>>(dReduc,dHisto);
    hipMemcpy(&histoSum, dReduc, sizeof(float), hipMemcpyDeviceToHost);

    //PDF
    float *dPDF;
    float *hPDF = new float[NUM_BINS];
    hipMalloc(&dPDF, NUM_BINS*sizeof(float));
    PDF<<<gridDim,blockDim>>>(histoSum, dHisto ,dPDF);
    hipMemcpy(hPDF, dPDF, NUM_BINS*sizeof(float), hipMemcpyDeviceToHost);
    temp = 0;
    //for(int i = 0; i<1024; i++)
    //{
    	//std::cout << "bin "<< i<<": " << hPDF[i] << std::endl;
        //temp += i;
    //}
    
    //Sum scan PDF for CDF
    float *dCDF;
    float *hCDF = new float[NUM_BINS];
    float xBeg = 0.0 - 5 *1;
    float xEnd = 0.0 +5 *1;
    float deltaX = (xEnd-xBeg)/(float)NUM_BINS;
    float *dX;
    hipMalloc(&dCDF, NUM_BINS*sizeof(float));
    hipMalloc(&dX, NUM_BINS*sizeof(float));
    DataGen<<<2, NUM_BINS / 2 + 1>>>(dPDF, dX, xBeg, deltaX, 0.0, 1.0, NUM_BINS);
    BlScan<<<1,NUM_BINS/2,2*NUM_BINS*sizeof(float)>>>(dCDF,dPDF,NUM_BINS);
    Ex2In<<<1, NUM_BINS, NUM_BINS*sizeof(float)>>>(dCDF, dPDF, NUM_BINS);
    hipMemcpy(x.data(), dX, sizeof(float)*NUM_BINS, hipMemcpyDeviceToHost);
    hipMemcpy(hCDF, dCDF, sizeof(float)*NUM_BINS, hipMemcpyDeviceToHost);    
    temp = 0;
    //for(int i = 0; i<1024; i++)
   // {
    	//std::cout << "bin "<< i<<": " << hCDF[i] << std::endl;
        //temp += i;
   // }

    std::cout << "Histo sum: " << histoSum << std::endl; 
    std::cout << "Max: " << maxValue << std::endl;
    std::cout << "Min: " << minValue << std::endl;  


    std::ofstream myFile;
    myFile.open("cdf.dat");
    for (int aa = 0; aa < NUM_BINS; aa++) {
      myFile << x[aa] << '\t' << hPDF[aa] << '\t' << hCDF[aa] << "\n";
    }
    myFile << std::endl;

    // destroy file object
    myFile.close();
    
    //Free memory
    hipFree(dReduc);
    hipFree(dData);

}

/* Driver for the computation of pi. */
int main()
{
    Histo();    
    
}

