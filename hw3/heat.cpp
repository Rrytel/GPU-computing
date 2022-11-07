#include <cmath>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "mat.h"
#include "CImg.h"

#define M 1024
#define BLOCK_SIZE 16
using GMat3 = GpuMatrix3;
using CMat3 = CpuMatrix3;
using GMat = GpuMatrix;
using CMat = CpuMatrix;
using namespace cimg_library;

__global__ void BlurrKernel(const unsigned char *in, int width, int height, const GMat Filter, unsigned char *out);
__global__ void L2Kernel(const unsigned char *blurImg, int widht, int height, const unsigned char *origImg, GMat L2);
__global__ void reduce(GMat dOut, GMat dIn);


__global__ void BlurrKernel(const unsigned char *in, int width, int height, const GMat Filter, unsigned char *out)
{
   //Image width by height
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   float3 pixelValue;
   pixelValue.x = 0.0f;
   pixelValue.y = 0.0f;
   pixelValue.z = 0.0f;
   //Edge of image
   if(row == 0 || row == height-1 || col == 0 || col == width-1)
   {
      //Load rgb pixel values for: input[row][col]
      pixelValue.x = (float)(in[row * width + col]);
      pixelValue.y = (float)(in[(row + height) * width + col]);
      pixelValue.z = (float)(in[(row + height*2) * width + col]);
       
   }
   else
   {
      for(int u = -1; u < 2; u ++)
      {
         for(int v = -1; v < 2; v++)
         {
            //Enumerate rgb pixel values for: input[row+u][col+v]*Filter[u+1][v+1]
            pixelValue.x += (float)(in[(row+u) * width + (col+v)])*Filter.elements[(u+1)*Filter.width+(v+1)];
            pixelValue.y += (float)(in[((row+u) + height) * width + (col+v)])*Filter.elements[(u+1)*Filter.width+(v+1)];
            pixelValue.z += (float)(in[((row+u) + height*2) * width + (col+v)])*Filter.elements[(u+1)*Filter.width+(v+1)];
         }
       }

   }
    
   //Load rgb values into output[row][col]
   out[row * width + col] = pixelValue.x;
   out[(row + height) * width + col] = pixelValue.y;
   out[(row + height*2) * width + col] = pixelValue.z;
}


__global__ void L2Kernel(const unsigned char *blurImg, int width, int height, const unsigned char *origImg, GMat L2)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float3 pixelValueB;
    pixelValueB.x = 0.0f;
    pixelValueB.y = 0.0f;
    pixelValueB.z = 0.0f;
    pixelValueB.x = (float)(blurImg[row * width + col]);
    pixelValueB.y = (float)(blurImg[(row + height) * width + col]);
    pixelValueB.z = (float)(blurImg[(row + height*2) * width + col]);

    float3 pixelValueO;
    pixelValueO.x = 0.0f;
    pixelValueO.y = 0.0f;
    pixelValueO.z = 0.0f;
    pixelValueO.x = (float)(origImg[row * width + col]);
    pixelValueO.y = (float)(origImg[(row + height) * width + col]);
    pixelValueO.z = (float)(origImg[(row + height*2) * width + col]);

    float temp = abs(pixelValueB.x - pixelValueO.x) + abs(pixelValueB.y - pixelValueO.y) + abs(pixelValueB.z - pixelValueO.z);
    temp = temp/3.f;
    temp *= temp;

    L2.elements[row*L2.width+col] = temp;

}


__global__ void reduce(GMat dOut, GMat dIn)
{
    extern __shared__ float sdata[];
    int myId = threadIdx.x + blockDim.x*blockIdx.x;
    int tid  = threadIdx.x;
    sdata[tid] = dIn.elements[myId];
    __syncthreads();
    for(int s = 1; s<blockDim.x; s*=2)
    {
	if(tid%(2*s)==0)
        {
	   sdata[tid] += sdata[tid+s];
        }
	__syncthreads();
    }
    if(tid == 0)
    {
        dOut.elements[blockIdx.x] = sdata[0];
    }


}


void L2Norm(const unsigned char *blurImg, int width, int height, const unsigned char *origImg)
{
   CMat CL2(width, height);
   GMat L2(width, height);
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   dim3 dimGrid(width / dimBlock.x+1, height / dimBlock.y);
   L2Kernel<<<dimGrid, dimBlock>>>(blurImg,width,height,origImg,L2);
   CL2.load(L2);
   float sum = 0;
   for(int i =0; i < CL2.width; i++)
   {
	for(int j =0; j<CL2.height; j++)
        {
           sum+= CL2.elements[i*CL2.width+j];
        }
   }
   std::cout << "Serial Sum: " << sqrt(sum) <<std::endl;
   CMat Creduced(1024,1);
   GMat reduced(1024,1);
   dim3 dimBlockReduce(1024);
   dim3 dimGridReduce((width*height+dimBlockReduce.x-1)/dimBlockReduce.x);
   reduce<<<dimGridReduce,dimBlockReduce,1024*sizeof(float)>>>(reduced,L2);
   reduce<<<1,dimBlockReduce,1024*sizeof(float)>>>(reduced,reduced);
   CL2.load(L2);
   Creduced.load(reduced);
   std::cout<< "L2: " <<sqrt(Creduced.elements[0])<<std::endl;
  

}


void Blurr(const unsigned char *in, int width, int height, const CMat F, unsigned char *out) {
    
   //Load filter on GPU
   GMat dF(F.width, F.height);
   dF.load(F);

   // Invoke kernel parameters
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE,1);
   dim3 dimGrid(width / dimBlock.x+1, height / dimBlock.y,1);

   //GPU kernel call
   BlurrKernel<<<dimGrid, dimBlock>>>(in, width, height,dF,out);

  // Free device memory
    dF.deAllocate();
}

int main() {

   CImg<unsigned char> src("ana_noise.bmp");
   //CImg<unsigned char> out("ana_noise.bmp");
   CImg<unsigned char> original("ana_sbk.bmp");
   int width = src.width();
   int height = src.height();
   size_t size = src.size();//width*height*sizeof(unsigned char);
   CImg<unsigned char> out(width, height,1,3,0);

    unsigned char *hSrc = src.data();
    unsigned char *hOut = out.data();
    unsigned char *hOrig = original.data();

    unsigned char *dSrc;
    unsigned char *dOut;
    unsigned char *dOrig;
    

    hipMalloc((void**)&dSrc, size);
    hipMalloc((void**)&dOut, size);
    hipMalloc((void**)&dOrig, size);

    hipMemcpy(dSrc, hSrc, size, hipMemcpyHostToDevice);
    hipMemcpy(dOut, hOut, size, hipMemcpyHostToDevice);
    hipMemcpy(dOrig, hOrig, size, hipMemcpyHostToDevice);

   //Create filter
   CMat Filter(3,3,0.f);
   Filter.elements[0]= .0625f;
   Filter.elements[1]= .125;
   Filter.elements[2]= .0625f;
   Filter.elements[3]= .125;
   Filter.elements[4]= .25;
   Filter.elements[5]= .125;
   Filter.elements[6]= .0625f;
   Filter.elements[7]= .125;
   Filter.elements[8]= .0625f;

   Blurr(dSrc,width,height,Filter,dOut);

   hipMemcpy(hOut, dOut, size, hipMemcpyDeviceToHost);
  
   //Save blurred image
   out.save("nonoise.bmp");

   std::cout << "Output vs original :";
   L2Norm(dOut,width, height, dOrig);
   std::cout << "Source vs original :";
   L2Norm(dSrc,width,height, dOrig);

}
