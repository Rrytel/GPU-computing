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

__global__ void BlurrKernel(const GMat3 In, const GMat Filter, GMat3 Out);


__global__ void BlurrKernel(const GMat3 In, const GMat Filter, GMat3 Out)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float3 pixelValue;
    pixelValue.x = 0.0f;
    pixelValue.y = 0.0f;
    pixelValue.z = 0.0f;
    if(row == 0 || row == In.height-1 || col == 0 || col == In.width-1)
    {
       pixelValue.x = In.elements[row*In.width+col].x;
       pixelValue.y = In.elements[row*In.width+col].y;
       pixelValue.z = In.elements[row*In.width+col].z;
    }
    else
    {
      for(int u = -1; u < 2; u ++)
      {
	for(int v = -1; v < 2; v++)
        {
           pixelValue.x += In.elements[(row+u)*In.width+(col+v)].x*Filter.elements[(u+1)*Filter.width+(v+1)];
           pixelValue.y += In.elements[(row+u)*In.width+(col+v)].y*Filter.elements[(u+1)*Filter.width+(v+1)];
           pixelValue.z += In.elements[(row+u)*In.width+(col+v)].z*Filter.elements[(u+1)*Filter.width+(v+1)];
        }
      }

    }
    
    Out.elements[row*Out.width+col].x = pixelValue.x;
    Out.elements[row*Out.width+col].y = pixelValue.y;
    Out.elements[row*Out.width+col].z = pixelValue.z;
}

/*
    Kernel: Fills f_new with the contents of f_old
    Inputs: FP32 array f_old, FP32 array f_new
    Output: FP32 array f_new
*/
__global__ void Equate(float fOld[], const float fNew[]) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  fOld[tId] = fNew[tId];
}

void IoFun(std::string file, std::vector<float> x, std::vector<float> f) {
  std::ofstream myFileTsN;
  myFileTsN.open(file);
  for (int i = 0; i < M; i++) {
    myFileTsN << x[i] << '\t';
    myFileTsN << f[i] << std::endl;
  }

  myFileTsN.close();
}


void Blurr(const CMat3 A, const CMat B, CMat3 C) {
  // Load A and B to device memory
  GMat3 dA(A.width, A.height);
  dA.load(A);
  GMat dB(B.width, B.height);
  dB.load(B);

  // Allocate C in device memory
  GMat3 dC(C.width, C.height);

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);

  // Use hipEvent type for timing
  std::cout << A.elements[0].x<< std::endl;

  BlurrKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);

  // Read C from device memory
  C.load(dC);
  
  for(int i =0; i < C.width; i++)
  {
    for(int j = 0; j < C.height; j++)
    {
     std::cout << C.elements[i*C.width+j].x<< "  ";
    }
    std::cout<< std::endl;
  }
  
  // Free device memory
  dA.deAllocate();
  dB.deAllocate();
  dC.deAllocate();
}

/*
    Driver function to simulate the heat profile in a 1-D bar.
*/
int main() {


   CImg<unsigned char> src("ana_noise.bmp");
   int width = src.width();
   int height = src.height();
   size_t size = src.size();//width*height*sizeof(unsigned char);

   unsigned char *h_src = src.data();

   
   std::cout << "Main " << h_src[0] << std::endl;


   CMat3 Image(48,48);
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
   CMat3 Output(48,48);
   Blurr(Image,Filter,Output);

}
