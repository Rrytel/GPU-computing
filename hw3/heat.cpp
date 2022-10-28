#include <cmath>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "mat.h"

#define M 1024
#define BLOCK_SIZE 16
using GMat = GpuMatrix;
using CMat = CpuMatrix;

__global__ void BlurrKernel(const GMat In, const GMat Filter, GMat Out);


__global__ void BlurrKernel(const GMat In, const GMat Filter, GMat Out)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float pixelValue = 0.0f;
    if(row == 0 || row == In.height-1 || col == 0 || col == In.width-1)
    {
       pixelValue = In.elements[row*In.width+col];
    }
    else
    {
      for(int u = -1; u < 2; u ++)
      {
	for(int v = -1; v < 2; v++)
        {
           pixelValue+= In.elements[(row+u)*In.width+(col+v)]*Filter.elements[(u+1)*Filter.width+(v+1)];
        }
      }

    }
    
    
    Out.elements[row*Out.width+col] = pixelValue;

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


void Blurr(const CMat A, const CMat B, CMat C) {
  // Load A and B to device memory
  GMat dA(A.width, A.height);
  dA.load(A);
  GMat dB(B.width, B.height);
  dB.load(B);

  // Allocate C in device memory
  GMat dC(C.width, C.height);

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);

  // Use hipEvent type for timing
  std::cout << A.elements[0]<< std::endl;

  BlurrKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);

  // Read C from device memory
  C.load(dC);
  
  for(int i =0; i < C.width; i++)
  {
    for(int j = 0; j < C.height; j++)
    {
     std::cout << C.elements[i*C.width+j]<< "  ";
    }
    std::cout<< std::endl;
  }
  std::cout << C.elements[4*C.width+5]<<std::endl;
  // Free device memory
  dA.deAllocate();
  dB.deAllocate();
  dC.deAllocate();
}

/*
    Driver function to simulate the heat profile in a 1-D bar.
*/
int main() {

   CMat Image(48,48, 5.f);
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
   CMat Output(48,48,0.f);
   Blurr(Image,Filter,Output);

}
