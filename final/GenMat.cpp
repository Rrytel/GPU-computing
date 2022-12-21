/*Matmul routine for AMS148, written by Steven Reeves, March 10 2018,
  major routines referenced from CUDA Programming Guide. */

#include <cstring>
#include <ctime>
#include <hip/hip_runtime.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <numeric>

/* Use Matrix Class! */
#include "mat.h"
#include "submat.h"

// Thread block size
#define BLOCK_SIZE 16


using GMat = GpuMatrix;
using CMat = CpuMatrix;

// Forward declaration of the mat mul kernel
__global__ void MatMulKernel(const GMat A, const GMat B, GMat C);
__global__ void NaiveKernel(const GMat A, const GMat B, GMat C);

void PrintMatrix(CMat A);

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

// Matrix Multiplication Kernel
__global__ void MatMulKernel(GMat A, GMat B, GMat C) {

   int trow = blockIdx.y * blockDim.y + threadIdx.y;
   int tcol = blockIdx.x * blockDim.x + threadIdx.x;

   if(trow > A.height || tcol > B.width)
   {
	//return;
   }

   // Static shared memory for Asub and Bsub
  __shared__ float aS[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float bS[BLOCK_SIZE][BLOCK_SIZE]; // Great name for an array

  // Block row and column;
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Thread block computes one sub matrix Csub of C
  subMatrix cSub(C, BLOCK_SIZE, blockRow, blockCol);

  // Each thread computes one element of Csub
  // By accumulating results into Cvalue
  float cValue = 0.0f;

  // Thread row and column index within the submatrix
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Loop over submatrices of A and B that are required for Csub
  // Multiply each pair of sub-matrices together
  // and summ the results
  for (int m = 0; m < (A.width / BLOCK_SIZE); m++) {

    // Get A submatrix
    subMatrix aSub(A, BLOCK_SIZE, blockRow, m);

    // Get B submatrix
    subMatrix bSub(B, BLOCK_SIZE, m, blockCol);

    // Load Asub and Bsub from global memory into shared;

    aS[row][col] = aSub.GetElem(row, col);
    bS[row][col] = bSub.GetElem(row, col);

    // Always sync threads when loading shared memory before doing computation
    __syncthreads();

    // Multiply the submatrices
    for (int e = 0; e < BLOCK_SIZE; e++)
      cValue += aS[row][e] * bS[e][col];

    // synchronize to make sure all threads are done computing
    __syncthreads();
  }
  // write Csub back into global memory
  // each thread writes one element
  cSub.SetElem(row, col, cValue);
}

__global__ void NaiveKernel(const GMat A, const GMat B, GMat C) {
  // Each Thread computes one element of C
  // by accumulating results into Cvalue
  float cValue = 0.0f;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < A.width; e++)
    cValue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
  C.elements[row * C.width + col] = cValue;
}


__global__ void MatByFloat(GMat A, float b, GMat C)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   C.elements[row * C.width + col] = A.elements[row * A.width +col] * b;

}

__global__ void MatByDouble(GMat A, double b, GMat C)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   C.elements[row * C.width + col] = A.elements[row * A.width +col] * b;

}

__global__ void MatAdd(GMat A, GMat B, GMat C)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   C.elements[row * C.width + col] = A.elements[row * A.width + col] + B.elements[row * B.width + col];

}

__global__ void Transpose(GMat A, GMat C)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   C.elements[row * C.width + col] =  A.elements[col * A.width + row];
}

__global__ void GpuCopy(GMat A, GMat C)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   C.elements[row * C.width + col] =  A.elements[row * A.width + col];
}

void GEMM(const int alpha, const bool opA, const CMat A, const bool opB, const CMat B, const int beta, const CMat C, CMat &D)
{


  // Load A, B, and C to device memory
  // Allocate Matrix D
  GMat dA(A.width, A.height);
  GMat dB(B.width, B.height);
  GMat dC(C.width, C.height);
  GMat dD(D.width, D.height);

  dA.load(A);
  dB.load(B);
  dC.load(C);

  CMat addResult(C.width, C.height);
  GMat dAddResult(C.width, C.height);

  //CMat AbyB(B.width,A.height);
  //GMat dAbyB(B.width,A.height);

  CMat CbyBeta(C.width, C.height);
  GMat dCbyBeta(C.width, C.height);

  int aTransHeight;
  int aTransWidth;
  if(opA)
  {
     aTransHeight = A.width;
     aTransWidth = A.height;
  }
  else
  {
     aTransHeight = A.height;
     aTransWidth = A.width;
  }

  int bTransHeight;
  int bTransWidth;
  if(opB)
  {
     bTransHeight = B.width;
     bTransWidth = B.height;
  }
  else
  {
     bTransHeight = B.height;
     bTransWidth = B.width;
  }
  
  CMat AbyB(bTransWidth, aTransHeight);
  GMat dAbyB(bTransWidth, aTransHeight);	

  CMat aT(aTransWidth,aTransHeight);
  GMat dAt(aTransWidth,aTransHeight);

  CMat bT(bTransWidth,bTransHeight);
  GMat dBt(bTransWidth,bTransHeight);

 
  // D = alpha * opA (A) * opB (B) + beta(C)

  // A transpose
  if(opA)
  {
     //Transpose A into aT

     //PrintMatrix(A);
     dim3 dimBlockTransA(BLOCK_SIZE,BLOCK_SIZE);
     dim3 dimGridTransA(aT.width/dimBlockTransA.x, aT.height/dimBlockTransA.y);
     Transpose<<<dimGridTransA,dimBlockTransA>>>(dA,dAt);
     aT.load(dAt);
     //std::cout << "Transpose" << std::endl;
     //PrintMatrix(aT);
  }
  else
  {
     //Copy A into aT
     dim3 dimBlockTransA(BLOCK_SIZE,BLOCK_SIZE);
     dim3 dimGridTransA(A.width/dimBlockTransA.x, A.height/dimBlockTransA.y);
     GpuCopy<<<dimGridTransA, dimBlockTransA>>>(dA,dAt);
     aT.load(dAt);
  }


  // B transpose
  if(opB)
  {
     //Transpose B into bT
     dim3 dimBlockTransB(BLOCK_SIZE,BLOCK_SIZE);
     dim3 dimGridTransB(bT.width/dimBlockTransB.x, bT.height/dimBlockTransB.y);
     Transpose<<<dimGridTransB,dimBlockTransB>>>(dB,dBt);
     bT.load(dBt);
  }
  else
  {
     //Copy B into bT
     dim3 dimBlockTransB(BLOCK_SIZE,BLOCK_SIZE);
     dim3 dimGridTransB(B.width/dimBlockTransB.x, B.height/dimBlockTransB.y);
     GpuCopy<<<dimGridTransB, dimBlockTransB>>>(dB,dBt);
     bT.load(dBt);

     //PrintMatrix(B);
  }

  // Multiply A by alpha
  dim3 dimBlock0(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid0(aT.width/dimBlock0.x, aT.height/dimBlock0.y);
  MatByFloat<<<dimGrid0,dimBlock0>>>(dAt,alpha,dAt);
  aT.load(dAt);

  //std::cout << "A * Alpha" << std::endl;
  //PrintMatrix(aT);

  // Multiply A and B
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(AbyB.width/dimBlock.x, AbyB.height/dimBlock.y);
  
  MatMulKernel<<<dimGrid, dimBlock>>>(dAt, dBt, dAbyB);
  //NaiveKernel<<<dimGrid, dimBlock>>>(dAt,dBt, dAbyB);


  AbyB.load(dAbyB);
  //std::cout<< "A * B" << std::endl;
  //PrintMatrix(AbyB);
  
  // Multiply C by beta
  dim3 dimBlock2(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid2(C.width / dimBlock2.x, C.height / dimBlock2.y);
  MatByFloat<<<dimGrid2, dimBlock2>>>(dC, beta, dCbyBeta);



  CbyBeta.load(dCbyBeta);
  //std::cout << "C * beta" << std::endl;
  //PrintMatrix(CbyBeta);

  
  // Add left and right side of equ
  dim3 dimBlock4(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid4(CbyBeta.width / dimBlock4.x, AbyB.height / dimBlock4.y);
  MatAdd<<<dimGrid4, dimBlock4>>>(dAbyB, dCbyBeta, dD);
 
  D.load(dD);
  /*std::cout << "D = " << std::endl;
  for (int i = 0; i < D.height; i++) {
    for (int j = 0; j < D.width; j++) {
       if(D.elements[i*D.width+j]!=0){
         std::cout << D.elements[i * D.width + j];
         std::cout << " ";
	}
       
     }
     if(D.elements[i*D.width] == 0)
     {

     }
     else
     {
       std::cout << std::endl;
     }
    
  }
  */

}

void NaiveMatMul(const CMat A, const CMat B, CMat C) {
  // Load A and B to device memory
  GMat dA(A.width, A.height);
  dA.load(A);
  GMat dB(B.width, B.height);
  dB.load(B);

  // Allocate C in device memory
  GMat dC(C.width, C.height);

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

  // Use hipEvent type for timing

  hipEvent_t start, stop;
  float elapsedSecs;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  NaiveKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);
  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&elapsedSecs, start, stop);
  std::cout << " Naive GPU MatMul Time = " << elapsedSecs << "ms" << std::endl;
  // Read C from device memory
  C.load(dC);
  // Free device memory
  dA.deAllocate();
  dB.deAllocate();
  dC.deAllocate();
}

void SerialMatMul(const CMat A, const CMat B, CMat &C) {
  for (int i = 0; i < A.height; i++) {
    for (int j = 0; j < B.width; j++) {
      float cValue = 0.0f;
      for (int k = 0; k < A.width; k++)
        cValue += A.elements[i * A.width + k] * B.elements[k * B.width + j];
      C.elements[i * C.width + j] = cValue;
    }
  }
}

void CPUPad(CMat &A, int size)
{
  CMat temp(A.width, A.height);
  temp.copy(A);
  A.width = size;
  A.height = size;
  A.resize();

  for(int row=0; row<size; row++)
  {
    for(int col=0; col<size; col++)
    {
      if(row > temp.height-1 || col > temp.width-1)
      {
        A.elements[row*A.width + col] = 0;
      }
      else
      {
        A.elements[row*A.width + col] = temp.elements[row*temp.width+col];
      }
    }
  }

}

void PrintMatrix(CMat A)
{
  for (int i = 0; i < A.height; i++) {
    for (int j = 0; j < A.width; j++) {
      std::cout << A.elements[i * A.width + j];
      std::cout << " ";
     }
     std::cout << std::endl;
   }
   std::cout << std::endl;
   std::cout << std::endl;
}

// Main program
int main() {
  // Set up matrices
  int temp = 2048;
  int N=temp,M=temp,K=temp;

  CMat A(N,M, 1.f), B(K, N, 1.f), C(K, M,1.f), D(K,M);
  CMat testA(N,M,4.f),testB(K,N,1.f), testD(K,M);

  //SerialMatMul(testA,testB,testD);
 
  //Pad matricies with zeros
  int padNum = std::max(N,std::max(N,K));
  while(padNum%BLOCK_SIZE!=0)
  {
	padNum++;
  }
  CPUPad(A,padNum);
  CPUPad(B,padNum);
  CPUPad(C,padNum);
  CPUPad(D,padNum);



  // Use hipEvent type for timing

  hipEvent_t start, stop;
  float elapsedSecs;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);  
  GEMM(1,false,A,false,B,1,C,D);
  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&elapsedSecs, start, stop);
  std::cout << "GEMM Time = " << elapsedSecs << "ms" << std::endl;

  return 0;

}
