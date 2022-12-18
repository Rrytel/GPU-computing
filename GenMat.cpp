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

// Matrix multiplication host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

/* Shared Matrix Multiplication Routines */

/* MatMul with shared memory
   :inputs: Matrix A, Matrix B
   :outputs: Matrix C = AB
 */

int gcd(int a, int b) 
{
   if (b == 0)
   return a;
   return gcd(b, a % b);
}

void MatMul(const CMat A, const CMat B, CMat C) {
  int Gpu = 1;
  // Load A and B to device memory
  // Allocate Matrix C
  GMat dA(A.width, A.height);
  GMat dB(B.width, B.height);
  GMat dC(C.width, C.height);
  dA.load(A);
  dB.load(B);

  // Invoke Kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  // Use HIP Events for timing
  hipEvent_t start, stop;
  float time;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  //MatMulKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);
  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&time, start, stop);
  std::cout << " Shared Memory Matrix Multiplication time =" << '\t' << time
            << "ms" << std::endl;

  // Read C from Device memory
  C.load(dC);

  // Free device memory
  dA.deAllocate();
  dB.deAllocate();
  dC.deAllocate();
}

// Matrix Multiplication Kernel
__global__ void MatMulKernel(GMat A, GMat B, GMat C, int blockSize) {


  int tcol = blockIdx.x * blockDim.x + threadIdx.x;
  int trow = blockIdx.y * blockDim.y + threadIdx.y;
  if (!(trow < A.height && tcol < B.width))
  {
	//return;
  }
  // Static shared memory for Asub and Bsub
  extern __shared__  float aS[];
  extern __shared__  float bS[]; // Great name for an array

  // Block row and column;
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Thread block computes one sub matrix Csub of C
  subMatrix cSub(C, blockSize, blockRow, blockCol);

  // Each thread computes one element of Csub
  // By accumulating results into Cvalue
  float cValue = 0.0f;

  // Thread row and column index within the submatrix
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Loop over submatrices of A and B that are required for Csub
  // Multiply each pair of sub-matrices together
  // and summ the results
  for (int m = 0; m < (A.width / blockSize); m++) {

    // Get A submatrix
    subMatrix aSub(A, blockSize, blockRow, m);

    // Get B submatrix
    subMatrix bSub(B, blockSize, m, blockCol);

    // Load Asub and Bsub from global memory into shared;

    aS[row * blockSize + col] = aSub.GetElem(row, col);
    bS[row * blockSize + col] = bSub.GetElem(row, col);

    // Always sync threads when loading shared memory before doing computation
    __syncthreads();

    // Multiply the submatrices
    for (int e = 0; e < blockSize; e++)
      cValue += aS[row * blockSize + e] * bS[e * blockSize + col];

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

void GEMM(const int alpha, const bool opA, const CMat A, const bool opB, const CMat B, const int beta, const CMat C, CMat D)
{


  // Load A, B, and C to device memory
  // Allocate Matrix D
  GMat dA(A.width, A.height);
  GMat dB(B.width, B.height);
  GMat dC(C.width, C.height);
  GMat dD(D.width, D.height);


  //CMat aT(A.height,A.width);
  //GMat dAt(A.height,A.width);

  CMat addResult(C.width, C.height);
  GMat dAddResult(C.width, C.height);

  CMat AbyB(A.height, B.height);
  GMat dAbyB(A.height, B.height);

  CMat CbyBeta(C.width, C.height);
  GMat dCbyBeta(C.width, C.height);


  dA.load(A);
  dB.load(B);
  dC.load(C);


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

  CMat aT(aTransWidth,aTransHeight);
  GMat dAt(aTransWidth,aTransHeight);

  CMat bT(bTransWidth,bTransHeight);
  GMat dBt(bTransWidth,bTransHeight);

 
  // D = alpha * opA (A) * opB (B) + beta(C)

  // A transpose
  if(opA)
  {
      for (int i = 0; i < A.height; i++) 
      {
       for (int j = 0; j < A.width; j++) {
        std::cout << A.elements[i * A.width + j];
        std::cout << " ";
      }
      std::cout << std::endl;
     }
     std::cout << std::endl;
     std::cout << std::endl;


     dim3 dimBlockTransA(gcd(aT.width,aT.height),gcd(aT.width,aT.height));
     dim3 dimGridTransA(aT.width/dimBlockTransA.x, aT.height/dimBlockTransA.y);
     Transpose<<<dimGridTransA,dimBlockTransA>>>(dA,dAt);
     aT.load(dAt);

     std::cout << "Transpose" << std::endl;
     for (int i = 0; i < aT.height; i++) 
     {
       for (int j = 0; j < aT.width; j++) {
        std::cout << aT.elements[i * aT.width + j];
        std::cout << " ";
      }
      std::cout << std::endl;
     }
     std::cout << std::endl;
     std::cout << std::endl;
  }
  else
  {
     dim3 dimBlockTransA(gcd(A.width,A.height),gcd(A.width,A.height));
     dim3 dimGridTransA(A.width/dimBlockTransA.x, A.height/dimBlockTransA.y);
     GpuCopy<<<dimGridTransA, dimBlockTransA>>>(dA,dAt);
     aT.load(dAt);
  }


  // B transpose
  if(opB)
  {

     //CMat bT(B.height,B.width);
     //GMat dBt(B.height,B.width);

     dim3 dimBlockTransB(gcd(bT.width,bT.height),gcd(bT.width,bT.width));
     dim3 dimGridTransB(bT.width/dimBlockTransB.x, bT.height/dimBlockTransB.y);
     Transpose<<<dimGridTransB,dimBlockTransB>>>(dB,dBt);
     bT.load(dBt);
  }
  else
  {
     //CMat bT(B.width,B.height);
     //GMat dBt(B.width,B.height);
     dim3 dimBlockTransB(gcd(B.width,B.height),gcd(B.width,B.height));
     dim3 dimGridTransB(B.width/dimBlockTransB.x, B.height/dimBlockTransB.y);
     GpuCopy<<<dimGridTransB, dimBlockTransB>>>(dB,dBt);
     bT.load(dBt);

     std::cout << "B" << std::endl;
     for (int i = 0; i < bT.height; i++) 
     {
       for (int j = 0; j < bT.width; j++) {
        std::cout << bT.elements[i * bT.width + j];
        std::cout << " ";
      }
      std::cout << std::endl;
     }
     std::cout << std::endl;
     std::cout << std::endl;
  }

  // Multiply A by alpha
  int blockSize = gcd(aT.width, aT.height);
  dim3 dimBlock0(blockSize, blockSize);
  
  //dim3 dimBlock0(gcd(aT.width,aT.height),gcd(aT.width,aT.height));
  dim3 dimGrid0(aT.width/dimBlock0.x, aT.height/dimBlock0.y);
  
  MatByFloat<<<dimGrid0,dimBlock0>>>(dAt,alpha,dAt);
  aT.load(dAt);

  std::cout << dimBlock0.x<< " " << dimGrid0.x << " " << dimGrid0.y << std::endl;
  std::cout << "alpha* A" << std::endl;
  for (int i = 0; i < aT.height; i++) 
    {
    for (int j = 0; j < aT.width; j++) {
       std::cout << aT.elements[i * aT.width + j];
       std::cout << " ";
    }
    std::cout << std::endl;
  }
   std::cout << std::endl;
   std::cout << std::endl;


  // Multiply A and B

  
  blockSize = gcd(AbyB.width, AbyB.height);
  size_t size = blockSize * blockSize * sizeof(float);
  dim3 dimBlock(blockSize,blockSize);
  dim3 dimGrid(AbyB.width / dimBlock.x, AbyB.height/ dimBlock.y);
  
  
  MatMulKernel<<<dimGrid, dimBlock, size>>>(dAt, dBt, dAbyB,blockSize);

  std::cout << aT.width << " " << bT.height<< std::endl;
  std::cout << dimBlock.x<< " " << dimGrid.x << " " << dimGrid.y << std::endl;

  AbyB.load(dAbyB);
  std::cout << "A * B" << std::endl;
  for (int i = 0; i < AbyB.height; i++) 
    {
    for (int j = 0; j < AbyB.width; j++) {
       std::cout << AbyB.elements[i * AbyB.width + j];
       std::cout << " ";
    }
    std::cout << std::endl;
  }
   std::cout << std::endl;
   std::cout << std::endl;
  
  // Multiply C by beta
  dim3 dimBlock2(gcd(C.width,C.height),gcd(C.width,C.height));
  dim3 dimGrid2(C.width / dimBlock.x, C.height / dimBlock.y);
  MatByFloat<<<dimGrid2, dimBlock2>>>(dC, beta, dCbyBeta);



  D.load(dCbyBeta);
  
  for (int i = 0; i < D.height; i++) 
    {
    for (int j = 0; j < D.width; j++) {
       std::cout << D.elements[i * D.width + j];
       std::cout << " ";
    }
    std::cout << std::endl;
  }
   std::cout << std::endl;
   std::cout << std::endl;


  // Add left and right side of equ
  dim3 dimBlock4(gcd(AbyB.width,CbyBeta.height),gcd(AbyB.width,CbyBeta.height));
  dim3 dimGrid4(CbyBeta.width / dimBlock4.x, AbyB.height / dimBlock4.y);
  MatAdd<<<dimGrid4, dimBlock4>>>(dAbyB, dCbyBeta, dD);
 

  //C.load(dC);
  D.load(dD);
  std::cout << "D = " << std::endl;
  for (int i = 0; i < D.height; i++) {
    for (int j = 0; j < D.width; j++) {
       std::cout << D.elements[i * D.width + j];
       std::cout << " ";
    }
    std::cout << std::endl;
  }

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

void SerialMatMul(const CMat A, const CMat B, CMat C) {
  for (int i = 0; i < A.width; i++) {
    for (int j = 0; j < B.height; j++) {
      float cValue = 0.0f;
      for (int k = 0; k < A.width; k++)
        cValue += A.elements[i * A.width + k] * B.elements[k * B.width + j];
      C.elements[i * C.width + j] = cValue;
    }
  }
}

void CPUMatMul(const CMat A, const CMat B, CMat C) {
  int i, j, k;
#pragma omp parallel for private(j, k)
  for (i = 0; i < A.width; i++) {
    for (j = 0; j < B.height; j++) {
      float cValue = 0.0f;
      for (k = 0; k < A.width; k++) {
        cValue += A.elements[i * A.width + k] * B.elements[k * B.width + j];
      }
      C.elements[i * C.width + j] = cValue;
    }
  }
}

// Main program
int main() {
  // Set up matrices
  int Cpu = 0;
  int N = 32;
  int M = 16;
  int K = 4;

  //CMat A(N, M, 2.f), B(M, K, 1.f), C(N, K,1.f), D(N,K);

  CMat A(N,M, 2.f), B(M, N, 1.f), C(M, M), D(M,N);
  //CMat A(2, 4, 2.f), B(4, 2, 1.f), C(4, 4,1.f), D(4,4);
  CMat nC(N, N);
  CMat test(32,16,5.f);

  //for (int i = 0; i < test.height; i++) {
    //for (int j = 0; j < test.width; j++) {
       //std::cout << test.elements[i * test.width + j];
       //std::cout << " ";
    //}
    //std::cout << std::endl;
  //}

// Call matrix multiplication.
#if 0
    CMat Ds(N, N), D(N,N);
//Serial 
	clock_t sstart = clock();	//Serial Start
	serialMatMul(A,B,Ds);
	clock_t send = clock(); 	//Serial End
	double serial = double(send - sstart) / CLOCKS_PER_SEC;	
	std::cout<< " Serial Time = " << serial << "s" << std::endl;
//OpenMP
	clock_t begin = clock();	
	CPUMatMul(A,B,D);
	clock_t end = clock();
	double fullcpu = double(end - begin) / (CLOCKS_PER_SEC*12);
	std::cout<< " CPU Time = " << fullcpu << "s" << std::endl;
#endif


     //CPUMatMul(A,B,D);
     //std::cout << "CPU" << std::endl;
     //for (int i = 0; i < D.height; i++) {
        //for (int j = 0; j < D.width; j++) {
       //std::cout << D.elements[i * D.width + j];
       //std::cout << " ";
    //}
    //std::cout << std::endl;
  //}
     
     GEMM(2,false,A,false,B,5,C,D);
  // Naive HIP
  //NaiveMatMul(A, B, nC);

  // With LDS
  //MatMul(A, B, C);
}

void printMatrix(CMat A)
{


}
