/*Matmul routine for AMS148, written by Steven Reeves, March 10 2018,
  major routines referenced from CUDA Programming Guide. */

#include <cstring>
#include <ctime>
#include <hip/hip_runtime.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>

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

  MatMulKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);
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
__global__ void MatMulKernel(GMat A, GMat B, GMat C) {
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


__global__ void FillMatRand(GMat A)
{
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   //A.elements[row * A.width +col] = rand() % 100;
}

__global__ void MatByFloat(GMat A, float b, GMat C)
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

void GEMM(const int alpha, const bool opA, const CMat A, const bool opB, const CMat B, const int beta, const CMat C, CMat D)
{
  // Load A, B, and C to device memory
  // Allocate Matrix D
  GMat dA(A.width, A.height);
  GMat dB(B.width, B.height);
  GMat dC(C.width, C.height);
  GMat dD(D.width, D.height);

  CMat addResult(C.width, C.height);
  GMat dAddResult(C.width, C.height);

  CMat AbyB(A.width, B.height);
  GMat dAbyB(A.width, B.height);

  CMat CbyBeta(C.width, C.height);
  GMat dCbyBeta(C.width, C.height);


  dA.load(A);
  dB.load(B);
  dC.load(C);

  
 
  // D = alpha * opA (A) * opB (B) + beta(C)
  // Multiply A and B
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(dA, dB, dAbyB);

  D.load(dAbyB);
  for (int i = 0; i < A.width; i++) 
    {
    for (int j = 0; j < B.height; j++) {
       std::cout << D.elements[i * D.width + j];
       std::cout << " ";
    }
    std::cout << std::endl;
  }
   std::cout << std::endl;
   std::cout << std::endl;
  
  // Multiply C by beta
  dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid2(C.width / dimBlock.x, C.height / dimBlock.y);
  MatByFloat<<<dimGrid2, dimBlock2>>>(dC, beta, dCbyBeta);



  D.load(dCbyBeta);
  for (int i = 0; i < A.width; i++) 
    {
    for (int j = 0; j < B.height; j++) {
       std::cout << D.elements[i * D.width + j];
       std::cout << " ";
    }
    std::cout << std::endl;
  }
   std::cout << std::endl;
   std::cout << std::endl;

  // Add left and right side of equ
  dim3 dimBlock3(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid3(CbyBeta.width / dimBlock.x, AbyB.height / dimBlock.y);
  MatAdd<<<dimGrid3, dimBlock3>>>(dAbyB, dCbyBeta, dAddResult);

  //C.load(dC);
  D.load(dAddResult);

  for (int i = 0; i < A.width; i++) {
    for (int j = 0; j < B.height; j++) {
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
  int N = 16;
  int M = 32;
  int K = 32;

  CMat A(N, M, 1.f), B(M, K, 1.f), C(N, K,1.f), D(N,K);
  CMat nC(N, N);

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


     GEMM(1,false,A,false,B,5,C,D);
  // Naive HIP
  //NaiveMatMul(A, B, nC);

  // With LDS
  //MatMul(A, B, C);
}
