
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
using GVec = GpuVector;
using CVec = CpuVector;

// Forward declaration of the mat mul kernel
__global__ void NaiveVecKernel(const GMat A, const GVec B, GVec C);
__global__ void MatVecKernel(const GMat A, const GVec B, GVec C);

// Matrix multiplication host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE

/* Shared Matrix Multiplication Routines */
void MatVec(const CMat A, const CVec B, CVec C) {
  int Gpu = 1;
  // Load A and B to device memory
  // Allocate Matrix C
  GMat dA(A.width, A.height);
  GVec dB(B.width);
  GVec dC(C.width);
  dA.load(A);
  dB.load(B);

  // Invoke Kernel
  //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE);
  //dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  dim3 dimGrid(B.width / dimBlock.x);
  // Use HIP Events for timing
  hipEvent_t start, stop;
  float time;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  MatVecKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);
  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&time, start, stop);
  std::cout << " Shared Memory Matrix Vector Multiplication time =" << '\t' << time
            << "ms" << std::endl;

  // Read C from Device memory
  C.load(dC);

  std::cout <<"First element of LDS MatVec"<<":" <<C.elements[0] << std::endl;
  
  // Free device memory
  dA.deAllocate();
  dB.deAllocate();
  dC.deAllocate();
}
//Matrix vector muliplication kernel
__global__ void MatVecKernel(const GMat A, const GVec B, GVec C){
  // Static shared memory for Asub and Bsub
  __shared__ float aS[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float bS[BLOCK_SIZE];

  // Block row and column;
  
  int blockRow = blockIdx.x;

  // Thread block computes one sub matrix Csub of C
  subVector cSub(C, BLOCK_SIZE, blockRow);

  // Each thread computes one element of Csub
  // By accumulating results into Cvalue
  float cValue = 0.0f;

  // Thread row and column index within the submatrix
  int row = threadIdx.x;

  // Loop over submatrices of A and B that are required for Csub
  // Multiply each pair of sub-matrices together
  // and summ the results
  
  for (int m = 0; m < (A.width / BLOCK_SIZE); m++) {

    // Get A submatrix
    subMatrix aSub(A, BLOCK_SIZE, blockRow, m);

    // Get B submatrix
    subVector bSub(B, BLOCK_SIZE, m);

    // Load Asub and Bsub from global memory into shared;
    for(int i = 0; i < BLOCK_SIZE; i++)
    {
       aS[row][i] = aSub.GetElem(row, i);
    }
 
    bS[row] = bSub.GetElem(row);
    
    // Always sync threads when loading shared memory before doing computation
    __syncthreads();

    // Multiply the submatrices
    for (int e = 0; e < BLOCK_SIZE; e++)
      cValue += aS[row][e] * bS[e];

    // synchronize to make sure all threads are done computing
    __syncthreads();
  }
  // write Csub back into global memory
  // each thread writes one element
  cSub.SetElem(row, cValue);  
}
//Matrix vector naive kernel
__global__ void NaiveVecKernel(const GMat A, const GVec B, GVec C) {
  // Each Thread computes one element of C
  // by accumulating results into Cvalue
  float cValue = 0.0f;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  //int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < A.width; e++)
    cValue += A.elements[row * A.width + e] * B.elements[e];
  C.elements[row] = cValue;
}

void NaiveMatVec(const CMat A, const CVec B, CVec C) {
  // Load A and B to device memory
  GMat dA(A.width, A.height);
  dA.load(A);
  GVec dB(B.width);
  dB.load(B);

  // Allocate C in device memory
  GVec dC(C.width);

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x);

  // Use hipEvent type for timing

  hipEvent_t start, stop;
  float elapsedSecs;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  NaiveVecKernel<<<dimGrid, dimBlock>>>(dA, dB, dC);
  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&elapsedSecs, start, stop);
  std::cout << " Naive GPU MatVec Time = " << elapsedSecs << "ms" << std::endl;
  // Read C from device memory
  C.load(dC);


  std::cout <<"First element of naive MatVec"<<":" <<C.elements[0] << std::endl;

  // Free device memory
  dA.deAllocate();
  dB.deAllocate();
  dC.deAllocate();
}

void CPUMatVec(const CMat A, const CVec B, CVec& C) {
  int i, j;
  for (i = 0; i < A.width; i++) {
    float cValue = 0.0f;
    for (j = 0; j < A.height; j++) {
      cValue += A.elements[i * A.width + j] * B.elements[j];
    }
    //std::cout <<i<<":" <<cValue << std::endl;
    C.elements[i] = cValue;
  }
}

// Main program
int main() {
  // Set up matrices
  int Cpu = 0;
  int N = 1024;
  int M = 1024;

  CMat A(N, M, 1.f), B(M, N, 1.f), C(N, N);
  CMat nC(N, N);

  //First test case
  CMat A2(128,128,1);
  CVec B2(128,2);
  CVec C2(128), nC2(128);
  
  clock_t sstart = clock();	//Serial Start
  CPUMatVec(A2,B2,nC2);
  clock_t send = clock(); 	//Serial End
  double serial = double(send - sstart) / CLOCKS_PER_SEC;	
  std::cout<< " Serial Time = " << serial*1000 << "ms" << std::endl;
  std::cout <<"First element of serial MatVec"<<":" <<nC2.elements[0] << std::endl;
 
  //NaiveMatVec(A2,B2,nC2);
  MatVec(A2,B2,nC2);
  std::cout << std::endl;

  
  //Second test case
  CMat A3(1024,1024);
  for (int i = 0; i < A3.width; i++) {
    for (int j = 0; j < A3.height; j++) {
      A3.elements[i * A3.width + j] = i;
    }
  }
  CVec B3(1024,1);
  CVec C3(1024), nC3(1024);

  
  sstart = clock();	//Serial Start
  CPUMatVec(A3,B3,nC3);
  send = clock(); 	//Serial End
  serial = double(send - sstart) / CLOCKS_PER_SEC;	
  std::cout<< " Serial Time = " << serial*1000 << "ms" << std::endl;
  std::cout <<"First element of serial MatVec"<<":" <<nC3.elements[0] << std::endl;
  
  NaiveMatVec(A3,B3,nC3);
  MatVec(A3,B3,nC3);
  std::cout << std::endl;
  //Third test case
  CMat A4(2048,2048,1);
  for (int i = 0; i < A4.width; i++) {
    for (int j = 0; j < A4.height; j++) {
      A4.elements[i * A4.width + j] = j;
    }
  }
  CVec B4(2048,1.5);
  CVec C4(2048), nC4(2048);

  sstart = clock();	//Serial Start
  CPUMatVec(A4,B4,nC4);
  send = clock(); 	//Serial End
  serial = double(send - sstart) / CLOCKS_PER_SEC;	
  std::cout<< " Serial Time = " << serial*1000 << "ms" << std::endl;
  std::cout <<"First element of serial MatVec"<<":" <<nC4.elements[0] << std::endl;
  
  NaiveMatVec(A4,B4,nC4);
  MatVec(A4,B4,nC4);

}
