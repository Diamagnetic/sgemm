/*
 * Author: Chirag Dhamange
 * File Name: matmul.cu
 * Purpose: CUDA code for Standard GEneral Matrix Multiplication (SGEMM).
 *          The following techniques were used and performance was
 *          measured:
 *          1. Naive
 *
 * C = beta*C + alpha*A*B, where A, B, and C are matrices, and
 * alpha, and beta are scalars
 *
 * Referenced from: https://siboehm.com/articles/22/CUDA-MMM
*/

#include "matmul.cuh"

// The kernel performs SGEMM on two matrices
// A particular thread calculates the result of one cell in
// the resultant matrix
//
// C = beta*C + alpha*A*B, where A, B, and C are matrices, and
// alpha, and beta are scalars
__global__ void sgemm_naive(int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C
    )
{
  // access matrix entry (row = x, col = y) in matrix C
  const uint x = blockDim.x * blockIdx.x + threadIdx.x;
  const uint y = blockDim.y * blockIdx.y + threadIdx.y;

  if(x < M && y < N)
  {
    float tmp = 0.0;

    // multiply row = x of A with col = y of B
    for(int i = 0; i < K; i++)
      tmp += A[x * K + i] * B[i * N + y];

    C[x * N + y] *= beta;
    C[x * N + y] += alpha * tmp;
  }
}

// A wrapper function that sets the dimensions of the grid and block
// and executes the kernel
void run_mat_mul(const int &M, const int &N, const int &K,
    const int &alpha, float *A, float *B,
    const int &beta, float *C
    )
{
  dim3 gridDim(M / 32 + 1, N / 32 + 1, 1);
  dim3 blockDim(32, 32, 1);

  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}

// Allocate space for the matrices on the device
void alloc_space(float **A, float **B, float **C, const int &M,
    const int &N, const int &K
    )
{
  cudaMalloc(A, M * K);
  cudaMalloc(B, K * N);
  cudaMalloc(C, M * N);
}

// Set all the cells in the matrices to a number
void init_mat(float *A, float *B, float *C, const int &M,
    const int &N, const int &K
    )
{
  cudaMemset(A, 1, M * K);
  cudaMemset(B, 1, K * N);
  cudaMemset(C, 1, M * N);
}

// Free the allocated space
void free_space(float *A, float *B, float *C)
{
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
