#include "matmul.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha,
    const float *A, const float *B,
    float beta, float *C
    )
{
  const uint x = blockDim.x * blockIdx.x + threadIdx.x;
  const uint y = blockDim.y * blockIdx.y + threadIdx.y;

  if(x < M && y < N)
  {
    float tmp = 0.0;

    for(int i = 0; i < K; i++)
      tmp += A[x * K + i] * B[i * N + y];

    C[x * N + y] *= beta;
    C[x * N + y] += alpha * tmp;
  }
}

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

void alloc_space(float **A, float **B, float **C, const int &M,
    const int &N, const int &K
    )
{
  cudaMalloc(A, M * K);
  cudaMalloc(B, K * N);
  cudaMalloc(C, M * N);
}

void init_mat(float *A, float *B, float *C, const int &M,
    const int &N, const int &K
    )
{
  cudaMemset(A, 1, M * K);
  cudaMemset(B, 1, K * N);
  cudaMemset(C, 1, M * N);
}

void free_space(float *A, float *B, float *C)
{
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
