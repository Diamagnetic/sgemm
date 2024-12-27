#ifndef MATMUL_CUDA_H
#define MATMUL_CUDA_H
__global__ void sgemm_naive(int M, int N, int K, float alpha,
    const float *A, const float *B, float beta, float *C
    );

void run_mat_mul(const int &M, const int &N, const int &K,
    const int &alpha, float *A, float *B,
    const int &beta, float *C
    );

void alloc_space(float **A, float **B, float **C, const int &M,
    const int &N, const int &K
    );

void init_mat(float *A, float *B, float *C, const int &M,
    const int &N, const int &K
    );

void free_space(float *A, float *B, float *C);
#endif
