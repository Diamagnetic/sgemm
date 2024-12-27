#include <cuda_runtime.h>
#include "matmul.cuh"

#include <iostream>
#include <chrono>
#include "matmul.h"

constexpr int SIZE = 1024;
constexpr int ROWS_A = 1024;
constexpr int COLS_A = 1024;
constexpr int COLS_B = 1024;

int main()
{
  int M = ROWS_A, N = COLS_B, K = COLS_A;
  float *A, *B, *C, alpha = 1, beta = 0;

  alloc_space(&A, &B, &C, M, N, K);
  init_mat(A, B, C, M, N, K);

  run_mat_mul_wrapper(M, N, K, alpha, A, B, beta, C);

  //    verify_result(&A, &B, &C, M, N, K);

  return 0;
}

void run_mat_mul_wrapper(const int &M, const int &N, const int &K,
    const int &alpha, float *A, float *B,
    const int &beta, float *C
    )
{
  std::chrono::high_resolution_clock::time_point t_start, t_end;

  t_start = std::chrono::high_resolution_clock::now();
  run_mat_mul(M, N, K, alpha, A, B, beta, C);
  t_end = std::chrono::high_resolution_clock::now();

  double time_taken = (std::chrono::duration_cast<std::chrono::nanoseconds>
    (t_end - t_start)).count() / (double) 1000000000;

  print_time_taken(time_taken);
}

void print_time_taken(const double &time_taken)
{
  std::cout.precision(14);

  std::cout << std::fixed << "Time taken = " << time_taken << "seconds\n";
}
