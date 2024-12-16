#ifndef MATMUL_H
#define MATMUL_H
void sgemm(int M, int N, int K, float alpha,
           const float A*, const float B*,
           float beta, float *C
           );
#endif
