/*
 * Author: Chirag Dhamange
 * File Name: matmul.h
 * Purpose: Function definitions for pure host code.
*/

#ifndef MATMUL_H
#define MATMUL_H
void print_time_taken(const double &time_taken);

void run_mat_mul_wrapper(const int &M, const int &N, const int &K,
    const int &alpha, float *A, float *B,
    const int &beta, float *C
    );
#endif
