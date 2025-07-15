/*
 * Code for IceFormer: Accelerated Inference with Long-Sequence Transformers on CPUs
 * 
 * This code implements the method described in the IceFormer paper, 
 * which can be found at https://openreview.net/forum?id=6RR3wU4mSZ
 * 
 * This file is a part of the Dynamic Continuous Indexing reference 
 * implementation.
 * 
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. 
 * 
 * Copyright (C) 2024    Yuzhen Mao, Ke Li
 */

#ifndef UTIL_H
#define UTIL_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#ifdef USE_MKL
#define SGEMM sgemm
#else
#define SGEMM sgemm_
#endif  // USE_MKL

// BLAS native Fortran interface
extern void SGEMM(const char* const transa, const char* const transb, const int* const m, const int* const n, const int* const k, const float* const alpha, const float* const A, const int* const lda, const float* const B, const int* const ldb, const float* const beta, float* const C, const int* const ldc);

void matmul(const int M, const int N, const int K, const float* const A, const float* const B, float* const C);

void gen_data(float* const data, const int ambient_dim, const int intrinsic_dim, const int num_points);

float compute_dist(const float* const vec1, const float* const vec2, const int dim, const float max_norm, const float norm1, const float norm2);

float compute_dist_query(const float* const vec1, const float* const vec2, const int dim);

float _compute_dist_query(const float* const vec1, const float* const vec2, const int dim);

float rand_normal();

void print_matrix(const float* const data, const int num_rows, const int num_cols);

void key_transform(const float const norms,  float* const projs, const int num_idx, const float* const add_proj_vec, const float max_norm);

void query_transform(const float* const data, const int num_points, const int dim, float* const projs, const int num_idx);

#ifdef __cplusplus
}
#endif

#endif // UTIL_H
