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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#include <malloc.h>
#include <math.h>
#include "util.h"
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#ifndef NO_SIMD
#include<immintrin.h>
#include <x86intrin.h>
#endif

#define INT_SIZE     (8 * sizeof(unsigned int))
static const int SLOT_NUM = 256/INT_SIZE;
#define BITSLOT(b) ((b) / INT_SIZE)
#define BITMASK(b) (1 << ((b) % INT_SIZE))
#define BITSET(a, b) ((a)[BITSLOT(b)] |= BITMASK(b))
#define BITNSLOTS(nb) ((nb + INT_SIZE - 1) / INT_SIZE)
#define BITTEST(a, b) ((a)[BITSLOT(b)] & BITMASK(b))

// Assuming column-major layout, computes A^T * B. A is K x M, B is K x N, and C is M x N. 
void matmul(const int M, const int N, const int K, const float* const A, const float* const B, float* const C) {
    const char TRANSA = 'T';
    const char TRANSB = 'N';
    const float ALPHA = 1.; 
    const float BETA = 0.; 
    const int LDA = K;
    const int LDB = K;
    const int LDC = M;
    SGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

static inline float vecmul(const float* const x, const float* const y, const int k) {
    float inner_prod = 0.0;
#ifndef NO_SIMD
    __m256 X, Y; // 256-bit values
	__m256 acc = _mm256_setzero_ps(); // set to (0, 0, 0, 0)
	float temp[8];
	long i;
	for (i = 0; i < k - 8; i += 8)
	{
		X = _mm256_loadu_ps(x + i); // load chunk of 8 floats
		Y = _mm256_loadu_ps(y + i);
		acc = _mm256_add_ps(acc, _mm256_mul_ps(X, Y));
	}
	_mm256_storeu_ps(&temp[0], acc); // store acc into an array of floats
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3]  + temp[4]  + temp[5]  + temp[6]  + temp[7];
	// add the remaining values
	for (; i < k; i++)
		inner_prod += x[i] * y[i];
#else
    for (int i = 0; i < k; i++)
        inner_prod += x[i] * y[i];
#endif
	return inner_prod;
}

void gen_data(float* const data, const int ambient_dim, const int intrinsic_dim, const int num_points) {
    int i;
    float* latent_data;
    float* transformation;
    assert(posix_memalign((void **)&latent_data, 32, sizeof(float)*intrinsic_dim*num_points) == 0);
    assert(posix_memalign((void **)&transformation, 32, sizeof(float)*intrinsic_dim*ambient_dim) == 0);
    
    //float* latent_data = (float *)memalign(32, sizeof(float)*intrinsic_dim*num_points);
    //float* transformation = (float *)memalign(32, sizeof(float)*intrinsic_dim*ambient_dim);
    for (i = 0; i < intrinsic_dim*num_points; i++) {
        latent_data[i] = 2 * drand48() - 1;
    }
    for (i = 0; i < intrinsic_dim*ambient_dim; i++) {
        transformation[i] = 2 * drand48() - 1;
    }
    // Assuming column-major layout, transformation is intrisic_dim x ambient_dim, 
    // latent_data is intrinsic_dim x num_points, data is ambient_dim x num_points
    matmul(ambient_dim, num_points, intrinsic_dim, transformation, latent_data, data);
    free(latent_data);
    free(transformation);
}

float compute_dist(const float* const vec1, const float* const vec2, const int dim, const float max_norm, const float norm1, const float norm2) {
    int i;
    float sq_dist = 0.0;
    float dots = 0.0;
    // for (i = 0; i < dim; i++) {
    //     dots += (vec1[i])*(vec2[i]);
    // }
    dots = vecmul(vec1, vec2, dim);
    sq_dist = 1.0 - dots/max_norm - sqrt((1.0-norm1/max_norm)*(1.0-norm2/max_norm));
    return sq_dist;
}

float compute_dist_query(const float* const vec1, const float* const vec2, const int dim) {
    float sudo_dist = 0.0;
    sudo_dist = vecmul(vec1, vec2, dim);

    return -1*sudo_dist;
}

float _compute_dist_query(const float* const vec1, const float* const vec2, const int dim) {
    float sudo_dist = 0.0;
    sudo_dist = vecmul(vec1, vec2, dim);

    return sudo_dist;
}

float rand_normal() {
    static float V1, V2, S;
    static int phase = 0;
    float X;

    if(phase == 0) {
        do {
            float U1 = drand48();
            float U2 = drand48();
            
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
            } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

// Print matrix assuming column-major layout
void print_matrix(const float* const data, const int num_rows, const int num_cols) {
    int i, j;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_cols; j++) {
            printf("%.4f\t", data[i+j*num_rows]);
        }
        printf("\n");
    }
}

// Apply key transformation
void key_transform(const float const norms,  float* const projs, const int num_idx, const float* const add_proj_vec, const float max_norm) {
    int j;
    float sqt = sqrt(max_norm);
    for (j = 0; j < num_idx; j++) {
        projs[j] = projs[j]/sqt + sqrt(1-norms/max_norm)*add_proj_vec[j];
    }
}

// Apply query transformation
void query_transform(const float* const data, const int num_points, const int dim, float* const projs, const int num_idx) {
    int i, j;
    float norm_list[num_points];
    float norm = 0.0;
    for (i = 0; i < num_points; i++) {
        for (j = 0; j < dim; j++) {
            norm += data[j+i*dim] * data[j+i*dim];
        }
        norm_list[i] = norm;
        norm = 0.0;
    }
    for (i = 0; i < num_points; i++) {
        for (j = 0; j < num_idx; j++) {
            projs[j+i*num_idx] = projs[j+i*num_idx]/sqrt(norm_list[i]);
        }
    }
}
