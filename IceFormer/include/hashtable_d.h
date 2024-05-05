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

#ifndef HASHTABLE_D_H
#define HASHTABLE_D_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef struct htentry_d {
    long long key;
    float value;
    struct htentry_d *next;
} htentry_d;

typedef struct hashtable_d {
    int size;
    htentry_d **entries;
    int key_interval;       // The minimum possible interval between adjacent pointers in bytes. Used by the hash function. When keys are pointers to vectors, should set this to sizeof(float)*dimensionality of the vectors. 
} hashtable_d;


void hashtable_d_init(hashtable_d* const ht, const int expected_num_entries, const int key_interval);

bool hashtable_d_exists(const hashtable_d* const ht, const long long key);

float hashtable_d_get(const hashtable_d* const ht, const long long key, const float default_value);

void hashtable_d_set(const hashtable_d* const ht, const long long key, const float value);

bool hashtable_d_delete(const hashtable_d* const ht, const long long key);

void hashtable_d_clear(const hashtable_d* const ht);

void hashtable_d_free(const hashtable_d* const ht);

void hashtable_d_dump(const hashtable_d* const ht);

#ifdef __cplusplus
}
#endif

#endif // HASHTABLE_D_H
