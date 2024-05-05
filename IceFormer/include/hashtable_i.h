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

#ifndef HASHTABLE_I_H
#define HASHTABLE_I_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

typedef struct htentry_i {
    long long key;
    int value;
    struct htentry_i *next;
} htentry_i;

typedef struct hashtable_i {
    int size;
    htentry_i **entries;
    int key_interval;       // The minimum possible interval between adjacent pointers in bytes. Used by the hash function. When keys are pointers to vectors, should set this to sizeof(float)*dimensionality of the vectors. 
} hashtable_i;


void hashtable_i_init(hashtable_i* const ht, const int expected_num_entries, const int key_interval);

bool hashtable_i_exists(const hashtable_i* const ht, const long long key);

int hashtable_i_get(const hashtable_i* const ht, const long long key, const int default_value);

void hashtable_i_set(const hashtable_i* const ht, const long long key, const int value);

bool hashtable_i_delete(const hashtable_i* const ht, const long long key);

void hashtable_i_clear(const hashtable_i* const ht);

void hashtable_i_free(const hashtable_i* const ht);

void hashtable_i_dump(const hashtable_i* const ht);

#ifdef __cplusplus
}
#endif

#endif // HASHTABLE_I_H
