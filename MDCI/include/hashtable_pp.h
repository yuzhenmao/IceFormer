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

#ifndef hashtable_pp_H
#define hashtable_pp_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "numpy/arrayobject.h"

typedef struct htentry_pp {
    long long key;
    PyArrayObject *value;
    struct htentry_pp *next;
} htentry_pp;

typedef struct hashtable_pp {
    int size;
    htentry_pp **entries;
    int key_interval;       // The minimum possible interval between adjacent pointers in bytes. Used by the hash function. When keys are pointers to vectors, should set this to sizeof(float)*dimensionality of the vectors. 
} hashtable_pp;


void hashtable_pp_init(hashtable_pp* const ht, const int key_interval, const int expected_num_entries);

// void hashtable_pp_add(hashtable_pp* const ht, const int expected_num_entries);

bool hashtable_pp_exists(const hashtable_pp* const ht, const long long key);

PyArrayObject * hashtable_pp_get(const hashtable_pp* const ht, const long long key, const PyArrayObject * default_value);

void hashtable_pp_set(const hashtable_pp* const ht, const long long key, const PyArrayObject * value);

bool hashtable_pp_delete(const hashtable_pp* const ht, const long long key);

void hashtable_pp_clear(const hashtable_pp* const ht);

void hashtable_pp_free(const hashtable_pp* const ht);

void hashtable_pp_dump(const hashtable_pp* const ht);

#ifdef __cplusplus
}
#endif

#endif // hashtable_pp_H
