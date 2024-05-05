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

#ifndef hashtable_p_H
#define hashtable_p_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "btree_p.h"

typedef struct addinfo_level {
    int level;
    additional_info* addinfo;
} addinfo_level;

typedef struct htentry_p {
    long long key;
    addinfo_level* value;
    struct htentry_p *next;
} htentry_p;

typedef struct hashtable_p {
    int size;
    htentry_p **entries;
    int key_interval;       // The minimum possible interval between adjacent pointers in bytes. Used by the hash function. When keys are pointers to vectors, should set this to sizeof(float)*dimensionality of the vectors. 
} hashtable_p;


void hashtable_p_init(hashtable_p* const ht, const int expected_num_entries, const int key_interval);

bool hashtable_p_exists(const hashtable_p* const ht, const long long key);

addinfo_level* hashtable_p_get(const hashtable_p* const ht, const long long key, const addinfo_level* default_value);

void hashtable_p_set(const hashtable_p* const ht, const long long key, const additional_info* addinfo, const int level);

bool hashtable_p_delete(const hashtable_p* const ht, const long long key);

void hashtable_p_clear(const hashtable_p* const ht);

void hashtable_p_free(const hashtable_p* const ht);

void hashtable_p_dump(const hashtable_p* const ht);

#ifdef __cplusplus
}
#endif

#endif // hashtable_p_H
