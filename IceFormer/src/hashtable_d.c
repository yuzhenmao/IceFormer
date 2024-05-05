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
#include <stdint.h>
#include <math.h>
#include "hashtable_d.h"

#define LOAD_FACTOR 1.0

void hashtable_d_init(hashtable_d* const ht, const int expected_num_entries, const int key_interval) {
    int num_buckets = (int)ceil(expected_num_entries * LOAD_FACTOR);

    ht->size = num_buckets;
    ht->key_interval = key_interval;
    ht->entries = (htentry_d**) calloc(num_buckets, sizeof(htentry_d*));
}

static inline int hashtable_d_hash(const hashtable_d* const ht, const long long key) {
    return ((key / ht->key_interval) + 84751) % ht->size;
}

bool hashtable_d_exists(const hashtable_d* const ht, const long long key) {
    int hashval = hashtable_d_hash(ht, key);
    htentry_d *x = ht->entries[hashval];
    
    while (x) {
        if (x->key == key)
            return true;
        x = x->next;
    }
    return false;
}

float hashtable_d_get(const hashtable_d* const ht, const long long key, const float default_value) {
    int hashval = hashtable_d_hash(ht, key);
    htentry_d *x = ht->entries[hashval];
    
    while (x) {
        if (x->key == key)
            return x->value;
        x = x->next;
    }
    return default_value;
}

void hashtable_d_set(const hashtable_d* const ht, const long long key, const float value) {
    int hashval = hashtable_d_hash(ht, key);
    htentry_d *x = ht->entries[hashval];
    
    while (x) {
        if (x->key == key)
            break;
        x = x->next;
    }
    if (x) {
        x->value = value;
    } else {
        x = (htentry_d*)malloc(sizeof(htentry_d));
        x->key = key;
        x->value = value;
        x->next = ht->entries[hashval];
        
        ht->entries[hashval] = x;
    }
}

bool hashtable_d_delete(const hashtable_d* const ht, const long long key) {
    int hashval = hashtable_d_hash(ht, key);
    htentry_d **x_loc = &(ht->entries[hashval]);
    htentry_d *to_delete;
    while (*x_loc) {
        if ((*x_loc)->key == key)
            break;
        x_loc = &((*x_loc)->next);
    }
    if (*x_loc) {
        to_delete = *x_loc;
        *x_loc = (*x_loc)->next;
        free(to_delete);
        return true;
    } else {
        return false;
    }
}

void hashtable_d_clear(const hashtable_d* const ht) {
    int i;
    htentry_d *x, *next;
    for (i = 0; i < ht->size; i++) {
        x = ht->entries[i];
        while (x) {
            next = x->next;
            free(x);
            x = next;
        }
        ht->entries[i] = NULL;
    }
}

void hashtable_d_free(const hashtable_d* const ht) {
    hashtable_d_clear(ht);
    free(ht->entries);
}

void hashtable_d_dump(const hashtable_d* const ht) {
    int i;
    htentry_d *x;
    for (i = 0; i < ht->size; i++) {
        if (ht->entries[i]) {
            printf("%d: ", i);
            x = ht->entries[i];
            while (x) {
                printf("%lld[%.4f]->", x->key, x->value);
                x = x->next;
            }
            printf("NIL\n");
        }
    }
}
