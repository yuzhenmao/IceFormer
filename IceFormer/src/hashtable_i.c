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
#include "hashtable_i.h"

#define LOAD_FACTOR 1.0

void hashtable_i_init(hashtable_i* const ht, const int expected_num_entries, const int key_interval) {
    int num_buckets = (int)ceil(expected_num_entries * LOAD_FACTOR);

    ht->size = num_buckets;
    ht->key_interval = key_interval;
    ht->entries = (htentry_i**) calloc(num_buckets, sizeof(htentry_i*));
}

static inline int hashtable_i_hash(const hashtable_i* const ht, const long long key) {
    return ((key / ht->key_interval) + 84751LL) % ht->size;
}

bool hashtable_i_exists(const hashtable_i* const ht, const long long key) {
    int hashval = hashtable_i_hash(ht, key);
    htentry_i *x = ht->entries[hashval];
    
    while (x) {
        if (x->key == key)
            return true;
        x = x->next;
    }
    return false;
}

int hashtable_i_get(const hashtable_i* const ht, const long long key, const int default_value) {
    int hashval = hashtable_i_hash(ht, key);
    htentry_i *x = ht->entries[hashval];
    
    while (x) {
        if (x->key == key)
            return x->value;
        x = x->next;
    }
    return default_value;
}

void hashtable_i_set(const hashtable_i* const ht, const long long key, const int value) {
    int hashval = hashtable_i_hash(ht, key);
    htentry_i *x = ht->entries[hashval];
    
    while (x) {
        if (x->key == key)
            break;
        x = x->next;
    }
    if (x) {
        x->value = value;
    } else {
        x = (htentry_i*)malloc(sizeof(htentry_i));
        x->key = key;
        x->value = value;
        x->next = ht->entries[hashval];
        
        ht->entries[hashval] = x;
    }
}

bool hashtable_i_delete(const hashtable_i* const ht, const long long key) {
    int hashval = hashtable_i_hash(ht, key);
    htentry_i **x_loc = &(ht->entries[hashval]);
    htentry_i *to_delete;
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

void hashtable_i_clear(const hashtable_i* const ht) {
    int i;
    htentry_i *x, *next;
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

void hashtable_i_free(const hashtable_i* const ht) {
    hashtable_i_clear(ht);
    free(ht->entries);
}

void hashtable_i_dump(const hashtable_i* const ht) {
    int i;
    htentry_i *x;
    for (i = 0; i < ht->size; i++) {
        if (ht->entries[i]) {
            printf("%d: ", i);
            x = ht->entries[i];
            while (x) {
                printf("%lld[%d]->", x->key, x->value);
                x = x->next;
            }
            printf("NIL\n");
        }
    }
}
