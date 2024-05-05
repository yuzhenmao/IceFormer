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

#ifndef BTREE_I_H
#define BTREE_I_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include "btree_common.h"

typedef struct btree_i_inner_node {
    // Level in the B+ tree, if level == 0 -> leaf node
    unsigned short level;

    // Number of keys used, so number of valid children or data
    // pointers
    unsigned short num_slots_used;

    // Keys of children or data pointers
    float* slot_keys;

    // Pointers to children
    void** slot_ptrs;

} btree_i_inner_node;

typedef struct btree_i_leaf_node {
    // Level in the B+ tree, if level == 0 -> leaf node
    unsigned short level;

    // Number of keys used, so number of valid children or data
    // pointers
    unsigned short num_slots_used;

    // float linked list pointers to traverse the leaves
    struct btree_i_leaf_node *prev_leaf;

    // float linked list pointers to traverse the leaves
    struct btree_i_leaf_node *next_leaf;

    // Keys of children or data pointers
    float* slot_keys;

    // Array of data
    int* slot_data;
} btree_i_leaf_node;

typedef struct btree_i {
    // Base B+ tree parameter: The number of key/data slots in each leaf
    unsigned short leaf_max_num_slots;

    // Base B+ tree parameter: The number of key slots in each inner node,
    // this can differ from slots in each leaf.
    unsigned short inner_max_num_slots;

    // Computed B+ tree parameter: The minimum number of key/data slots used
    // in a leaf. If fewer slots are used, the leaf will be merged or slots
    // shifted from it's siblings.
    unsigned short leaf_min_num_slots;

    // Computed B+ tree parameter: The minimum number of key slots used
    // in an inner node. If fewer slots are used, the inner node will be
    // merged or slots shifted from it's siblings.
    unsigned short inner_min_num_slots;
    // Pointer to the B+ tree's root node, either leaf or inner node
    void* root;

    // Pointer to first leaf in the float linked leaf chain
    btree_i_leaf_node *first_leaf;

    // Pointer to last leaf in the float linked leaf chain
    btree_i_leaf_node *last_leaf;
} btree_i;

typedef struct btree_i_search_res {
	btree_i_leaf_node* n;
	int slot;
} btree_i_search_res;

void btree_i_init(btree_i* tree);
void btree_i_clear(btree_i* tree);
bool btree_i_insert(btree_i* const tree, const float key, const int value);
void btree_i_bulk_load(btree_i* const tree, float* keybegin, float* keyend, int* databegin, int* dataend);
// Node is considered a match if its key is within the range between key-(1e-8) and 
// key+(1e-8) inclusive and its value is exactly equal to value. 
bool btree_i_delete(btree_i* const tree, const float key, const int value);
btree_i_search_res btree_i_search(btree_i* const tree, const float key);
bool btree_i_is_end(const btree_i* const tree, const btree_i_search_res src);
btree_i_search_res btree_i_find_prev(btree_i_search_res src);
btree_i_search_res btree_i_find_next(btree_i_search_res src);
btree_i_search_res btree_i_first(const btree_i* const tree);
btree_i_search_res btree_i_last(const btree_i* const tree);
void btree_i_dump(btree_i* const tree);
int btree_i_valueof(const btree_i_search_res src);
float btree_i_keyof(const btree_i_search_res src);

#ifdef __cplusplus
}
#endif

#endif
