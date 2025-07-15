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

#ifndef BTREE_P_H
#define BTREE_P_H
#include <omp.h> 

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include "btree_common.h"

typedef struct data_pt data_pt;
typedef struct bulk_data_pt bulk_data_pt;
typedef struct additional_info additional_info;
typedef struct idx_arr idx_arr;

typedef struct btree_p_inner_node {
    // Level in the B+ tree, if level == 0 -> leaf node
    unsigned short level;

    // Number of keys used, so number of valid children or data
    // pointers
    unsigned short num_slots_used;

    // Keys of children or data pointers
    float* slot_keys;

    // Pointers to children
    void** slot_ptrs;

} btree_p_inner_node;

typedef struct btree_p_leaf_node {
    // Level in the B+ tree, if level == 0 -> leaf node
    unsigned short level;

    // Number of keys used, so number of valid children or data
    // pointers
    unsigned short num_slots_used;

    // float linked list pointers to traverse the leaves
    struct btree_p_leaf_node *prev_leaf;

    // float linked list pointers to traverse the leaves
    struct btree_p_leaf_node *next_leaf;

    // Keys of children or data pointers
    float* slot_keys;

    // Array of data
    data_pt* slot_data;
} btree_p_leaf_node;

typedef struct btree_p {
    // number of data currently in the tree
    unsigned int num_data;

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
    btree_p_leaf_node *first_leaf;

    // Pointer to last leaf in the float linked leaf chain
    btree_p_leaf_node *last_leaf;
} btree_p;

struct additional_info {
    long long id;
    int local_id;
    float* local_dist;
    int* num_finest_level_points;
    const float* data_loc;
    float parent_dist;  // distance to the parent
    additional_info* parent_info;  // parent's additional_info, useful for deletion
    btree_p* cell_indices;  // indices of the cell (points in the lower level which this point is the parent of)
    idx_arr* arr_indices;
    bool flag;  // if arr_indices needs to update or not
};


struct data_pt {
    additional_info* info;
};

struct idx_arr {
    float key;
    int local_id;
    additional_info* info;
};

struct bulk_data_pt {
    data_pt data_pt;
    float local_parent_dist; // local_distance to the parent  // Add these two line will slow down the query time
    long long parent_id;                                                           //  from 0.037 to 0.041 (slow down construction more: 0.55 to 0.61)
};

typedef struct btree_p_search_res {
	btree_p_leaf_node* n;
	int slot;
} btree_p_search_res;

void btree_p_init(btree_p* tree);
void btree_p_clear(btree_p* tree);
bool btree_p_insert(btree_p* const tree, const float key, const data_pt value);
void btree_p_bulk_load(btree_p* const tree, float* keybegin, float* keyend, data_pt* databegin, data_pt* dataend);
// Node is considered a match if its key is within the range between key-(1e-4) and 
// key+(1e-4) inclusive and its ID is exactly equal to value. 
bool btree_p_delete(btree_p* const tree, const float key, const long long value);
btree_p_search_res btree_p_search(btree_p* const tree, const float key);
bool btree_p_is_end(const btree_p* const tree, const btree_p_search_res src);
btree_p_search_res btree_p_find_prev(btree_p_search_res src);
btree_p_search_res btree_p_find_next(btree_p_search_res src);
btree_p_search_res btree_p_first(const btree_p* const tree);
btree_p_search_res btree_p_last(const btree_p* const tree);
void btree_p_dump(btree_p* const tree);
float btree_p_keyof(const btree_p_search_res src);
data_pt btree_p_valueof(const btree_p_search_res src);

#ifdef __cplusplus
}
#endif

#endif
