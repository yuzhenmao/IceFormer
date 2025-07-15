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

#ifndef BTREE_COMMON_H
#define BTREE_COMMON_H

#define BTREE_LEAF_MAX_NUM_SLOTS 32
#define BTREE_INNER_MAX_NUM_SLOTS 16

/// If node size is larger than this threshold, performs binary search in 
/// find_lower_inner() and find_lower_leaf(); otherwise, performs linear 
/// search. 
#define BTREE_BIN_SEARCH_THRESH 256

static const bool allow_duplicates = true;
/// Result flags of recursive deletion.
typedef enum btree_c_result_flags {
    /// Deletion successful and no fix-ups necessary.
    btree_ok = 0,

    /// Deletion not successful because key was not found.
    btree_not_found = 1,

    /// Deletion successful, the last key was updated so id slotkeys
    /// need updates.
    btree_update_last_key = 2,

    /// Deletion successful, children nodes were merged and the id
    /// needs to remove the empty node.
    btree_fix_merge = 4
} btree_c_result_flags;

/// B+ tree recursive deletion has much information which is needs to be
/// passed upward.
typedef struct {
    /// Merged result flags
    btree_c_result_flags flags;

    /// The key to be updated at the id's slot
    float last_key;
} btree_c_result;

#endif
