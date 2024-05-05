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

#ifndef DCI_H
#define DCI_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "btree_p.h"
#include "hashtable_p.h"

typedef struct dci{
    int dim;                        // (Ambient) dimensionality of data
    int num_comp_indices;           // Number of composite cell_indices
    int num_simp_indices;           // Number of simple cell_indices in each composite index
    int num_points;
    int num_levels;
    long long next_point_id;
    additional_info* root; // this contains the root of our tree, i.e., the additional_info containing the cell_indices of the highest level
    float* proj_vec;               // Assuming column-major layout, matrix of size dim x (num_comp_indices*num_simp_indices)
    float* add_proj_vec;       // projection for transformation
    int* num_points_on_level;  // Number of points of each level
    int** points_on_level;  // point ids of each level
    float lambda;   // fraction
    additional_info* info_addr;
    hashtable_p* inserted_points;
    int max_volume;
    int max_children;
    float* norm_list;  // Norm of each data, correpsonding to the order of id
    float max_norm;
    int parallel_level;
} dci;

// Setting num_to_retrieve and prop_to_retrieve has no effect when blind is true
// Setting field_of_view has no effect when there is only one level
// min_num_finest_level_points is for internal use only; setting it has no effect (since it will be overwritten)
typedef struct dci_query_config {
    bool blind;
    // Querying algorithm terminates whenever we have visited max(num_visited, prop_visited*num_points) points or retrieved max(num_retrieved, prop_retrieved*num_points) points, whichever happens first
    int num_to_visit;
    int num_to_retrieve;
    float prop_to_visit;
    float prop_to_retrieve;
    int field_of_view;
    int min_num_finest_level_points;
    int target_level;
} dci_query_config;

void dci_init(dci* const dci_inst, const int dim, const int num_comp_indices, const int num_simp_indices, float lambda, int max_volume);

void data_projection(int num_comp_indices, int num_simp_indices,
    float* proj_vec, float* add_proj_vec, additional_info** root, const int dim,
    const int num_points, const float* const data, bool* mask,
    float* norm_list, float* max_norm, float** data_proj_ret);

void initialize_tree(int num_comp_indices, int num_simp_indices, additional_info** root);

// Note: the data itself is not kept in the index and must be kept in-place
long long dci_add(dci* const dci_inst, const int dim, const int num_points, const float* const data, const int num_levels, bool* mask, dci_query_config construction_query_config, long long data_id, float* data_proj, int target_level);

int dci_delete(dci*const dci_inst, const int num_points, const long long *const data_ids, dci_query_config deletion_config);

// CAUTION: This function allocates memory for each nearest_neighbours[j], nearest_neighbour_dists[j], so we need to deallocate them outside of this function!
void dci_query(dci* const dci_inst, const int dim, const int num_queries, const float* const query, const int num_neighbours, dci_query_config query_config, bool* mask, int** const nearest_neighbours, float** const nearest_neighbour_dists, int* const num_returned, float scale, int num_populated_levels);

void dci_clear(dci* const dci_inst);

void free_instance(additional_info* info_addr, int* num_points_on_level, int** points_on_level, int num_points, additional_info* root, int num_indices, int num_levels);

// Clear cell_indices and reset the projection directions
void dci_reset(dci* const dci_inst);

void dci_free(dci* const dci_inst);

#ifdef __cplusplus
}
#endif

#endif // DCI_H
