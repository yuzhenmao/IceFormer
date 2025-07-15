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

 //#include <malloc.h>
#include "dci.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "btree_i.h"
#include "hashtable_d.h"
#include "hashtable_i.h"
#include "hashtable_p.h"
#ifndef NO_SIMD
#include<immintrin.h>
#include <x86intrin.h>
#endif
#include <limits.h>
#include "util.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif  // USE_OPENMP

#define INT_SIZE     (8 * sizeof(unsigned int))
static const int SLOT_NUM = 256/INT_SIZE;    // # of int in one SIMD register

#define BITSLOT(b) ((b) / INT_SIZE)
#define BITMASK(b) (1 << ((b) % INT_SIZE))
#define BITSET(a, b) ((a)[BITSLOT(b)] |= BITMASK(b))
#define BITNSLOTS(nb) ((nb + INT_SIZE - 1) / INT_SIZE)   // # of int for nb bits
#define BITTEST(a, b) ((a)[BITSLOT(b)] & BITMASK(b))

#define CLOSEST 128

#ifndef NO_SIMD
static inline void BitAnd(const unsigned int* const x, unsigned int* const y, const int k) {
    __m256i X, Y; // 256-bit values
	long i=0;
	for (i = 0; i < k; i += SLOT_NUM) {
		X = _mm256_load_si256(x + i); // load chunk of ints
		Y = _mm256_load_si256(y + i);
		_mm256_store_si256(y + i, _mm256_and_si256(X, Y));
	}
}

static inline void BitNot_And(const unsigned int* const x, unsigned int* const y, const int k) {
    __m256i X, Y; // 256-bit values
    __m256i mask = _mm256_set1_epi32(-1);
	long i=0;
	for (i = 0; i < k; i += SLOT_NUM) {
		X = _mm256_load_si256(x + i); // load chunk of ints
		Y = _mm256_load_si256(y + i);
		_mm256_store_si256(y + i, _mm256_and_si256(_mm256_xor_si256(X, mask), Y));
	}
}
#else
static inline void BitAnd(const unsigned int* const x, unsigned int* const y, const int k) {
    for (int i = 0; i < k; i++) {
        y[i] = x[i] & y[i];
    }
}

static inline void BitNot_And(const unsigned int* const x, unsigned int* const y, const int k) {
    for (int i = 0; i < k; i++) {
        y[i] = (~x[i]) & y[i];
    }
}
#endif

static inline float abs_d(float x) { return x > 0 ? x : -x; }

static inline int min_i(int a, int b) { return a < b ? a : b; }

static inline int max_i(int a, int b) { return a > b ? a : b; }

typedef struct tree_node {
    additional_info* parent;
    long long child;
    float dist;
} tree_node;

void free_cell(struct additional_info* cell, int num_indices) {
    if(cell == NULL) return;
    // free indices only if needed
    int i;
    if (cell->cell_indices) {
        for (i = 0; i < num_indices; i++) {
            btree_p_clear(&(cell->cell_indices[i]));
        }
        free(cell->cell_indices);
    }
    if (cell->arr_indices)
        free(cell->arr_indices);
    if (cell->num_finest_level_points)
        free(cell->num_finest_level_points);
    if (cell->local_dist)
        free(cell->local_dist);
}

void free_instance(additional_info* info_addr, int* num_points_on_level, int** points_on_level, int num_points, additional_info* root, int num_indices, int num_levels) {
    if (root == NULL)
        return;
    free_cell(root, num_indices);
    for (int i = 0; i < num_points; i++) {
        free_cell(info_addr+i, num_indices);
    }
    free(root);
    if (points_on_level != NULL) {
        for (int i = 0; i < num_levels; i++) {
            free(points_on_level[i]);
        }
        free(points_on_level);
    }
    if (num_points_on_level != NULL) {
        free(num_points_on_level);
    }
    free(info_addr);
}

static inline int add_to_list(int num_candidates, int num_neighbours, idx_arr* top_candidates, int* num_returned, 
                        float cur_dist, additional_info* cur_points, float* last_top_candidate_dist, dci_query_config query_config, 
                        int* num_returned_finest_level_points, int* last_top_candidate, int i, float init, int num_finest) {
    if (num_candidates < num_neighbours) {
        top_candidates[*num_returned].key = cur_dist;
        top_candidates[*num_returned].info = cur_points;
        if (cur_dist > (*last_top_candidate_dist)) {
            (*last_top_candidate_dist) = cur_dist;
            (*last_top_candidate) = *num_returned;
        }
        (*num_returned)++;
        if (query_config.min_num_finest_level_points) {
            (*num_returned_finest_level_points) += num_finest;
        }
    }
    else if (cur_dist < (*last_top_candidate_dist)) {
        if (query_config.min_num_finest_level_points &&
            (*num_returned_finest_level_points) + num_finest -
            top_candidates[(*last_top_candidate)].info->num_finest_level_points[query_config.target_level] <
            query_config.min_num_finest_level_points) { // Add
            top_candidates[*num_returned].key = cur_dist;
            top_candidates[*num_returned].info = cur_points;
            (*num_returned)++;
            (*num_returned_finest_level_points) += num_finest;
        }
        else {
            // Replace
            if (query_config.min_num_finest_level_points) {
                (*num_returned_finest_level_points) += num_finest -
                    top_candidates[(*last_top_candidate)].info->num_finest_level_points[query_config.target_level];
            }
            top_candidates[(*last_top_candidate)].key = cur_dist;
            top_candidates[(*last_top_candidate)].info = cur_points;
            (*last_top_candidate_dist) = init;
            for (int j = 0; j < *num_returned; j++) {
                if (top_candidates[j].key > (*last_top_candidate_dist)) {
                    (*last_top_candidate_dist) = top_candidates[j].key;
                    (*last_top_candidate) = j;
                }
            }
        }
    }
    else if (query_config.min_num_finest_level_points && 
        (*num_returned_finest_level_points) <  query_config.min_num_finest_level_points) { // Also Add
        top_candidates[*num_returned].key = cur_dist;
        top_candidates[*num_returned].info = cur_points;
        (*num_returned)++;
        (*num_returned_finest_level_points) += num_finest;
    }
}

static void dci_gen_proj_vec(float* const proj_vec, const int dim,
    const int num_indices) {
    int i, j;
    float sq_norm, norm;
    for (i = 0; i < dim * num_indices; i++) {
        proj_vec[i] = rand_normal();
    }
    for (j = 0; j < num_indices; j++) {
        sq_norm = 0.0;
        for (i = 0; i < dim; i++) {
            sq_norm += (proj_vec[i + j * dim] * proj_vec[i + j * dim]);
        }
        norm = sqrt(sq_norm);
        for (i = 0; i < dim; i++) {
            proj_vec[i + j * dim] /= norm;
        }
    }
}

void data_projection(int num_comp_indices, int num_simp_indices,
    float* proj_vec, float* add_proj_vec, additional_info** root, const int dim,
    const int num_points, const float* const data, bool* mask,
    float* norm_list, float* max_norm, float** data_proj_ret) {

    int i, j, ii;
    int num_indices = num_comp_indices * num_simp_indices;
    // True if data_proj is (# of points) x (# of cell_indices)
    // column-major; used only for error-checking

    float* data_proj;
    if (posix_memalign((void**)&data_proj, 32, sizeof(float) * num_indices * num_points) != 0) {
        perror("Memory allocation failed!\n");
        return;
    }
    *data_proj_ret = data_proj;

    // Calculate the norm of all points
    float temp_norm;
    for (i = 0; i < num_points; i++) {
        if (mask == NULL || mask[i]) {
            temp_norm = 0.0;
            for (j = 0; j < dim; j++) {
                temp_norm += data[j+i*dim] * data[j+i*dim];
            }
            norm_list[i] = temp_norm;
            if (*max_norm <  temp_norm) {
                *max_norm = temp_norm;
            }
        }
    }

    // data_proj is (# of cell_indices) x (# of points) column-major
    matmul(num_indices, num_points, dim, proj_vec, data, data_proj);
    // key_transform
    for (i = 0; i < num_points; i++) {
        if (mask == NULL || mask[i]) {
            for (j = 0; j < num_indices; j++) {
                data_proj[j+i*num_indices] = data_proj[j+i*num_indices] + sqrt((*max_norm) - (norm_list[i])) * add_proj_vec[j];
            }
        }
    }
}

static inline int dci_next_closest_proj(const idx_arr* const index, int* const left_pos, int* const right_pos, const float query_proj, const int num_elems, int* returned_ids, float* index_priority) {
    
    int returned_num = 0;
    int temp_right = *right_pos;
    int temp_left = *left_pos;
    int i = 0;
    int num = 0;
    if (temp_left == -1 && temp_right == num_elems) {
        return 0;
    } else if (temp_left == -1) {
        if (temp_right <= num_elems - CLOSEST) {
            returned_num = CLOSEST;
            for (i = 0; i < CLOSEST; i++) {
                returned_ids[num++] = index[temp_right].local_id;
                ++temp_right;
            }
        }
        else {
            returned_num = num_elems - temp_right;
            for (i = 0; temp_right < num_elems; i++) {
                returned_ids[num++] = index[temp_right].local_id;
                ++temp_right;
            }
        }
        *index_priority = abs_d(index[temp_right-1].key - query_proj);
    } else if (temp_right == num_elems) {
        if (temp_left >= CLOSEST-1) {
            returned_num = CLOSEST;
            for (i = 0; i < CLOSEST; i++) {
                returned_ids[num++] = index[temp_left].local_id;
                --temp_left;
            }
        }
        else {
            returned_num = temp_left + 1;
            for (i = 0; temp_left > -1; i++) {
                returned_ids[num++] = index[temp_left].local_id;
                --temp_left;
            }
        }
        *index_priority = abs_d(index[temp_left+1].key - query_proj);
    } else if (index[temp_right].key - query_proj < query_proj - index[temp_left].key) {
        if (temp_right <= num_elems - CLOSEST) {
            returned_num = CLOSEST;
            for (i = 0; i < CLOSEST; i++) {
                returned_ids[num++] = index[temp_right].local_id;
                ++temp_right;
            }
            *index_priority = abs_d(index[temp_right-1].key - query_proj);
        }
        else {
            returned_num = num_elems - temp_right;
            for (i = 0; temp_right < num_elems; i++) {
                returned_ids[num++] = index[temp_right].local_id;
                ++temp_right;
            }
            if (temp_left >= CLOSEST-1-i) {
                returned_num = CLOSEST;
                for (; i < CLOSEST; i++) {
                    returned_ids[num++] = index[temp_left].local_id;
                    --temp_left;
                }
            }
            else {
                returned_num += temp_left + 1;
                for (; temp_left > -1; i++) {
                    returned_ids[num++] = index[temp_left].local_id;
                    --temp_left;
                }
            }
            *index_priority = abs_d(index[temp_left+1].key - query_proj);
        }
    } else {
        if (temp_left >= CLOSEST-1) {
            returned_num = CLOSEST;
            for (i = 0; i < CLOSEST; i++) {
                returned_ids[num++] = index[temp_left].local_id;
                --temp_left;
            }
            *index_priority = abs_d(index[(temp_left)+1].key - query_proj);
        }
        else {
            returned_num = (temp_left) + 1;
            for (i = 0; temp_left > -1; i++) {
                returned_ids[num++] = index[temp_left].local_id;
                --temp_left;
            }
            if (temp_right <= num_elems - CLOSEST + i) {
                returned_num = CLOSEST;
                for (; i < CLOSEST; i++) {
                    returned_ids[num++] = index[temp_right].local_id;
                    ++temp_right;
                }
            }
            else {
                returned_num += num_elems - (temp_right);
                for (; temp_right < num_elems; i++) {
                    returned_ids[num++] = index[temp_right].local_id;
                    ++temp_right;
                }
            }
            *index_priority = abs_d(index[temp_right-1].key - query_proj);
        }
    }
    (*left_pos) = temp_left;
    (*right_pos) = temp_right;
    return returned_num;
}

static inline int dci_next_closest_proj_(const idx_arr* const index, int* const left_pos, int* const right_pos, const float query_proj, const int num_elems) {

    int cur_pos;
    if (*left_pos == -1 && *right_pos == num_elems) {
        cur_pos = -1;
    } else if (*left_pos == -1) {
        cur_pos = *right_pos;
        ++(*right_pos);
    } else if (*right_pos == num_elems) {
        cur_pos = *left_pos;
        --(*left_pos);
    } else if (index[*right_pos].key - query_proj < query_proj - index[*left_pos].key) {
        cur_pos = *right_pos;
        ++(*right_pos);
    } else {
        cur_pos = *left_pos;
        --(*left_pos);
    }
    return cur_pos;
}

// Returns the index of the element whose key is the largest that is less than the key
// Returns an integer from -1 to num_elems - 1 inclusive
// Could return -1 if all elements are greater or equal to key
static inline int dci_search_index(const idx_arr* const index, const float key, const int num_elems) {
    int start_pos, end_pos, cur_pos;
    
    start_pos = -1;
    end_pos = num_elems - 1;
    cur_pos = (start_pos + end_pos + 2) / 2;
    
    while (start_pos < end_pos) {
        if (index[cur_pos].key < key) {
            start_pos = cur_pos;
        } else {
            end_pos = cur_pos - 1;
        }
        cur_pos = (start_pos + end_pos + 2) / 2;
    }
    
    return start_pos;
}

static inline int dci_compare_data_pt_parent(const void *a, const void *b) {
    float key_diff = ((bulk_data_pt *)a)->parent_id - ((bulk_data_pt *)b)->parent_id;
    return (key_diff > 0) - (key_diff < 0);
}

static inline int dci_compare_data_pt_dist(const void *a, const void *b) {
    float key_diff = ((bulk_data_pt *)a)->local_parent_dist - ((bulk_data_pt *)b)->local_parent_dist;
    return (key_diff > 0) - (key_diff < 0);
}

static inline int dci_compare_data_idx_arr_dist(const void *a, const void *b) {
    float key_diff = ((idx_arr *)a)->key - ((idx_arr *)b)->key;
    return (key_diff > 0) - (key_diff < 0);
}

static inline int dci_compare_distance(const void *a, const void *b) {
    float key_diff = *(float*)a - *(float*)b;
    return (key_diff < 0) - (key_diff > 0);
}

static inline int dci_compare_id(const void *a, const void *b) {
    float key_diff = *(int*)a - *(int*)b;
    return (key_diff < 0) - (key_diff > 0);
}

void update_arr_indices(int num_indices, additional_info* point) {
    int num_points = point->cell_indices[0].num_data;
    if (num_points == 0) {
        point->arr_indices = NULL;
    }
    else {
        data_pt cur_point;
        idx_arr* arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*num_points));
        for (int i = 0; i < num_indices; i++) {
            btree_p_search_res cur = btree_p_first(&(point->cell_indices[i]));
            for (int j = 0; j < num_points; j++) {
                cur_point = btree_p_valueof(cur);
                if (i==0) {
                    cur_point.info->local_id = j;
                    arr_indices[i*num_points+j].local_id = j;
                }
                else {
                    arr_indices[i*num_points+j].local_id = cur_point.info->local_id;
                }
                arr_indices[i*num_points+j].info = cur_point.info;
                arr_indices[i*num_points+j].key = cur_point.info->local_dist[i];
                cur = btree_p_find_next(cur);
            }
        }
        free(point->arr_indices);
        point->arr_indices = arr_indices;
    }
    point->flag = 0;
}

static int dci_query_single_point(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels, int parallel_level,
    additional_info* root, int num_populated_levels, int num_neighbours, idx_arr* points_to_expand, 
    idx_arr** points_to_expand_next, int* num_top_candidates,
    const float* const query, const float* const query_proj, float max_norm, float* norm_list, int query_id, 
    dci_query_config query_config, idx_arr* const top_candidates, bool cumu);

static int dci_query_single_point_(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels, int parallel_level,
    additional_info* root, int num_populated_levels, int num_neighbours,
    const float* const query, const float* const query_proj, float max_norm, float* norm_list, int query_id,
    dci_query_config query_config, idx_arr* const top_candidates, bool cumu);

static inline void initialize_indices(btree_p* tree, int num_indices) {
    for (int i = 0; i < num_indices; i++) {
        btree_p_init(&(tree[i]));
    }
}

void dci_init(dci* const dci_inst, const int dim,
    const int num_comp_indices, const int num_simp_indices, float lambda, int max_volume) {
    srand48(time(NULL));
    
    int num_indices = num_comp_indices * num_simp_indices;
    float* temp_proj_vec;
    dci_inst->dim = dim;
    dci_inst->num_comp_indices = num_comp_indices;
    dci_inst->num_simp_indices = num_simp_indices;
    dci_inst->num_points = 0;
    dci_inst->num_levels = 0;
    dci_inst->next_point_id = 0;
    dci_inst->max_children = 0;
    dci_inst->num_points_on_level = NULL;
    dci_inst->points_on_level = NULL;
    dci_inst->root = NULL;
    dci_inst->info_addr = NULL;
    dci_inst->lambda = lambda;
    dci_inst->max_volume = max_volume;
    dci_inst->norm_list = (float*)malloc(sizeof(float)*max_volume);
    dci_inst->max_norm = 0;
    dci_inst->parallel_level = 1;

    if (posix_memalign((void**)&temp_proj_vec, 32, sizeof(float) * (dim+1) * num_indices) != 0) {
        perror("Memory allocation failed!\n");
        return;
    }
    if (posix_memalign((void**)&(dci_inst->proj_vec), 32, sizeof(float) * dim * num_indices) != 0) {
        perror("Memory allocation failed!\n");
        return;
    }
    if (posix_memalign((void**)&(dci_inst->add_proj_vec), 32, sizeof(float) * 1 * num_indices) != 0) {
        perror("Memory allocation failed!\n");
        return;
    }
    dci_gen_proj_vec(temp_proj_vec, dim+1, num_indices);

    for (int j = 0; j < num_indices; j++) {
        for (int i = 0; i < dim; i++) {
            dci_inst->proj_vec[i + j * (dim)] = temp_proj_vec[i + j * (dim+1)];
        }
        dci_inst->add_proj_vec[j] = temp_proj_vec[dim + j * (dim+1)];
    }
    free(temp_proj_vec);
}

void dci_clear(dci* const dci_inst) {
    free_instance(dci_inst->info_addr, dci_inst->num_points_on_level, dci_inst->points_on_level, dci_inst->num_points, dci_inst->root, dci_inst->num_simp_indices * dci_inst->num_comp_indices, dci_inst->num_levels);
    dci_inst->num_points = 0;
    dci_inst->num_levels = 0;
    dci_inst->next_point_id = 0;
    dci_inst->num_points_on_level = NULL;
    dci_inst->points_on_level = NULL;
    dci_inst->root = NULL;
    dci_inst->info_addr = NULL;
}

void dci_free(dci* const dci_inst) {
    dci_clear(dci_inst);
    free(dci_inst->norm_list);
    dci_inst->norm_list = NULL;
    free(dci_inst->proj_vec);
    dci_inst->proj_vec = NULL;
    free(dci_inst->add_proj_vec);
    dci_inst->add_proj_vec = NULL;
}

void dci_reset(dci* const dci_inst) {
    srand48(time(NULL)); 
    dci_clear(dci_inst);
    int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
    int dim = dci_inst->dim;
    float* temp_proj_vec;
    if (posix_memalign((void**)&temp_proj_vec, 32, sizeof(float) * (dim+1) * num_indices) != 0) {
        perror("Memory allocation failed!\n");
        return;
    }
    dci_gen_proj_vec(temp_proj_vec, dim+1, num_indices);
    for (int j = 0; j < num_indices; j++) {
        for (int i = 0; i < dim; i++) {
            dci_inst->proj_vec[i + j * (dim)] = temp_proj_vec[i + j * (dim+1)];
        }
        dci_inst->add_proj_vec[j] = temp_proj_vec[dim + j * (dim+1)];
    }
    free(temp_proj_vec);
} 

static void dci_assign_parent(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels, int parallel_level,
    additional_info* root, const int num_populated_levels, const int num_queries,
    const int* selected_query_pos, const float* const query, float max_norm, float* norm_list,
    const float* const query_proj, const dci_query_config query_config,
    tree_node* const assigned_parent);

void initialize_tree(int num_comp_indices, int num_simp_indices, additional_info** root) {

    int num_indices = num_comp_indices * num_simp_indices;

    *root = (additional_info*)malloc(sizeof(additional_info));
    (*root)->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
    (*root)->arr_indices = NULL;
    (*root)->num_finest_level_points = NULL;
    (*root)->local_dist = NULL;
    (*root)->id = -1;
    (*root)->flag = 1;
    initialize_indices((*root)->cell_indices, num_indices);
}

void construct_new_tree(int num_comp_indices, int num_simp_indices, int parallel_level,
    float* proj_vec, float* add_proj_vec, additional_info** root, const int dim,
    const int num_points, int* actual_num_levels,
    const float* const data, int* num_points_on_level,
    float* data_proj, const int num_levels,
    additional_info* level_cells_ret, bool* mask, int** points_on_level,
    const dci_query_config construction_query_config, float* norm_list, float* max_norm) {

    // If add multiple points, do the following
    int h, i, j;
    int num_points_on_upper_levels, num_points_on_cur_levels;
    // Only populated when actual_num_levels >= 2
    int num_indices = num_comp_indices * num_simp_indices;
    // True if data_proj is (# of points) x (# of cell_indices)
    bool data_proj_transposed = false;
    // column-major; used only for error-checking
    float promotion_prob;

    if (num_levels < 2) {
        *actual_num_levels = num_levels;
        int j = 0;
        for (i=0; i< num_points; i++){
            if (mask[i]) {
                points_on_level[0][j++] = i;
            }
        }
        num_points_on_level[0] = j;
    }
    else {
        int data_levels[num_points];
        promotion_prob = pow((float)num_points, -1.0 / num_levels);
        int level_relabelling[num_levels];

        *actual_num_levels = 0;
        while (*actual_num_levels != num_levels) {
            for (i = 0; i < num_levels; i++) {
                num_points_on_level[i] = 0;
            }
            for (j = 0; j < num_points; j++) {
                if (mask[j]) {
                    for (i = 0; i < num_levels - 1; i++) {
                        if (drand48() > promotion_prob) {
                            break;
                        }
                    }
                    num_points_on_level[i]++;
                    data_levels[j] = i;
                }
            }

            // Remove all levels with no points
            h = 0;
            for (i = 0; i < num_levels; i++) {
                if (num_points_on_level[i] > 0) {
                    level_relabelling[i] = h;
                    h++;
                }
                else {
                    level_relabelling[i] = -1;
                }
            }
            *actual_num_levels = h;
        }

        // for (i = 0; i < num_levels; i++) {
        //     if (level_relabelling[i] >= 0) {
        //         num_points_on_level[level_relabelling[i]] = num_points_on_level[i];
        //     }
        // }

        if (*actual_num_levels >= 2) {
            int level_num[*actual_num_levels];
            for (i = 0; i < *actual_num_levels; i++) {
                level_num[i] = 0;
            }
            for (j = 0; j < num_points; j++) {
                if (mask[j]) {
                    h = level_relabelling[data_levels[j]];
                    points_on_level[h][level_num[h]] = j;
                    level_num[h]++;
                }
            }
        }
        else {
            int j = 0;
            for (i=0; i< num_points; i++){
                if (mask[i]) {
                    points_on_level[0][j] = i;
                }
            }
        }
    }

    i = (*actual_num_levels) - 1;
    num_points_on_cur_levels = num_points_on_level[i];

    tree_node assigned_parent[num_points];
    (*root)->arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*num_points_on_level[i]));

    bulk_data_pt bbulk[num_points_on_cur_levels];

    for (j = 0; j < num_points_on_cur_levels; j++) {
        int k;
        additional_info* cur_cell = level_cells_ret + points_on_level[i][j];
        cur_cell->id = points_on_level[i][j];
        cur_cell->arr_indices = NULL;
        cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
        initialize_indices(cur_cell->cell_indices, num_indices);
        cur_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
        cur_cell->num_finest_level_points = (int*)malloc(sizeof(int) * (i + 1));
        for (int l = i; l >= 0; l--) {
            cur_cell->num_finest_level_points[l] = 0;
        }
        cur_cell->num_finest_level_points[0] = 1;
        cur_cell->flag = 0;

        data_pt cur_point;
        cur_point.info = cur_cell;
        cur_point.info->parent_dist = 0.;
        cur_point.info->data_loc = &(data[points_on_level[i][j] * dim]);
        cur_point.info->parent_info = *root;
        bbulk[j].data_pt = cur_point;
        bbulk[j].parent_id = -1;
        for (k = 0; k < num_indices; k++) {
            cur_point.info->local_dist[k] = data_proj[k + points_on_level[i][j] * num_indices];
            btree_p_insert(&((*root)->cell_indices[k]), data_proj[k + points_on_level[i][j] * num_indices], cur_point);
        }
    }

    for (int k = 0; k < num_indices; k++) {
        for (j = 0; j < num_points_on_cur_levels; j++) {
            bbulk[j].local_parent_dist = data_proj[bbulk[j].data_pt.info->id * num_indices + k];
        }
        qsort(bbulk, num_points_on_cur_levels, sizeof(bulk_data_pt), dci_compare_data_pt_dist);
        for (j = 0; j < num_points_on_cur_levels; j++) {
            (*root)->arr_indices[k*num_points_on_cur_levels+j].info = bbulk[j].data_pt.info;
            (*root)->arr_indices[k*num_points_on_cur_levels+j].key = bbulk[j].local_parent_dist;
            if (k == 0) {
                bbulk[j].data_pt.info->local_id = j;
                (*root)->arr_indices[k*num_points_on_cur_levels+j].local_id = j;
            }
            else {
                (*root)->arr_indices[k*num_points_on_cur_levels+j].local_id = bbulk[j].data_pt.info->local_id;
            }

        }
    }
    (*root)->flag = 0;

    int max_num_points_on_level = num_points_on_level[i];
    for (i = (*actual_num_levels) - 2; i >= 0; i--) {
        if (num_points_on_level[i] > max_num_points_on_level) {
            max_num_points_on_level = num_points_on_level[i];
        }
    }

    float bulk_data_proj[max_num_points_on_level];
    data_pt bulk_data[max_num_points_on_level];
    bulk_data_pt bulk[max_num_points_on_level];
    int parent_idx[max_num_points_on_level + 1];

    num_points_on_upper_levels = num_points_on_cur_levels;

    for (i = (*actual_num_levels) - 2; i >= 0; i--) {
        assert(!data_proj_transposed);
        dci_assign_parent(num_comp_indices, num_simp_indices, dim,
            (*actual_num_levels), parallel_level, *root, (*actual_num_levels) - i - 1,
            num_points_on_level[i], points_on_level[i], data, *max_norm, norm_list,
            data_proj, construction_query_config,
            assigned_parent);

        num_points_on_cur_levels = num_points_on_level[i];

        btree_p* cur_index;
        for (j = 0; j < num_points_on_cur_levels; j++) {
            int k;
            int cur_id = assigned_parent[j].child;

            additional_info* cur_cell = level_cells_ret + cur_id;
            cur_cell->id = cur_id;
            cur_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
            cur_cell->cell_indices = NULL;
            cur_cell->arr_indices = NULL;
            cur_cell->num_finest_level_points = NULL;
            if (i) {  // we don't need to allocate for the finest level
                cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
                initialize_indices(cur_cell->cell_indices, num_indices);
                cur_cell->num_finest_level_points = (int*)malloc(sizeof(int) * (i + 1));
                for (int l = i; l >= 0; l--) {
                    cur_cell->num_finest_level_points[l] = 0;
                }
                cur_cell->num_finest_level_points[0] = 1;
            }
            cur_cell->flag = 0;

            data_pt cur_point;
            cur_point.info = cur_cell;
            cur_point.info->parent_dist = assigned_parent[j].dist;
            cur_point.info->data_loc = &(data[cur_id * dim]);
            cur_point.info->parent_info = assigned_parent[j].parent;
            bulk[j].data_pt = cur_point;
            bulk[j].parent_id = assigned_parent[j].parent->id;

            for (k = 0; k < num_indices; k++) {
                cur_point.info->local_dist[k] = data_proj[k + cur_id * num_indices];
            }
            for (additional_info* temp_cell = cur_cell->parent_info;
                temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                temp_cell->num_finest_level_points[i + 1] += 1;
                temp_cell->num_finest_level_points[0] += 1;
            }
        }

        qsort(bulk, num_points_on_cur_levels, sizeof(bulk_data_pt), dci_compare_data_pt_parent);
        int p_idx = 0;
        parent_idx[p_idx++] = 0;
        for (j=1; j < num_points_on_cur_levels; j++) {
            if (bulk[j].parent_id != bulk[j-1].parent_id) {
                bulk[j-1].data_pt.info->parent_info->arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*(j - parent_idx[p_idx-1])));
                parent_idx[p_idx++] = j;
            }
        }

        parent_idx[p_idx] = num_points_on_cur_levels;
        bulk[j-1].data_pt.info->parent_info->arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*(j - parent_idx[p_idx-1])));
        int p, begin, end;
        for (int k = 0; k < num_indices; k++) {
            for (j=0; j < num_points_on_cur_levels; j++) {
                bulk[j].local_parent_dist = data_proj[bulk[j].data_pt.info->id * num_indices + k];
            }
            for (p=0; p<p_idx; p++) {
                begin = parent_idx[p];
                end = parent_idx[p + 1];
                qsort(&(bulk[begin]), end - begin, sizeof(bulk_data_pt), dci_compare_data_pt_dist);
                additional_info* parent = bulk[begin].data_pt.info->parent_info;
                for (j = begin; j < end; j++) {
                    bulk_data_proj[j] = bulk[j].local_parent_dist;
                    bulk_data[j] = bulk[j].data_pt;
                    parent->arr_indices[k*(end-begin)+(j-begin)].info = bulk_data[j].info;
                    parent->arr_indices[k*(end-begin)+(j-begin)].key = bulk_data_proj[j];
                    if (k == 0) {
                        bulk_data[j].info->local_id = j-begin;
                        parent->arr_indices[k*(end-begin)+(j-begin)].local_id  = j-begin;
                    }
                    else {
                        parent->arr_indices[k*(end-begin)+(j-begin)].local_id  = bulk_data[j].info->local_id;
                    }

                }
                cur_index = &(bulk_data[begin].info->parent_info->cell_indices[k]);
                btree_p_bulk_load(cur_index, &(bulk_data_proj[begin]), &(bulk_data_proj[end]), &(bulk_data[begin]), &(bulk_data[end]));
            }
        }
        num_points_on_upper_levels += num_points_on_cur_levels;
    }

    assert(num_points_on_cur_levels == num_points_on_level[0]);
}

int dci_delete(dci* const dci_inst, const int num_points, const long long* const data_ids,
    dci_query_config deletion_config) {
        printf("Not used");
}

long long dci_add(dci* const dci_inst, const int dim, const int num_points,
    const float* const data, const int num_levels, bool* mask,
    dci_query_config construction_query_config, long long data_id, float* data_proj, int target_level) {
    int i, j, h, k;
    int new_data_level, num_points_on_cur_levels, num_points_on_upper_levels, new_data_num_points_on_cur_levels;
    additional_info *cur_cell, *prev_parent;
    int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;

    if (num_points == 0)
        return 0;

    assert(num_points > 0);

    dci_query_config construction_query_config_ = construction_query_config;
    construction_query_config_.num_to_visit = min_i(construction_query_config.num_to_visit, num_points);
    construction_query_config_.num_to_retrieve = min_i(construction_query_config.num_to_retrieve, num_points);

    if (num_points > 1) {
        construct_new_tree(dci_inst->num_comp_indices, dci_inst->num_simp_indices, dci_inst->parallel_level,
            dci_inst->proj_vec, dci_inst->add_proj_vec, &(dci_inst->root), dim, num_points,
            &new_data_level, data, dci_inst->num_points_on_level,
            data_proj, num_levels, dci_inst->info_addr, mask, dci_inst->points_on_level,
            construction_query_config_, dci_inst->norm_list, &(dci_inst->max_norm));
        
        dci_inst->num_levels = new_data_level;
    }
    else if (num_points == 1) {
        float* current_proj = data_proj + num_indices * data_id;

        if (num_levels -1 == target_level) {
            // Directly add to the root
            additional_info* cur_cell = (dci_inst->info_addr) + data_id;
            cur_cell->id = data_id;
            cur_cell->arr_indices = NULL;

            cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
            initialize_indices(cur_cell->cell_indices, num_indices);
            cur_cell->num_finest_level_points = (int*)malloc(sizeof(int)* num_levels);
            for (int l = target_level; l >= 0; l--) {
                cur_cell->num_finest_level_points[l] = 0;
            }
            cur_cell->num_finest_level_points[0] = 1;
            cur_cell->flag = 0;
            data_pt cur_point;
            cur_point.info = cur_cell;
            cur_point.info->parent_dist = 0.;
            cur_point.info->data_loc = &(data[data_id * dim]);
            cur_point.info->parent_info = dci_inst->root;
            cur_point.info->local_dist = (float*)malloc(sizeof(float) * num_indices);
            for (k = 0; k < num_indices; k++) {
                cur_point.info->local_dist[k] = current_proj[k];
            }
            for (k = 0; k < num_indices; k++) {
                btree_p_insert(&(dci_inst->root->cell_indices[k]), current_proj[k], cur_point);
            }
            update_arr_indices(num_indices, dci_inst->root);

            dci_inst->next_point_id += 1;
            dci_inst->num_points += 1;
            dci_inst->points_on_level[target_level][dci_inst->num_points_on_level[target_level]] = data_id;
            dci_inst->num_points_on_level[target_level] += 1;
        }
        else {
            additional_info* cur_cell = (dci_inst->info_addr) + data_id;
            cur_cell->id = data_id;

            tree_node assigned_parent[1];
            dci_assign_parent(dci_inst->num_comp_indices, dci_inst->num_simp_indices, dim,
                num_levels, dci_inst->parallel_level, dci_inst->root, num_levels - target_level - 1,
                1, &data_id, data, dci_inst->max_norm, dci_inst->norm_list, data_proj, 
                construction_query_config, assigned_parent);
            
            cur_cell->arr_indices = NULL;
            cur_cell->cell_indices = NULL;
            cur_cell->num_finest_level_points = NULL;
            cur_cell->flag = 0;

            data_pt cur_point;
            cur_point.info = cur_cell;
            cur_point.info->parent_dist = assigned_parent[0].dist;  // distance to the id
            cur_point.info->data_loc = &(data[data_id * dim]);
            cur_point.info->parent_info = assigned_parent[0].parent;
            cur_point.info->local_dist = (float*)malloc(sizeof(float) * num_indices);
            for (k = 0; k < num_indices; k++) {
                cur_point.info->local_dist[k] = current_proj[k];
            }

            for (additional_info* temp_cell = cur_cell->parent_info;
                temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                temp_cell->num_finest_level_points[target_level + 1] += 1;
                temp_cell->num_finest_level_points[0] += 1;
            }

            for (k = 0; k < num_indices; k++) {
                btree_p_insert(&(cur_cell->parent_info->cell_indices[k]), current_proj[k], cur_point);
            }
            // Use lazy update
            // update_arr_indices(num_indices, cur_cell->parent_info);
            cur_cell->parent_info->flag = 1;

            if (target_level) {
                // we don't need to allocate for the finest level
                cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
                initialize_indices(cur_cell->cell_indices, num_indices);
                cur_cell->num_finest_level_points = (int*)malloc(sizeof(int) * (target_level + 1));
                for (int l = target_level; l >= 0; l--) {
                    cur_cell->num_finest_level_points[l] = 0;
                }
                cur_cell->num_finest_level_points[0] = 1;

                int* points_on_lower_level = dci_inst->points_on_level[target_level - 1];
                float* query_proj;
                // for the target_level-1 level, decide if they need to change parent to new instered point
                for (int jj = dci_inst->num_points_on_level[target_level - 1] - 1; jj >= 0; jj--) {
                    int low_id = points_on_lower_level[jj];
                    additional_info* lower_cell = (dci_inst->info_addr) + low_id;
                    query_proj = data_proj + num_indices * low_id;
                    // caculater the distance
                    float cur_dist = compute_dist(lower_cell->data_loc, &(data[data_id * dim]), dim, dci_inst->max_norm, dci_inst->norm_list[low_id], dci_inst->norm_list[data_id]);

                    // need to change the parent if promote point is better
                    if (lower_cell->parent_dist - cur_dist > 1e-8) {
                        data_pt cur_data_point;
                        additional_info* prev_parent = lower_cell->parent_info;
                        cur_data_point.info = lower_cell;
                        cur_data_point.info->parent_info = cur_cell;
                        cur_data_point.info->parent_dist = cur_dist;
                        lower_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
                        for (k = 0; k < num_indices; k++) {
                            lower_cell->local_dist[k] = query_proj[k];
                        }

                        for (k = 0; k < num_indices; k++) {
                            btree_p_delete(&(prev_parent->cell_indices[k]), query_proj[k], lower_cell->id);
                            btree_p_insert(&(cur_cell->cell_indices[k]), query_proj[k], cur_data_point);
                        }
                        prev_parent->flag = 1;
                        cur_cell->flag = 1;

                        for (additional_info* temp_cell = prev_parent;
                            temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                            temp_cell->num_finest_level_points[target_level] -= 1;
                            assert(temp_cell->num_finest_level_points[target_level] >= 0);
                            if (target_level > 1) {
                                for (int l = 0; l <= target_level - 1; l++) {
                                    temp_cell->num_finest_level_points[l] -= cur_data_point.info->num_finest_level_points[l];
                                    assert(temp_cell->num_finest_level_points[l] >= 0);
                                }
                            }
                            else {
                                temp_cell->num_finest_level_points[0] -= 1;
                                assert(temp_cell->num_finest_level_points[0] >= 1);
                            }
                            if (temp_cell->parent_info == NULL) break;
                        }
                        for (additional_info* temp_cell = cur_cell;
                            temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                            temp_cell->num_finest_level_points[target_level] += 1;
                            if (target_level > 1) {
                                for (int l = 0; l <= target_level - 1; l++) {
                                    temp_cell->num_finest_level_points[l] += cur_data_point.info->num_finest_level_points[l];
                                }
                            }
                            else {
                                temp_cell->num_finest_level_points[0] += 1;
                            }
                            if (temp_cell->parent_info == NULL) break;
                        }
                    }
                }
            }

            dci_inst->next_point_id += 1;
            dci_inst->num_points += 1;
            dci_inst->points_on_level[target_level][dci_inst->num_points_on_level[target_level]] = data_id;
            dci_inst->num_points_on_level[target_level] += 1;
        }
    }

    return dci_inst->next_point_id - num_points;
}

static int dci_query_single_point_single_level(
    int num_comp_indices, int num_simp_indices, int dim,
    additional_info* point, int num_neighbours,
    const float* const query, const float* const query_proj, float max_norm, float* norm_list, int query_id,
    const dci_query_config query_config, idx_arr* const top_candidates, bool cumu) {
    int i, j, k, m, h, top_h;
    int num_indices = num_comp_indices * num_simp_indices;
    float top_index_priority, cur_dist;
    int num_returned_finest_level_points = 0;
    int num_candidates = 0;
    int num_points = point->cell_indices[0].num_data;
    float init;

    init = (-1e16);

    float last_top_candidate_dist = init;  // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;
    int num_returned = 0;
    idx_arr* arr_indices;
    int num_finest = 0;

    int left_pos[num_indices];
    int right_pos[num_indices];
    additional_info* cur_points;
    float index_priority[num_indices];
    float candidate_dists[num_points];
    bool checked[num_points];
    int returned_num[num_indices];
    int returned_ids[num_indices*CLOSEST];
    for (i = 0; i < num_indices; i++) {
        returned_num[i] = 0;
    }
    for (i = 0; i < num_indices*CLOSEST; i++) {
        returned_ids[i] = -1;
    }

    int bitnslots = BITNSLOTS(num_points);  // # of int that is needed to represent the # of points we have
    int min_slot = (bitnslots + SLOT_NUM - 1) / SLOT_NUM * SLOT_NUM;    // (# of int stored in the smallest # of whole SIMD registers that can represent # of points we have)

    assert(num_neighbours > 0);

    int num_points_to_retrieve =
        max_i(query_config.num_to_retrieve,
            (int)ceil(query_config.prop_to_retrieve * num_points));
    int num_projs_to_visit = max_i(
        query_config.num_to_visit * num_simp_indices,
        (int)ceil(query_config.prop_to_visit * num_points * num_simp_indices));

    int count_num = num_indices*min_slot;
    _Alignas(32) unsigned int mask[min_slot];
    _Alignas(32) unsigned int merged_bitarray[min_slot];
    _Alignas(32) unsigned int count_bitarray[count_num];

    for (i = 0; i < count_num; i++) {
        count_bitarray[i] = 0;
    }
    for (i = 0; i < min_slot; i++) {
        merged_bitarray[i] = 0;
    }
    for (i = 0; i < min_slot; i++) {
        mask[i] = -1;
    }

    if (point->flag == 1) {  // need to update arr_indices
       update_arr_indices(num_indices, point);
    }
    arr_indices = point->arr_indices;

    for (i = 0; i < num_points; i++) {
        candidate_dists[i] = init;
    }

    for (i = 0; i < num_points; i++) {
        checked[i] = 1;
    }

    for (i = 0; i < num_indices; i++) {
        left_pos[i] = dci_search_index(&(arr_indices[i*num_points]), query_proj[i], num_points);
        right_pos[i] = left_pos[i] + 1;
    }

    for (i = 0; i < num_indices; i++) {
        returned_num[i] = dci_next_closest_proj(&(arr_indices[i*num_points]), &(left_pos[i]), &(right_pos[i]), query_proj[i], num_points, &(returned_ids[i*CLOSEST]), &(index_priority[i]));
        if (returned_num[i] == 0) return 0;
    }

    int visit_proj = 0;

    k = 0;
    while (k < num_points * num_simp_indices * num_comp_indices) {
        visit_proj = 0;
        for (m = 0; m < num_comp_indices; m++) {
            top_index_priority = DBL_MAX;
            top_h = -1;
            for (h = 0; h < num_simp_indices; h++) {
                if (index_priority[h + m * num_simp_indices] < top_index_priority) {
                    top_index_priority = index_priority[h + m * num_simp_indices];
                    top_h = h;
                }
            }
            if (top_h >= 0) {
                i = top_h + m * num_simp_indices;
                visit_proj += returned_num[i];
                for (j = 0; j < returned_num[i]; j++) {
                    BITSET(&(count_bitarray[i*min_slot]), returned_ids[i*CLOSEST+j]);
                }
                for (j = 0; j < min_slot; j++) {
                    merged_bitarray[j] = count_bitarray[m*num_simp_indices*min_slot+j];
                }
                for (j = m * num_simp_indices+1; j < (m+1) * num_simp_indices; j++) {
                    BitAnd(&(count_bitarray[j*min_slot]), merged_bitarray, min_slot);
                }
                BitAnd(mask, merged_bitarray, min_slot);
                BitNot_And(merged_bitarray, mask, min_slot);
                // --------------------------------------------
                for (j = 0; j < num_points; j++) {
                    if(BITTEST(merged_bitarray, j)) {
                        int local_idx = j;
                        cur_points = arr_indices[local_idx].info;
                        if (query_config.min_num_finest_level_points > 0) {
                            num_finest = cur_points->num_finest_level_points[query_config.target_level];
                        }
                        if (!(query_config.blind)) {
                            if (checked[local_idx]) {
                                // Compute distance
                                // don't need to do sqrt at the end since our aim is to compare distances
                                cur_dist = compute_dist_query(cur_points->data_loc, query, dim);
                                candidate_dists[local_idx] = cur_dist;
                                checked[local_idx] = 0;

                                add_to_list(num_candidates, num_neighbours, top_candidates, &num_returned,
                                                    cur_dist, cur_points, &last_top_candidate_dist, query_config,
                                                    &num_returned_finest_level_points, &last_top_candidate, i, init, num_finest);
                                num_candidates++;
                            }
                            else {
                                cur_dist = candidate_dists[local_idx];
                            }
                        }
                        else {
                            if (checked[local_idx]) {
                                if (num_finest > 0) {
                                    candidate_dists[local_idx] = top_index_priority;
                                    checked[local_idx] = 0;
                                    top_candidates[num_candidates].info = cur_points;
                                    num_candidates++;
                                    if (query_config.min_num_finest_level_points > 0) {
                                        num_returned_finest_level_points += num_finest;
                                    }
                                }
                            }
                            else if (top_index_priority > candidate_dists[local_idx]) {
                                candidate_dists[local_idx] = top_index_priority;
                            }
                        }
                    }
                }
                returned_num[i] = dci_next_closest_proj(&(arr_indices[i*num_points]), &(left_pos[i]), &(right_pos[i]), query_proj[i], num_points, &(returned_ids[i*CLOSEST]), &(index_priority[i]));
                if (returned_num[i] == 0) {
                    index_priority[i] = DBL_MAX;
                }
            }
        }
        if (num_candidates >= num_neighbours &&
            num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
            if (k + visit_proj >= num_projs_to_visit || num_candidates >= num_points_to_retrieve) {
                break;
            }
        }
        k += visit_proj;
    }

    if (query_config.blind) {
        for (int j = 0; j < num_candidates; j++) {
            top_candidates[j].key = candidate_dists[top_candidates[j].info->local_id];
        }
        num_returned = min_i(num_candidates, num_points_to_retrieve);
    }
    else {
        if (num_returned > num_neighbours) {
            qsort(top_candidates, num_returned, sizeof(idx_arr), dci_compare_data_idx_arr_dist);
            if (query_config.min_num_finest_level_points > 0) {
                num_returned_finest_level_points = 0;
                int j = 0;
                for (j = 0; j < num_returned-1; j++) {
                    num_returned_finest_level_points += top_candidates[j].info->num_finest_level_points[query_config.target_level];
                    if (num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
                        break;
                    }
                }
                num_returned = max_i(min_i(num_neighbours, num_points), j + 1);
            }
            else {
                num_returned = num_neighbours;
            }
        }
    }

    return num_returned;
}

static int dci_query_single_point_single_level_(
    int num_comp_indices, int num_simp_indices, int dim,
    additional_info* point, int num_neighbours,
    const float* const query, const float* const query_proj, float max_norm, float* norm_list, int query_id,
    const dci_query_config query_config, idx_arr* const top_candidates, bool cumu) {
    int i, j, k, m, h, top_h;
    int num_indices = num_comp_indices * num_simp_indices;
    float top_index_priority, cur_dist;
    int cur_pos;
    int num_returned_finest_level_points = 0;
    int num_candidates = 0;
    int num_points = point->cell_indices[0].num_data;
    float init;
    int num_finest = 0;

    init = -1.0;

    float last_top_candidate_dist = init;  // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;
    int num_returned = 0;
    idx_arr* arr_indices;

    int left_pos[num_indices];
    int right_pos[num_indices];
    additional_info* cur_points[num_indices];
    float index_priority[num_indices];
    bool checked[num_points];
    i = exp2(ceil(log2(num_comp_indices*num_points)));
    int counts[i];
    float candidate_dists[num_points];

    assert(num_neighbours > 0);

    int num_points_to_retrieve =
        max_i(query_config.num_to_retrieve,
            (int)ceil(query_config.prop_to_retrieve * num_points));
    int num_projs_to_visit = max_i(
        query_config.num_to_visit * num_simp_indices,
        (int)ceil(query_config.prop_to_visit * num_points * num_simp_indices));

    if (point->flag == 1) {  // need to update arr_indices
       update_arr_indices(num_indices, point);
    }
    arr_indices = point->arr_indices;

    for (i = 0; i < num_comp_indices*num_points; i++) {
        counts[i] = 0;
    }
    for (i = 0; i < num_points; i++) {
        candidate_dists[i] = init;
    }
    for (i = 0; i < num_points; i++) {
        checked[i] = 1;
    }

    for (i = 0; i < num_indices; i++) {
        left_pos[i] = dci_search_index(&(arr_indices[i*num_points]), query_proj[i], num_points);
        right_pos[i] = left_pos[i] + 1;
    }

    for (i = 0; i < num_indices; i++) {
        cur_pos = dci_next_closest_proj_(&(arr_indices[i*num_points]), &(left_pos[i]), &(right_pos[i]), query_proj[i], num_points);
        // assert(cur_pos >= 0);    // There should be at least one point in the index
        if (cur_pos == -1) return 0;
        index_priority[i] = abs_d(arr_indices[cur_pos+i*num_points].key - query_proj[i]);
        cur_points[i] = arr_indices[cur_pos+i*num_points].info;
    }

    k = 0;
    while (k < num_points * num_simp_indices) {
        for (m = 0; m < num_comp_indices; m++) {
            top_index_priority = DBL_MAX;
            top_h = -1;
            for (h = 0; h < num_simp_indices; h++) {
                if (index_priority[h + m * num_simp_indices] < top_index_priority) {
                    top_index_priority = index_priority[h + m * num_simp_indices];
                    top_h = h;
                }
            }
            if (top_h >= 0) {
                i = top_h + m * num_simp_indices;
                int local_idx = cur_points[i]->local_id;
                if (query_config.min_num_finest_level_points > 0) {
                    num_finest = cur_points[i]->num_finest_level_points[query_config.target_level];
                }
                counts[local_idx+m*num_points]++;
                if (counts[local_idx+m*num_points] == num_simp_indices) {
                    if (!(query_config.blind)) {
                        if (checked[local_idx]) {
                            // Compute distance
                            cur_dist = compute_dist(cur_points[i]->data_loc, query, dim, max_norm, norm_list[cur_points[i]->id], norm_list[query_id] );
                            candidate_dists[local_idx] = cur_dist;
                            checked[local_idx] = 0;

                            if (query_config.min_num_finest_level_points == 0 || num_finest > 0) {
                                add_to_list(num_candidates, num_neighbours, top_candidates, &num_returned,
                                                    cur_dist, cur_points[i], &last_top_candidate_dist, query_config,
                                                    &num_returned_finest_level_points, &last_top_candidate, i, init, num_finest);
                                num_candidates++;
                            }
                        }
                        else {
                            cur_dist = candidate_dists[local_idx];
                        }
                    }
                    else {
                        if (checked[local_idx]) {
                            if (num_finest > 0) {
                                candidate_dists[local_idx] = top_index_priority;
                                checked[local_idx] = 0;
                                top_candidates[num_candidates].info = cur_points[i];
                                num_candidates++;
                                if (query_config.min_num_finest_level_points > 0) {
                                    num_returned_finest_level_points += num_finest;
                                }
                            }
                        }
                        else if (top_index_priority > candidate_dists[local_idx]) {
                            candidate_dists[local_idx] = top_index_priority;
                        }
                    }
                }
                cur_pos = dci_next_closest_proj_(&(arr_indices[i*num_points]), &(left_pos[i]), &(right_pos[i]), query_proj[i], num_points);
                if (cur_pos >= 0) {
                    index_priority[i] = abs_d(arr_indices[cur_pos+i*num_points].key - query_proj[i]);
                    cur_points[i] = arr_indices[cur_pos+i*num_points].info;
                }
                else {
                    index_priority[i] = DBL_MAX;
                }
            }
        }
        if (num_candidates >= num_neighbours &&
            num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
            if (k + 1 >= num_projs_to_visit || num_candidates >= num_points_to_retrieve) {
                break;
            }
        }
        k++;
    }

    if (query_config.blind) {
        for (int j = 0; j < num_candidates; j++) {
            top_candidates[j].key = candidate_dists[top_candidates[j].info->local_id];
        }
        num_returned = min_i(num_candidates, num_points_to_retrieve);
    }
    else {
        if (num_returned > num_neighbours) {
            qsort(top_candidates, num_returned, sizeof(idx_arr), dci_compare_data_idx_arr_dist);
            if (query_config.min_num_finest_level_points > 0) {
                num_returned_finest_level_points = 0;
                int j = 0;
                for (j = 0; j < num_returned-1; j++) {
                    num_returned_finest_level_points += top_candidates[j].info->num_finest_level_points[query_config.target_level];
                    if (num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
                        break;
                    }
                }
                num_returned = max_i(min_i(num_neighbours, num_points), j + 1);
            }
            else {
                num_returned = num_neighbours;
            }
        }
    }

    return num_returned;
}

static int dci_query_single_point(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels, int parallel_level,
    additional_info* root, int num_populated_levels, int num_neighbours, idx_arr* points_to_expand,
    idx_arr** points_to_expand_next, int* num_top_candidates,
    const float* const query, const float* const query_proj, float max_norm, float* norm_list, int query_id,
    dci_query_config query_config, idx_arr* const top_candidates, bool cumu) {
    int i, j, k;
    int init = (-1e16);
    int candidates_num = 0;
    float last_top_candidate_dist = init;  // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;
    int num_indices = num_comp_indices * num_simp_indices;
    int num_points_to_expand;
    int max_num_points_to_expand = max_i(query_config.field_of_view, num_neighbours);
    if (query_config.blind) {
        max_num_points_to_expand += num_comp_indices - 1;
    }
    int num_finest_level_points_to_expand;

    assert(num_populated_levels <= num_levels);

    int temp_idx = 0;
    if (num_populated_levels <= 1) {
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;

        num_points_to_expand = dci_query_single_point_single_level(
            num_comp_indices, num_simp_indices, dim, root,
            num_neighbours, query, query_proj, max_norm, norm_list, query_id,
            query_config, points_to_expand, cumu);

        temp_idx = num_points_to_expand;
    }
    else {
        assert(query_config.field_of_view > 0);

        if (query_config.blind) {
            query_config.num_to_retrieve = query_config.field_of_view;
            query_config.prop_to_retrieve = -1.0;
        }
        if (cumu) {
            query_config.target_level = 0;
        }
        else {
            query_config.target_level = num_levels - num_populated_levels + 1;
        }
        query_config.min_num_finest_level_points = num_neighbours;
        num_points_to_expand = dci_query_single_point_single_level(
            num_comp_indices, num_simp_indices, dim, root,
            query_config.field_of_view, query,
            query_proj, max_norm, norm_list, query_id, query_config, points_to_expand, cumu);
        assert(num_points_to_expand > 0);

        for (i = num_levels - 2; i >= num_levels - num_populated_levels + 1; i--) {
            if (cumu && num_populated_levels > 1) {
                idx_arr cur = points_to_expand[0];
                for (j = 0; j < num_points_to_expand; j++) {
                    if (candidates_num >= num_neighbours) {
                        if (top_candidates[last_top_candidate].key > cur.key) {
                            top_candidates[last_top_candidate] = cur;
                            last_top_candidate_dist = init;
                            for (int jj = 0; jj < candidates_num; jj++) {
                                if (top_candidates[jj].key > last_top_candidate_dist) {
                                    last_top_candidate_dist = top_candidates[jj].key;
                                    last_top_candidate = jj;
                                }
                            }
                        }
                        else
                            break;
                    }
                    else {
                        top_candidates[candidates_num++] = cur;
                        if (cur.key > last_top_candidate_dist) {
                            last_top_candidate_dist = cur.key;
                            last_top_candidate = candidates_num-1;
                        }
                    }
                    cur = points_to_expand[j+1];
                }
            }
// #pragma omp parallel for if(parallel_level >= 3)
            for (int j = 0; j < num_points_to_expand; j++) {
                additional_info* point = points_to_expand[j].info;
                num_top_candidates[j] = dci_query_single_point_single_level(
                    num_comp_indices, num_simp_indices, dim, point,
                    query_config.field_of_view, query,
                    query_proj, max_norm, norm_list, query_id, query_config, points_to_expand_next[j], cumu);

                assert(num_top_candidates[j] <= max_num_points_to_expand);
            }

            temp_idx = 0;
            for (int j = 0; j < num_points_to_expand; j++) {
                for (k = 0; k < num_top_candidates[j]; k++) {
                    points_to_expand[temp_idx++] = points_to_expand_next[j][k];
                }
            }
            qsort(points_to_expand, temp_idx, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

            if (num_neighbours > 1) {
                num_finest_level_points_to_expand = 0;
                k = 0;
                if (cumu && num_populated_levels > 1) {
                    for (int j = 0; j < temp_idx; j++) {
                        assert(points_to_expand[j].info->num_finest_level_points[query_config.target_level] > 0);
                        num_finest_level_points_to_expand +=
                            points_to_expand[j].info->num_finest_level_points[query_config.target_level];
                        k++;
                        if (num_finest_level_points_to_expand >= num_neighbours) {
                            break;
                        }
                    }
                }
                else {
                    for (int j = 0; j < temp_idx; j++) {
                        assert(points_to_expand[j].info->num_finest_level_points[query_config.target_level] > 0);
                        num_finest_level_points_to_expand +=
                            points_to_expand[j].info->num_finest_level_points[query_config.target_level];
                        k++;
                        if (num_finest_level_points_to_expand >= num_neighbours) {
                            break;
                        }
                    }
                }
                num_points_to_expand = max_i(min_i(query_config.field_of_view, temp_idx), k);
            }
            else {
                num_points_to_expand = min_i(query_config.field_of_view, temp_idx);
            }
        }
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;

        if (cumu && num_populated_levels > 1) {
            idx_arr cur = points_to_expand[0];
            for (j = 0; j < num_points_to_expand; j++) {
                if (candidates_num >= num_neighbours) {
                    if (top_candidates[last_top_candidate].key > cur.key) {
                        top_candidates[last_top_candidate] = cur;
                        last_top_candidate_dist = init;
                        for (int jj = 0; jj < candidates_num; jj++) {
                            if (top_candidates[jj].key > last_top_candidate_dist) {
                                last_top_candidate_dist = top_candidates[jj].key;
                                last_top_candidate = jj;
                            }
                        }
                    }
                    else
                        break;
                }
                else {
                    top_candidates[candidates_num++] = cur;
                    if (cur.key > last_top_candidate_dist) {
                        last_top_candidate_dist = cur.key;
                        last_top_candidate = candidates_num-1;
                    }
                }
                cur = points_to_expand[j+1];
            }
        }
// #pragma omp parallel for if(parallel_level >= 3)
        for (int j = 0; j < num_points_to_expand; j++) {
            additional_info* point = points_to_expand[j].info;
            num_top_candidates[j] = dci_query_single_point_single_level(
                num_comp_indices, num_simp_indices, dim, point,
                num_neighbours, query, query_proj, max_norm, norm_list, query_id,
                query_config, points_to_expand_next[j], cumu);

            assert(num_top_candidates[j] <= max_num_points_to_expand);
        }

        temp_idx = 0;
        for (int j = 0; j < num_points_to_expand; j++) {
            for (k = 0; k < num_top_candidates[j]; k++) {
                points_to_expand[temp_idx++] = points_to_expand_next[j][k];
            }
        }
        qsort(points_to_expand, temp_idx, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

        num_points_to_expand = min_i(num_neighbours, temp_idx);
    }

    assert(num_points_to_expand <= temp_idx);

    if (cumu && num_populated_levels > 1) {
        idx_arr cur = points_to_expand[0];
        for (int j = 0; j < temp_idx; j++) {
            if (candidates_num >= num_neighbours) {
                if (top_candidates[last_top_candidate].key > cur.key) {
                    top_candidates[last_top_candidate] = cur;
                    last_top_candidate_dist = init;
                    for (int jj = 0; jj < candidates_num; jj++) {
                        if (top_candidates[jj].key > last_top_candidate_dist) {
                            last_top_candidate_dist = top_candidates[jj].key;
                            last_top_candidate = jj;
                        }
                    }
                }
                else
                    break;
            }
            else {
                top_candidates[candidates_num++] = cur;
                if (cur.key > last_top_candidate_dist) {
                    last_top_candidate_dist = cur.key;
                    last_top_candidate = candidates_num-1;
                }
            }
            cur = points_to_expand[j+1];
        }
        num_points_to_expand = min_i(num_neighbours, candidates_num);
    }
    else {
        assert(num_points_to_expand > 0);
        idx_arr cur = points_to_expand[0];
        for (j = 0; j < num_points_to_expand; j++) {
            top_candidates[candidates_num++] = cur;
            cur = points_to_expand[j+1];
        }
    }

    qsort(top_candidates, candidates_num, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

    return num_points_to_expand;
}

static inline int dci_query_single_point_(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels, int parallel_level,
    additional_info* root, int num_populated_levels, int num_neighbours,
    const float* const query, const float* const query_proj, float max_norm, float* norm_list, int query_id,
    dci_query_config query_config, idx_arr* const top_candidates, bool cumu) {
    int i, j, k;
    int init = (-1e16);
    int candidates_num = 0;
    float last_top_candidate_dist = init;  // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;
    int num_indices = num_comp_indices * num_simp_indices;
    int num_points_to_expand;
    int max_num_points_to_expand = max_i(query_config.field_of_view, num_neighbours);
    if (query_config.blind) {
        max_num_points_to_expand += num_comp_indices - 1;
    }
    idx_arr points_to_expand[max_num_points_to_expand * max_num_points_to_expand];
    idx_arr points_to_expand_next[max_num_points_to_expand * max_num_points_to_expand];

    int num_top_candidates[max_num_points_to_expand];
    int num_finest_level_points_to_expand;

    assert(num_populated_levels <= num_levels);

    int temp_idx = 0;
    if (num_populated_levels <= 1) {
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;

        num_points_to_expand = dci_query_single_point_single_level_(
            num_comp_indices, num_simp_indices, dim, root,
            num_neighbours, query, query_proj, max_norm, norm_list, query_id,
            query_config, points_to_expand, cumu);

        temp_idx = num_points_to_expand;
    }
    else {
        assert(query_config.field_of_view > 0);

        if (query_config.blind) {
            query_config.num_to_retrieve = query_config.field_of_view;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.target_level = num_levels - num_populated_levels + 1;
        query_config.min_num_finest_level_points = num_neighbours;
        num_points_to_expand = dci_query_single_point_single_level_(
            num_comp_indices, num_simp_indices, dim, root,
            query_config.field_of_view, query,
            query_proj, max_norm, norm_list, query_id, query_config, points_to_expand, cumu);
        assert(num_points_to_expand > 0);

        for (i = num_levels - 2; i >= num_levels - num_populated_levels + 1; i--) {
#pragma omp parallel for if(parallel_level >= 3)
            for (int j = 0; j < num_points_to_expand; j++) {
                additional_info* point = points_to_expand[j].info;
                num_top_candidates[j] = dci_query_single_point_single_level_(
                    num_comp_indices, num_simp_indices, dim, point,
                    query_config.field_of_view, query,
                    query_proj, max_norm, norm_list, query_id, query_config, &points_to_expand_next[j * max_num_points_to_expand], cumu);

                assert(num_top_candidates[j] <= max_num_points_to_expand);
            }

            temp_idx = 0;
            for (int j = 0; j < num_points_to_expand; j++) {
                for (k = 0; k < num_top_candidates[j]; k++) {
                    points_to_expand[temp_idx++] = points_to_expand_next[j * max_num_points_to_expand + k];
                }
            }
            qsort(points_to_expand, temp_idx, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

            num_points_to_expand = min_i(query_config.field_of_view, temp_idx);
        }
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;
#pragma omp parallel for if(parallel_level >= 3)
        for (int j = 0; j < num_points_to_expand; j++) {
            additional_info* point = points_to_expand[j].info;
            num_top_candidates[j] = dci_query_single_point_single_level_(
                num_comp_indices, num_simp_indices, dim, point,
                num_neighbours, query, query_proj, max_norm, norm_list, query_id,
                query_config, &points_to_expand_next[j * max_num_points_to_expand], cumu);

            assert(num_top_candidates[j] <= max_num_points_to_expand);
        }

        temp_idx = 0;
        for (int j = 0; j < num_points_to_expand; j++) {
            for (k = 0; k < num_top_candidates[j]; k++) {
                points_to_expand[temp_idx++] = points_to_expand_next[j * max_num_points_to_expand + k];
            }
        }
        qsort(points_to_expand, temp_idx, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

        num_points_to_expand = min_i(num_neighbours, temp_idx);
    }

    assert(num_points_to_expand <= temp_idx);

    assert(num_points_to_expand > 0);
    idx_arr cur = points_to_expand[0];
    for (j = 0; j < num_points_to_expand; j++) {
        top_candidates[candidates_num++] = cur;
        cur = points_to_expand[j+1];
    }

    qsort(top_candidates, candidates_num, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

    return num_points_to_expand;
}

static void dci_assign_parent(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels, int parallel_level,
    additional_info* root, const int num_populated_levels, const int num_queries,
    const int* selected_query_pos, const float* const query, float max_norm, float* norm_list,
    const float* const query_proj, const dci_query_config query_config,
    tree_node* const assigned_parent) {
    int num_indices = num_comp_indices * num_simp_indices;

#pragma omp parallel for if(parallel_level >= 2)
    for (int j = 0; j < num_queries; j++) {
        int cur_num_returned;
        idx_arr top_candidate;

        cur_num_returned = dci_query_single_point_(
            num_comp_indices, num_simp_indices, dim, num_levels, parallel_level,
            root, num_populated_levels, 1,
            &(query[((long long int) selected_query_pos[j]) * dim]),
            &(query_proj[selected_query_pos[j] * num_indices]), max_norm, norm_list, selected_query_pos[j], query_config,
            &top_candidate, false);

        assigned_parent[j].parent = top_candidate.info;
        assigned_parent[j].dist = top_candidate.key;
        assigned_parent[j].child = selected_query_pos[j];

    }
}

void dci_query(dci* const dci_inst, const int dim,
    const int num_queries, const float* const query,
    const int num_neighbours,
    dci_query_config query_config, bool* mask,
    int** const nearest_neighbours,
    float** const nearest_neighbour_dists,
    int* const num_returned, float scale, int num_populated_levels) {  //const long long* const data_ids
    int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;

    int j;
    float* query_proj;
    srand(time(NULL));
    assert(dci_inst->root != NULL);
    assert(dim == dci_inst->dim);
    assert(num_neighbours > 0);
    if (posix_memalign((void**)&query_proj, 32, sizeof(float) * num_indices * num_queries) != 0) {
        perror("Memory allocation failed!\n");
        return;
    }
    matmul(num_indices, num_queries, dim, dci_inst->proj_vec, query, query_proj);
    query_transform(query, num_queries, dim, query_proj, num_indices);

    int num_neighbours_ = num_neighbours;
    if (num_neighbours > dci_inst->num_points) {
        num_neighbours_ = dci_inst->num_points;
    }
    if (query_config.field_of_view > dci_inst->num_points) {
        query_config.field_of_view = dci_inst->num_points;
    }
    if (num_populated_levels == 0) {
        num_populated_levels = dci_inst->num_levels;
    }

    query_config.num_to_visit = min_i(query_config.num_to_visit, dci_inst->num_points);
    query_config.num_to_retrieve = min_i(query_config.num_to_retrieve, dci_inst->num_points);

    float sqrt_dim = sqrt(dim);
    int max_thread = 1;
    #ifdef USE_OPENMP
    if (dci_inst->parallel_level >= 2) {
        max_thread = omp_get_max_threads();
    }
    #endif
    int max_num_points_to_expand = max_i(query_config.field_of_view, num_neighbours_);
    idx_arr* points_to_expand = (idx_arr *)malloc(sizeof(idx_arr) * max_num_points_to_expand * max_num_points_to_expand * max_thread);
    idx_arr** points_to_expand_next = (idx_arr**)malloc(sizeof(idx_arr*) * max_num_points_to_expand * max_thread);
    for (j = 0; j < max_num_points_to_expand * max_thread; j++) {
        points_to_expand_next[j] = (idx_arr *)malloc(sizeof(idx_arr) * max_num_points_to_expand);
    }
    int* num_top_candidates = (int*)malloc(sizeof(int)*max_num_points_to_expand * max_thread);

#pragma omp parallel for if(dci_inst->parallel_level >= 2)
    for (j = 0; j < num_queries; j++) {
        if (!(mask[j])) continue;
        int t_index = 0;
        #ifdef USE_OPENMP
        if (dci_inst->parallel_level >= 2) {
            t_index = omp_get_thread_num();
        }
        #endif
        int i;
        int cur_num_returned;
        idx_arr top_candidate[num_neighbours_];

        cur_num_returned = dci_query_single_point(
            dci_inst->num_comp_indices, dci_inst->num_simp_indices,
            dci_inst->dim, dci_inst->num_levels, dci_inst->parallel_level, dci_inst->root,
            num_populated_levels, num_neighbours_, &(points_to_expand[max_num_points_to_expand * max_num_points_to_expand * t_index]),
            &(points_to_expand_next[max_num_points_to_expand * t_index]), &(num_top_candidates[max_num_points_to_expand * t_index]),
            &(query[j * dim]), &(query_proj[j * num_indices]), dci_inst->max_norm, dci_inst->norm_list, 0, query_config, top_candidate, true);

        nearest_neighbours[j] = (int*)malloc(sizeof(int) * cur_num_returned);
        if (num_returned) {
            num_returned[j] = cur_num_returned;
        }
        if (nearest_neighbour_dists) {
            nearest_neighbour_dists[j] = (float*)malloc(sizeof(float) * cur_num_returned);
        }
        btree_p_search_res k;
        float exp_sum = 0.0;
        float temp_exp = 0.0;
        float max_norm = top_candidate[0].key;
        for (i = 0; i < cur_num_returned; i++) {
            nearest_neighbours[j][i] = top_candidate[i].info->id;
            if (nearest_neighbour_dists) {
                temp_exp = exp(-1*(top_candidate[i].key - max_norm) / sqrt_dim  * scale);  // Scaled-dot-product+softmax
                nearest_neighbour_dists[j][i] = temp_exp;
                exp_sum += temp_exp;
            }
        }
        if (nearest_neighbour_dists){
            for (i = 0; i < cur_num_returned; i++) {
                    nearest_neighbour_dists[j][i] /= exp_sum;
            }
        }
    }
    for (int i = 0; i < max_num_points_to_expand * max_thread; i++) {
        free(points_to_expand_next[i]);
    }
    free(points_to_expand);
    free(points_to_expand_next);
    free(num_top_candidates);
    free(query_proj);
}