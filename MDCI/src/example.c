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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dci.h"
#include "util.h"
#include "hashtable_p.h"

int main(int argc, char **argv) {
    srand48(time(NULL));

    int j, k;

    int dim = 5000;
    int intrinsic_dim = 50;
    int num_points = 10000;
    int num_queries = 5;
    int num_neighbours = 8;  // The k in k-NN

    // Guide for tuning hyperparameters:

    // num_comp_indices trades off accuracy vs. construction and query time -
    // high values lead to more accurate results, but slower construction and
    // querying num_simp_indices trades off accuracy vs. construction and query
    // time - high values lead to more accurate results, but slower construction
    // and querying; if num_simp_indices is increased, may need to increase
    // num_comp_indices num_levels trades off construction time vs. query time -
    // higher values lead to faster querying, but slower construction; if
    // num_levels is increased, may need to increase query_field_of_view and
    // construction_field_of_view construction_field_of_view trades off
    // accuracy/query time vs. construction time - higher values lead to
    // *slightly* more accurate results and/or *slightly* faster querying, but
    // *slightly* slower construction construction_prop_to_retrieve trades off
    // acrruacy vs. construction time - higher values lead to *slightly* more
    // accurate results, but slower construction query_field_of_view trades off
    // accuracy vs. query time - higher values lead to more accurate results,
    // but *slightly* slower querying query_prop_to_retrieve trades off accuracy
    // vs. query time - higher values lead to more accurate results, but slower
    // querying

    int num_comp_indices = 2;
    int num_simp_indices = 7;
    int num_levels = 2;
    int construction_field_of_view = 10;
    float construction_prop_to_retrieve = 0.002;
    int query_field_of_view = 170;
    float query_prop_to_retrieve = 0.8;
    float lambda = 0.5;   // less than one, not used in the current version

    // Generate data
    // Assuming column-major layout, data is dim x num_points
    float *data;
    assert(posix_memalign((void **) &data, 32, sizeof(float) * dim * (num_points + num_queries)) == 0);
    gen_data(data, dim, intrinsic_dim, num_points + num_queries);
    // Assuming column-major layout, query is dim x num_queries
    float *query = data + dim * ((long long int) num_points);
    // print_matrix(data, dim, num_points);

    // Mask for data points
    bool *mask;
    mask = (bool*)malloc(sizeof(bool) * num_points);
    for (int i = 0; i < num_points; i++) {
        mask[i] = true;
    }

    dci dci_inst;
    dci_init(&dci_inst, dim, num_comp_indices, num_simp_indices, lambda, num_points);
    dci_inst.parallel_level = 3;

    // print_matrix(dci_inst.proj_vec, dim, num_comp_indices*num_simp_indices);

    float* data_proj;

    dci_inst.num_levels = num_levels;
    dci_inst.info_addr = (additional_info*)malloc(sizeof(additional_info)*num_points);
    dci_inst.num_points_on_level = (int*)malloc(sizeof(int)*num_levels);
    dci_inst.points_on_level = (int**)malloc(sizeof(int*)*num_levels);
    for (int i = 0; i < num_levels; i++) {
        dci_inst.num_points_on_level[i] = 0;
        dci_inst.points_on_level[i] = (int*)malloc(sizeof(int)*num_points);
    }
    dci_inst.norm_list = (float*)malloc(sizeof(float)*num_points);

    data_projection(num_comp_indices, num_simp_indices,
        dci_inst.proj_vec, dci_inst.add_proj_vec, &(dci_inst.root), dim, num_points,
        data, mask, dci_inst.norm_list, &(dci_inst.max_norm), &data_proj);

    initialize_tree(num_comp_indices, num_simp_indices, &(dci_inst.root));

    // Add and Query
    dci_query_config construction_query_config;

    construction_query_config.blind = false;
    construction_query_config.num_to_visit = 5000;
    construction_query_config.num_to_retrieve = -1;
    construction_query_config.prop_to_visit = 1.0;
    construction_query_config.prop_to_retrieve = construction_prop_to_retrieve;
    construction_query_config.field_of_view = construction_field_of_view;
    construction_query_config.target_level = 0;
    long long* d_ids = NULL;

    dci_query_config query_config;

    query_config.blind = false;
    query_config.num_to_visit = 5000;
    query_config.num_to_retrieve = -1;
    query_config.prop_to_visit = 1.0;
    query_config.prop_to_retrieve = query_prop_to_retrieve;
    query_config.field_of_view = query_field_of_view;
    query_config.target_level = 0;
    float scale = 1.0;

    // Assuming column-major layout, matrix is of size num_neighbours x
    // num_queries
    int **nearest_neighbours = (int **) malloc(sizeof(int *) * num_queries);
    float **nearest_neighbour_dists =
            (float **) malloc(sizeof(float *) * num_queries);
    int *num_returned = (int *) malloc(sizeof(int) * num_queries);

    for (int i = 0; i < 1; i++) {
        long long first_id = dci_add(&dci_inst, dim, num_points, data,
                                        num_levels, mask, construction_query_config, 0, data_proj, 0);
            
        dci_inst.next_point_id += num_points;
        dci_inst.num_points += num_points;
        
        dci_query(&dci_inst, dim, num_queries, query, num_neighbours, query_config, 
                            mask, nearest_neighbours, nearest_neighbour_dists, 
                            num_returned, scale, num_levels);

        for (j = 0; j < num_queries; j++) {
            printf("%d: ", j + 1);
            for (k = 0; k < num_returned[j]; k++) {
                printf("%d: %.4f, ", nearest_neighbours[j][k],
                       nearest_neighbour_dists[j][k]);
            }
            printf("%d: %.4f\n", nearest_neighbours[j][num_neighbours - 1],
                   nearest_neighbour_dists[j][num_neighbours - 1]);
        }
        for (j = 0; j < num_queries; j++) {
            free(nearest_neighbours[j]);
        }
        for (j = 0; j < num_queries; j++) {
            free(nearest_neighbour_dists[j]);
        }
    }

    dci_free(&dci_inst);
    free(nearest_neighbours);
    free(nearest_neighbour_dists);
    free(num_returned);
    free(data);
    free(mask);

    return 0;
}
