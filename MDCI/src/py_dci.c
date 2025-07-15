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

#include "Python.h"
#include "numpy/arrayobject.h"
#include "dci.h"
#include "hashtable_pp.h"
#ifndef NO_SIMD
#include<immintrin.h>
#include <x86intrin.h>
#endif
#include <assert.h>

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

#ifdef PY3K

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

#endif

// DCI struct with some additional structures for Python-specific bookkeeping
typedef struct py_dci {
    dci dci_inst;
    hashtable_pp hashtable;
    // PyArrayObject *py_array;
} py_dci;


static inline void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    // Use your specific gc implementation in place of free if you have to
    free(memory);
}

// Called automatically by the garbage collector
static void py_dci_free(PyObject *py_dci_inst_wrapper) {
    
    py_dci *py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    // if (py_dci_inst->py_array) {
    //     Py_DECREF(py_dci_inst->py_array);
    // }
    hashtable_pp_free(&(py_dci_inst->hashtable));
    
    if (py_dci_inst->dci_inst.num_points > 0) {
        dci_free(&(py_dci_inst->dci_inst));
    }
    
    // free(py_dci_inst);
}

static PyObject *py_dci_new(PyObject *self, PyObject *args) {
    
    int dim, num_comp_indices, num_simp_indices, max_volume;
    float lambda;
    
    if (!PyArg_ParseTuple(args, "iiifi", &dim, &num_comp_indices, &num_simp_indices, &lambda, &max_volume)) return NULL;
    
    py_dci *py_dci_inst = (py_dci *)malloc(sizeof(py_dci));
        
    dci_init(&(py_dci_inst->dci_inst), dim, num_comp_indices, num_simp_indices, lambda, max_volume);
    
    // py_dci_inst->py_array = NULL;
    hashtable_pp_init(&(py_dci_inst->hashtable), 1, max_volume);
    // py_dci_inst->data_idx_offset = 0;
    
    // Returns new reference
    PyObject *py_dci_inst_wrapper = PyCapsule_New(py_dci_inst, "py_dci_inst", py_dci_free);
    
    return py_dci_inst_wrapper;
}

// Borrows *py_dci_inst_wrapper, py_dci_inst owns at most one copy of *py_data
static PyObject *py_dci_add(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_data;
    PyArrayObject *py_data_id, *py_mask;
    int dim, num_levels, num_to_visit, num_to_retrieve, field_of_view, num_new_points;
    float prop_to_visit, prop_to_retrieve;
    unsigned char blind;
    dci_query_config construction_query_config;
    py_dci *py_dci_inst;
    float *data;
    long long *d_id;
    bool *mask;
    
    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "OO!O!O!ibiiffi", &py_dci_inst_wrapper, &PyArray_Type, &py_data, &PyArray_Type, &py_mask, &PyArray_Type, &py_data_id, &num_levels, &blind, &num_to_visit, &num_to_retrieve, &prop_to_visit, &prop_to_retrieve, &field_of_view)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    if (!py_data) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    // Assuming row-major layout, py_data->data is N x D, where N is the number of data points and D is the dimensionality
    data = (float *)py_data->data;
    mask = (bool *)py_mask->data;
	num_new_points = py_data->dimensions[0];
	dim = py_data->dimensions[1];
    d_id = (long long *)py_data_id->data;
    if (py_data_id->dimensions[0] == 0) {
        d_id = NULL;
    }
	
    if (num_new_points > 0) {
        
        construction_query_config.blind = blind;
        construction_query_config.num_to_visit = num_to_visit;
        construction_query_config.num_to_retrieve = num_to_retrieve;
        construction_query_config.prop_to_visit = prop_to_visit;
        construction_query_config.prop_to_retrieve = prop_to_retrieve;
        construction_query_config.field_of_view = field_of_view;
        construction_query_config.target_level = 0;
        
       long long first_id = dci_add(&(py_dci_inst->dci_inst), dim, num_new_points, data, num_levels, mask, construction_query_config, NULL, NULL, 0);
    }
    
    Py_INCREF(Py_None);
    return Py_None;
}

// Borrows *py_dci_inst_wrapper, py_dci_inst owns at most one copy of *py_data
static PyObject *py_dci_delete(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_data;
    int dim, num_to_visit, num_to_retrieve, field_of_view, num_delete_points;
    float prop_to_visit, prop_to_retrieve;
    unsigned char blind;
    dci_query_config construction_query_config;
    py_dci *py_dci_inst;
    long long *data_ids;
    
    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "OO!biiffi", &py_dci_inst_wrapper, &PyArray_Type, &py_data, &blind, &num_to_visit, &num_to_retrieve, &prop_to_visit, &prop_to_retrieve, &field_of_view)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    if (!py_data) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    // Assuming row-major layout, py_data->data is N x D, where N is the number of data points and D is the dimensionality
    data_ids = (long long *)py_data->data;
	num_delete_points = py_data->dimensions[0];
	dim = py_data->dimensions[1];
	
    if (num_delete_points > 0) {
        
        construction_query_config.blind = blind;
        construction_query_config.num_to_visit = num_to_visit;
        construction_query_config.num_to_retrieve = num_to_retrieve;
        construction_query_config.prop_to_visit = prop_to_visit;
        construction_query_config.prop_to_retrieve = prop_to_retrieve;
        construction_query_config.field_of_view = field_of_view;
        construction_query_config.target_level = 0;
        
       int num_deleted = dci_delete(&(py_dci_inst->dci_inst), num_delete_points, data_ids, construction_query_config);
        // py_dci_inst->data_idx_offset = start_idx;
        // py_dci_inst->py_array = py_data;

        for (int i = 0; i < num_delete_points; i++) {
            PyArrayObject *py_array = hashtable_pp_get(&(py_dci_inst->hashtable), data_ids[i], NULL);
            if (py_array) {
                assert(hashtable_pp_delete(&(py_dci_inst->hashtable), data_ids[i]));
                Py_DECREF(py_array);
                num_deleted -= 1;
            }
        }
        assert(num_deleted == 0);
    }
    
    Py_DECREF(Py_None);
    return Py_None;
}

static PyObject *py_dci_query(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_query, *py_mask, *py_nearest_neighbour_idx, *py_nearest_neighbour_dists, *py_num_returned;
    int i, j, k, dim, num_neighbours, num_to_visit, num_to_retrieve, num_queries, field_of_view;
    unsigned char blind;
    float prop_to_visit, prop_to_retrieve, scale;
    py_dci *py_dci_inst;
    bool *mask;
    float *query, *nearest_neighbour_dists_flattened;
    int *nearest_neighbour_idx, *num_returned;
    dci_query_config query_config;
    int **nearest_neighbours;
    float **nearest_neighbour_dists;
    npy_intp py_nearest_neighbours_shape[1];
    npy_intp py_num_returned_shape[1];
    
    if (!PyArg_ParseTuple(args, "OO!O!ibiiffif", &py_dci_inst_wrapper, &PyArray_Type, &py_query, &PyArray_Type, &py_mask, &num_neighbours, &blind, &num_to_visit, &num_to_retrieve, &prop_to_visit, &prop_to_retrieve, &field_of_view, &scale)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    if (!py_query) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    // Assuming row-major layout, py_query->data is N x D, where N is the number of queries and D is the dimensionality
    query = (float *)py_query->data;
	num_queries = py_query->dimensions[0];
	dim = py_query->dimensions[1];
        
    py_num_returned_shape[0] = num_queries;
    mask = (bool *)py_mask->data;
    
    py_num_returned = (PyArrayObject *)PyArray_SimpleNew(1, py_num_returned_shape, NPY_INT);
    num_returned = (int *)py_num_returned->data;
    
    query_config.blind = blind;
    query_config.num_to_visit = num_to_visit;
    query_config.num_to_retrieve = num_to_retrieve;
    query_config.prop_to_visit = prop_to_visit;
    query_config.prop_to_retrieve = prop_to_retrieve;
    query_config.field_of_view = field_of_view;
    query_config.target_level = 0;
    
    nearest_neighbours = (int **)malloc(sizeof(int *)*num_queries);
    nearest_neighbour_dists = (float **)malloc(sizeof(float *)*num_queries);
    
    dci_query(&(py_dci_inst->dci_inst), dim, num_queries, query, num_neighbours, query_config, mask, nearest_neighbours, nearest_neighbour_dists, num_returned, scale, 0);

    py_nearest_neighbours_shape[0] = 0;
    for (i = 0; i < num_queries; i++) {
        py_nearest_neighbours_shape[0] += num_returned[i];
    }
    
    py_nearest_neighbour_idx = (PyArrayObject *)PyArray_SimpleNew(1, py_nearest_neighbours_shape, NPY_INT);
    nearest_neighbour_idx = (int *)py_nearest_neighbour_idx->data;
    
    k = 0;
    for (i = 0; i < num_queries; i++) {
        for (j = 0; j < num_returned[i]; j++) {
            // nearest_neighbour_idx[k] = nearest_neighbours[i][j] + py_dci_inst->data_idx_offset;
            nearest_neighbour_idx[k] = nearest_neighbours[i][j];
            k++;
        }
    }
    
    // Assuming row-major layout, matrix is of size num_queries x num_neighbours
    py_nearest_neighbour_dists = (PyArrayObject *)PyArray_SimpleNew(1, py_nearest_neighbours_shape, NPY_FLOAT);
    nearest_neighbour_dists_flattened = (float *)py_nearest_neighbour_dists->data;
    k = 0;
    for (i = 0; i < num_queries; i++) {
        for (j = 0; j < num_returned[i]; j++) {
            nearest_neighbour_dists_flattened[k] = nearest_neighbour_dists[i][j];
            k++;
        }
    }
    
    for (i = 0; i < num_queries; i++) {
        free(nearest_neighbours[i]);
    }
    free(nearest_neighbours);
    for (i = 0; i < num_queries; i++) {
        free(nearest_neighbour_dists[i]);
    }
    free(nearest_neighbour_dists);
    
    return Py_BuildValue("NNN", py_nearest_neighbour_idx, py_nearest_neighbour_dists, py_num_returned);
}

static PyObject *py_dci_add_query(PyObject *self, PyObject *args) {

    PyArrayObject *py_data, *py_data_id, *py_new_value, *py_attention_mask;
    PyArrayObject *py_query, *py_value, *py_num_returned, *py_mask;
    int i, j, idx, dim, dim_v, num_levels, num_inst, c_num_to_visit, c_num_to_retrieve, c_field_of_view, max_num_points;
    int num_neighbours, q_num_to_visit, q_num_to_retrieve, q_field_of_view;
    float c_prop_to_visit, c_prop_to_retrieve, q_prop_to_visit, q_prop_to_retrieve, scale;
    unsigned char blind;
    dci_query_config construction_query_config, query_config;
    float *data;
    bool *mask;
    int* attention_mask;
    long long *d_id;
    float *query, *value;
    int num_comp_indices, num_simp_indices;
    float lambda;
    int parallel_level;
    
    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiibiiiiffffiiiififiO!", &PyArray_Type, &py_data, &PyArray_Type, &py_data_id, 
                                                                                &PyArray_Type, &py_query,  &PyArray_Type, &py_value, &PyArray_Type, &py_mask, 
                                                                                &num_inst, &num_levels, &num_neighbours, &blind, &c_num_to_visit, &q_num_to_visit, 
                                                                                &c_num_to_retrieve, &q_num_to_retrieve, &c_prop_to_visit, &q_prop_to_visit, 
                                                                                &c_prop_to_retrieve, &q_prop_to_retrieve, &c_field_of_view, &q_field_of_view, 
                                                                                &num_comp_indices, &num_simp_indices, &lambda, &max_num_points, &scale, &parallel_level,
                                                                                &PyArray_Type, &py_attention_mask))  return NULL;
    if (!py_data) return NULL;
    
    // Assuming row-major layout, py_data->data is N x D, where N is the number of data points and D is the dimensionality
    data = (float *)py_data->data;
	dim = py_data->dimensions[1];
    dim_v = py_value->dimensions[1];
    d_id = NULL;
    // Assuming row-major layout, py_query->data is N x D, where N is the number of queries and D is the dimensionality
    query = (float *)py_query->data;
    value = (float *)py_value->data;
    mask = (bool *)py_mask->data;
    attention_mask = (int *)py_attention_mask->data;
    
    construction_query_config.blind = blind;
    construction_query_config.num_to_visit = c_num_to_visit;
    construction_query_config.num_to_retrieve = c_num_to_retrieve;
    construction_query_config.prop_to_visit = c_prop_to_visit;
    construction_query_config.prop_to_retrieve = c_prop_to_retrieve;
    construction_query_config.field_of_view = c_field_of_view;
    construction_query_config.target_level = 0;

    query_config.blind = blind;
    query_config.num_to_visit = q_num_to_visit;
    query_config.num_to_retrieve = q_num_to_retrieve;
    query_config.prop_to_visit = q_prop_to_visit;
    query_config.prop_to_retrieve = q_prop_to_retrieve;
    query_config.field_of_view = q_field_of_view;
    query_config.target_level = 0;

    int *num_returned = (int *)malloc(sizeof(int)*max_num_points*num_inst);

    float* new_value_flattened = (float*)calloc(max_num_points * dim_v * num_inst, sizeof(float));
    npy_intp py_new_value_shape[1];
    py_new_value_shape[0] = max_num_points * dim_v * num_inst;
    py_new_value = (PyArrayObject *)PyArray_SimpleNewFromData(1, py_new_value_shape, NPY_FLOAT, new_value_flattened);
    PyObject *capsule = PyCapsule_New(new_value_flattened, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) py_new_value, capsule);
    
    
    int **nearest_neighbours = (int **)malloc(sizeof(int *)*max_num_points*num_inst);
    float **nearest_neighbour_dists = (float **)malloc(sizeof(float *)*max_num_points*num_inst);

    py_dci py_dci_inst[num_inst];
    
    for (idx = 0; idx < num_inst; idx++) {
        dci_init(&(py_dci_inst[idx].dci_inst), dim, num_comp_indices, num_simp_indices, lambda, max_num_points);
    }

    int num_points = 0;
    for (i = 0; i < max_num_points; i++) {
        if (mask[i])
            num_points++;
    }

    for (i = 0; i < max_num_points; i++) {
        if (attention_mask[i] > num_points)
            attention_mask[i] = num_points;
    }

#pragma omp parallel for if(parallel_level >= 1)
    for (idx = 0; idx < num_inst; idx++) {
        dci *py_dci_inst_temp = &(py_dci_inst[idx].dci_inst);
        py_dci_inst_temp->parallel_level = parallel_level;

        float* data_proj;

        int promotion_ratio = (int)ceil(pow((float)num_points, 1.0 / num_levels));
        additional_info* new_data_root;
        int num_indices = num_comp_indices * num_simp_indices;
        additional_info* new_data_level_cells = (additional_info*)malloc(sizeof(additional_info) * num_points);
        int* new_data_num_points_on_level = (int*)malloc(sizeof(int)*num_levels);
        int** new_points_on_level = (int**)malloc(sizeof(int*)*num_levels);
        for (int i = 0; i < num_levels; i++) {
            new_data_num_points_on_level[i] = 0;
            // TODO: need to think a better way to allocate memory (now is wasteful)
            new_points_on_level[i] = (int*)malloc(sizeof(int)*num_points);
        }
        float* norm_list = (float*)malloc(sizeof(float)*max_num_points);

        data_projection(py_dci_inst_temp->num_comp_indices, py_dci_inst_temp->num_simp_indices,
            py_dci_inst_temp->proj_vec, py_dci_inst_temp->add_proj_vec, &new_data_root, dim, max_num_points,
            &(data[max_num_points * idx * dim]), &(mask[max_num_points * idx]), norm_list, &(py_dci_inst_temp->max_norm), &data_proj);

        initialize_tree(num_comp_indices, num_simp_indices, &new_data_root);

        for (int i = 0; i < max_num_points; i++) {
            if (mask[i]) {
                py_dci_inst_temp->norm_list[i] = norm_list[i];
            }
        }
        free(norm_list);
        
        py_dci_inst_temp->root = new_data_root;
        py_dci_inst_temp->num_levels = num_levels;
        py_dci_inst_temp->num_points_on_level = new_data_num_points_on_level;
        py_dci_inst_temp->points_on_level = new_points_on_level;
        py_dci_inst_temp->info_addr = new_data_level_cells;

        float **nearest_neighbour_dists_temp = &(nearest_neighbour_dists[max_num_points * idx]);
        int **nearest_neighbour_temp = &(nearest_neighbours[max_num_points * idx]);
        float *value_temp = &(value[max_num_points * idx * dim_v]);

        if (attention_mask[0] > 1) {

            long long first_id = dci_add(py_dci_inst_temp, dim, attention_mask[0], &(data[max_num_points * idx * dim]),
                                        num_levels, &(mask[max_num_points * idx]), construction_query_config, 0, data_proj, 0);
            
            py_dci_inst_temp->next_point_id += attention_mask[0];
            py_dci_inst_temp->num_points += attention_mask[0];
            
            dci_query(py_dci_inst_temp, dim, attention_mask[0], &(query[max_num_points * idx * dim]), num_neighbours, query_config, 
                                &(mask[max_num_points * idx]), nearest_neighbour_temp, nearest_neighbour_dists_temp, 
                                &(num_returned[max_num_points * idx]), scale, py_dci_inst_temp->num_levels);

            float *new_value_temp;
            for (int i = 0; i < attention_mask[0]; i++) {
                if (!(mask[max_num_points * idx + i])) continue;
                new_value_temp = &(new_value_flattened[(max_num_points * idx + i) * dim_v]);
                for (int j = 0; j < num_returned[max_num_points * idx + i]; j++) {
                    float* x = value_temp + dim_v * nearest_neighbour_temp[i][j];
                    float weight = nearest_neighbour_dists_temp[i][j];
#ifndef NO_SIMD
                    __m256 X, Y; // 256-bit values
                    __m256 dot = _mm256_setzero_ps(); // set to (0, 0, 0, 0, 0, 0, 0, 0)
                    float temp[8];
                    float y[8] = {weight, weight, weight, weight, weight, weight, weight, weight};
                    Y = _mm256_loadu_ps(y);
                    int ii;
                    for (ii = 0; ii < dim_v - 8; ii += 8)
                    {
                        X = _mm256_loadu_ps(x + ii); // load chunk of 8 floats
                        dot =  _mm256_mul_ps(X, Y);
                        _mm256_storeu_ps(&temp[0], dot);
                        new_value_temp[ii] += temp[0];
                        new_value_temp[ii+1] += temp[1];
                        new_value_temp[ii+2] += temp[2];
                        new_value_temp[ii+3] += temp[3];
                        new_value_temp[ii+4] += temp[4];
                        new_value_temp[ii+5] += temp[5];
                        new_value_temp[ii+6] += temp[6];
                        new_value_temp[ii+7] += temp[7];
                    }
                    for (; ii < dim_v; ii++)
                        new_value_temp[ii] += x[ii] * weight;
#else
                    for (int ii = 0; ii < dim_v; ii++)
                        new_value_temp[ii] += x[ii] * weight;
#endif
                }
            }
            for (int i = 0; i < attention_mask[0]; i++) {
                if ((mask[max_num_points * idx + i])) {
                    free(nearest_neighbour_temp[i]);
                    free(nearest_neighbour_dists_temp[i]);
                } 
            }
        }
        int r_idx = 0;
        if (attention_mask[0] > 1) {
            for (r_idx = 1; r_idx < max_num_points; r_idx++) {
                if (attention_mask[r_idx] > attention_mask[r_idx - 1])
                    break;
            }
        }
        if (r_idx < max_num_points) {
            int target_level;
            int num_populated_levels = 0;
            int next_target_level = 0;

            for (int i = r_idx; i < max_num_points; i++) {
                if (!(mask[max_num_points * idx + i])) continue;

                // Decide which level to add in
                if (i < num_levels) {
                    target_level = num_levels - 1 - i;
                }
                else {
                    target_level = next_target_level;
                    if ((target_level < num_levels - 1) && ((py_dci_inst_temp->num_points_on_level[target_level] + 1) == promotion_ratio * (py_dci_inst_temp->num_points_on_level[target_level + 1])))
                        next_target_level = target_level + 1;
                    else
                        next_target_level = 0;
                }
                if (target_level < num_levels - num_populated_levels){
                    num_populated_levels = num_levels - target_level;
                }
                long long first_id = dci_add(py_dci_inst_temp, dim, 1, &(data[max_num_points * idx * dim]), 
                                                                num_levels, &(mask[max_num_points * idx + i]), construction_query_config, i, data_proj, target_level);
                
                dci_query(py_dci_inst_temp, dim, 1, &(query[max_num_points * idx * dim + i * dim]), num_neighbours, query_config, 
                                    &(mask[max_num_points * idx + i]), nearest_neighbour_temp, nearest_neighbour_dists_temp, 
                                    &(num_returned[max_num_points * idx]), scale, num_populated_levels);

                float *new_value_temp;
                new_value_temp = &(new_value_flattened[(max_num_points * idx + i) * dim_v]);
                for (int j = 0; j < num_returned[max_num_points * idx]; j++) {
                    float* x = value_temp + dim_v * nearest_neighbour_temp[0][j];
                    float weight = nearest_neighbour_dists_temp[0][j];
#ifndef NO_SIMD
                    __m256 X, Y; // 256-bit values
                    __m256 dot = _mm256_setzero_ps(); // set to (0, 0, 0, 0, 0, 0, 0, 0)
                    float temp[8];
                    float y[8] = {weight, weight, weight, weight, weight, weight, weight, weight};
                    Y = _mm256_loadu_ps(y);
                    int ii;
                    for (ii = 0; ii < dim_v - 8; ii += 8)
                    {
                        X = _mm256_loadu_ps(x + ii); // load chunk of 8 floats
                        dot =  _mm256_mul_ps(X, Y);
                        _mm256_storeu_ps(&temp[0], dot);
                        new_value_temp[ii] += temp[0];
                        new_value_temp[ii+1] += temp[1];
                        new_value_temp[ii+2] += temp[2];
                        new_value_temp[ii+3] += temp[3];
                        new_value_temp[ii+4] += temp[4];
                        new_value_temp[ii+5] += temp[5];
                        new_value_temp[ii+6] += temp[6];
                        new_value_temp[ii+7] += temp[7];
                    }
                    for (; ii < dim_v; ii++)
                        new_value_temp[ii] += x[ii] * weight;
#else
                    for (int ii = 0; ii < dim_v; ii++)
                        new_value_temp[ii] += x[ii] * weight;
#endif
                }
                free(nearest_neighbour_temp[0]);
                free(nearest_neighbour_dists_temp[0]);
            }
        }
        dci_free(py_dci_inst_temp);
        free(data_proj);     
    }
    
    free(nearest_neighbours);
    free(nearest_neighbour_dists);
    free(num_returned);
    
    return Py_BuildValue("N", py_new_value);
}

// Borrows *py_dci_inst_wrapper, relinquishes *py_data
static PyObject *py_dci_clear(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
	
    // if (py_dci_inst->py_array) {
    //     Py_DECREF(py_dci_inst->py_array);
    // }
    
    dci_clear(&(py_dci_inst->dci_inst));
    hashtable_pp_clear(&(py_dci_inst->hashtable));
    // py_dci_inst->py_array = NULL;
    // py_dci_inst->data_idx_offset = 0;
    
    Py_INCREF(Py_None);
    return Py_None;
    
}

// Borrows *py_dci_inst_wrapper, relinquishes *py_data
static PyObject *py_dci_reset(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    // if (py_dci_inst->py_array) {
    //     Py_DECREF(py_dci_inst->py_array);
    // }
    
    dci_reset(&(py_dci_inst->dci_inst));
    int size = py_dci_inst->hashtable.size;
    hashtable_pp_clear(&(py_dci_inst->hashtable));
    hashtable_pp_init(&(py_dci_inst->hashtable), 1, size);
    // py_dci_inst->py_array = NULL;
    // py_dci_inst->data_idx_offset = 0;
    
    Py_INCREF(Py_None);
    return Py_None;
    
}

static PyObject *py_dci_get_num_points(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
	return Py_BuildValue("i", (py_dci_inst->dci_inst).num_points);
}

static PyObject *py_dci_get_num_levels(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
	return Py_BuildValue("i", (py_dci_inst->dci_inst).num_levels);
}

static PyObject *py_dci_get_proj_vec(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    PyArrayObject *py_proj_vec;
    npy_intp py_proj_vec_shape[2];
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst = (py_dci *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst");
    
    py_proj_vec_shape[0] = (py_dci_inst->dci_inst).num_comp_indices*(py_dci_inst->dci_inst).num_simp_indices;
    py_proj_vec_shape[1] = (py_dci_inst->dci_inst).dim;
    // Assuming row-major layout, matrix is of size (num_comp_indices*num_simp_indices) x dim
    py_proj_vec = (PyArrayObject *)PyArray_SimpleNewFromData(2, py_proj_vec_shape, NPY_FLOAT, (py_dci_inst->dci_inst).proj_vec);
    // py_proj_vec owns a reference to py_dci_inst_wrapper
    py_proj_vec->base = py_dci_inst_wrapper;
    Py_INCREF(py_dci_inst_wrapper);
    
    return (PyObject *)py_proj_vec;
}

// Methods table - maps names in Python to C functions  
static PyMethodDef py_dci_module_methods[] = {
    {"_dci_new", py_dci_new, METH_VARARGS, "Create new DCI instance."},
    {"_dci_add", py_dci_add, METH_VARARGS, "Add data."},
    {"_dci_query", py_dci_query, METH_VARARGS, "Search for nearest neighbours."},
    {"_dci_add_query", py_dci_add_query, METH_VARARGS, "Add and Search for nearest neighbours."},
    {"_dci_clear", py_dci_clear, METH_VARARGS, "Delete all data."},
    {"_dci_reset", py_dci_reset, METH_VARARGS, "Delete all data and regenerate projection directions."},
    {"_dci_get_num_points", py_dci_get_num_points, METH_VARARGS, "Get the number of points indexed by DCI instance. "},
    {"_dci_get_num_levels", py_dci_get_num_levels, METH_VARARGS, "Get the number of levels in DCI instance. "},
    {"_dci_get_proj_vec", py_dci_get_proj_vec, METH_VARARGS, "Get the projection vectors used by DCI instance. "},
    {"_dci_delete", py_dci_delete, METH_VARARGS, "Delete data."},
    {NULL, NULL, 0, NULL}
};

#ifdef PY3K

static int py_dci_module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int py_dci_module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef py_dci_module_def = {
        PyModuleDef_HEAD_INIT,
        "_dci",
        NULL,
        sizeof(struct module_state),
        py_dci_module_methods,
        NULL,
        py_dci_module_traverse,
        py_dci_module_clear,
        NULL
};

// Module name is "_dci"
PyMODINIT_FUNC PyInit__dci(void) {
    PyObject *module = PyModule_Create(&py_dci_module_def);
    import_array();     // Import Numpy
    return module;
}

#else

// Module name is "_dci"
PyMODINIT_FUNC init_dci(void) {
    (void) Py_InitModule("__dci", py_dci_module_methods);
    import_array();     // Import Numpy
}

#endif
