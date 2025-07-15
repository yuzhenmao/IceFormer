'''
Code for IceFormer: Accelerated Inference with Long-Sequence Transformers on CPUs

This code implements the method described in the IceFormer paper, 
which can be found at https://openreview.net/forum?id=6RR3wU4mSZ

This file is a part of the Dynamic Continuous Indexing reference 
implementation.


This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Copyright (C) 2024    Yuzhen Mao, Ke Li
'''

import numpy as np
import torch
import os
import sys
import h5py

try:
    from mdci import DCI
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
    from mdci import DCI

from time import time

def gen_data():
    dim = 128
    num_points = 4096
    num_heads = 32
    num_batch = 1
    key = torch.rand((num_batch, num_heads, num_points, dim), dtype=torch.float32)
    query = torch.rand((num_batch, num_heads, num_points, dim), dtype=torch.float32)
    value = torch.rand((num_batch, num_heads, num_points, dim), dtype=torch.float32)

    key = np.array(key.detach().numpy().reshape(-1, dim), order='C').astype(np.float32)
    query = np.array(query.detach().numpy().reshape(-1, dim), order='C').astype(np.float32)
    value = np.array(value.detach().numpy().reshape(-1, dim), order='C').astype(np.float32)
    attention_mask = np.arange(num_points, dtype=np.intc) + 1
    attention_mask[:2000] = (np.ones(2000, dtype=np.intc) * 2000)

    return key, query, value, attention_mask

def main(*args):
    
    #############################################################################################################################################
    #                                                                                                                                           #
    # Data Generation Hyperparameters                                                                                                           #
    #                                                                                                                                           #
    #############################################################################################################################################
    
    dim = 128              # Dimensionality of data
    num_points = 4096      # Number of data points
    num_heads = 32         # Number of attention heads
    num_batch = 1          # Number of batches
    
    #############################################################################################################################################
    #                                                                                                                                           #
    # Problem Hyperparameter                                                                                                                    #
    #                                                                                                                                           #
    #############################################################################################################################################
    
    num_neighbours = 10      # The k in k-NN
    
    #############################################################################################################################################
    #                                                                                                                                           #
    # DCI Hyperparameters                                                                                                                       #
    #                                                                                                                                           #
    #############################################################################################################################################
    
    # Guide for tuning hyperparameters:
    
    # num_comp_indices:                 trades off accuracy vs. construction and query time - high values lead to more accurate results, but 
    #                                   slower construction and querying
    # num_simp_indices:                 trades off accuracy vs. construction and query time - high values lead to more accurate results, but 
    #                                   slower construction and querying. If num_simp_indices is increased, may need to increase 
    #                                   num_comp_indices. If the intrisic dimensionality of dataset increases, should increase num_simp_indices 
    #                                   proportionally. 
    # num_levels:                       trades off construction time vs. query time - higher values lead to faster querying, but slower 
    #                                   construction. If num_levels is increased, may need to increase query_field_of_view and 
    #                                   construction_field_of_view. If the number of data points increases substantially, should increase
    #                                   num_levels logarithmically. 
    # construction_field_of_view:       trades off accuracy/query time vs. construction time - higher values lead to *slightly* more accurate 
    #                                   results and/or *slightly* faster querying, but *slightly* slower construction. If the number of data
    #                                   points increases, need to increase construction_field_of_view at a rate of n^(1/num_levels) unless 
    #                                   num_levels is increased. 
    # construction_prop_to_retrieve:    trades off acrruacy vs. construction time - higher values lead to *slightly* more accurate results, 
    #                                   but slower construction. If the number of data points increases, construction_prop_to_retrieve should 
    #                                   remain roughly the same. If the intrisic dimensionality of dataset increases, should increase 
    #                                   construction_prop_to_retrieve slightly.  
    # query_field_of_view:              trades off accuracy vs. query time - higher values lead to more accurate results, but *slightly* slower 
    #                                   querying. If the number of data points increases, need to increase query_field_of_view at a rate of 
    #                                   n^(1/num_levels) unless num_levels is increased. 
    # query_prop_to_retrieve:           trades off accuracy vs. query time - higher values lead to more accurate results, but slower querying. 
    #                                   If the number of data points increases, query_prop_to_retrieve should remain roughly the same. If the 
    #                                   intrisic dimensionality of dataset increases, should increase query_prop_to_retrieve slightly.
    # parallel_level:                   To control how much of your program uses multi-threading. 
    #                                   Setting a higher parallel_level allows more parts of the program to run in parallel, 
    #                                   but it’s important to match this setting to what your computer can handle, like the number of CPU cores. 
    #                                   A higher setting doesn’t always mean faster speeds. 
    
    num_comp_indices = 2
    num_simp_indices = 4
    num_levels = 3
    construction_field_of_view = 5
    query_field_of_view = num_neighbours
    construction_prop_to_retrieve = 0.002
    query_prop_to_retrieve = 0.8

    num_to_visit = 4096
    num_to_retrieve = -1
    prop_to_visit = 1.0
    
    print("Generating Data... ")
    t0 = time()
    key, query, value, attention_mask = gen_data()
    padding_mask = np.ones([num_batch, num_heads, num_points], dtype=np.bool_).reshape(-1)
    
    print("Took %.4fs" % (time() - t0))

    t0 = time()
    
    # DCI()
    # 
    # Constructs a new DCI database. 
    # 
    # The constructor takes in the following parameters:
    #
    # dim:                              Dimensionality of the vectors. 
    # num_comp_indices:                 Number of composite indices (a small integer like 2 or 3 is recommended). 
    # num_simp_indices:                 Number of simple indices per composite index (a larger integer like 7 or 10 is recommended).
    # max_num_points:                   Maximum number of points to store. 
    dci_db = DCI(dim, num_comp_indices, num_simp_indices, max_num_points=num_points)

    print("Constructing DCI database... ")
    
    # DCI.add_query()
    # 
    # Add data to DCI database and then query.
    # 
    # The method takes in the following parameters:
    # 
    # key:                              A float32 matrix of shape [(batch_size x num of attention heads x num of tokens), dim] 
    #                                   containing the database of points to search over. 
    # query:                            A float32 matrix of shape [(batch_size x num of attention heads x num of tokens), dim] 
    #                                   containing the query points.
    # value:                            A float32 matrix of shape [(batch_size x num of attention heads x num of tokens), dim] 
    #                                   containing the values associated with the database points.
    # padding_mask:                     A boolean matrix of shape [batch_size x num of attention heads x num of tokens] containing the padding mask.
    # num_levels:                       Number of levels (needs to be adjusted according to the num of tokens). 
    # num_neighbours:                   The number of nearest neighbours to return. 
    # c/q_field_of_view:                Maximum number of probes into the next level when querying. Has no effect when num_levels = 1. 
    # c/q_num_to_visit:                 Maximum number of points to visit when construction/querying. A large number like (num of tokens) is recommended.
    # c/q_prop_to_visit:                Maximum proportion of points to visit when construction/querying. A large number like 1.0 is recommended.
    # c/q_prop_to_retrieve:             Maximum proportion of points to retrieve when construction/querying.
    # parallel_level:                   The level at which to parallelize the construction/querying. [0, 1, 2, 3] are valid values.
    # causal:                           Whether to use a causal mask.
    # attention_mask:                   A int32 matrix of shape [num of tokens] containing the attention mask. 
    #                                   Has no effect when causal = True. For vanilla attention, set all values to (num of tokens).
    #
    # The method returns the following:
    # 
    # approx_value:            A list of int32 arrays containing the indices of the nearest neighbours to each query. 

    approx_value = dci_db.add_query(key, 
                                    query, 
                                    value, 
                                    padding_mask,
                                    num_levels=num_levels, 
                                    num_inst = num_heads * num_batch,
                                    num_points=num_points,
                                    num_neighbours=num_neighbours,
                                    c_num_to_visit=num_to_visit, 
                                    c_num_to_retrieve=num_to_retrieve,
                                    c_prop_to_visit=prop_to_visit,
                                    c_prop_to_retrieve=construction_prop_to_retrieve,
                                    c_field_of_view=construction_field_of_view, 
                                    q_num_to_visit=num_to_visit,
                                    q_field_of_view=query_field_of_view,
                                    q_num_to_retrieve=num_to_retrieve,
                                    q_prop_to_visit=prop_to_visit,
                                    q_prop_to_retrieve=query_prop_to_retrieve,
                                    parallel_level=1,
                                    causal=True,
                                    # attention_mask=attention_mask,
                                )    
    print("Took %.4fs" % (time() - t0))
    
if __name__ == '__main__':
    main(*sys.argv[1:])
