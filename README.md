# IceFormer: Accelerated Inference with Long-Sequence Transformers on CPUs

[![LicenseMPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://github.com/yuzhenmao/Iceformer/blob/main/LICENSE)

### [Project](https://yuzhenmao.github.io/IceFormer/) | [Talk Video](https://www.youtube.com/watch?v=6W0DtYRzFng&t=1s) | [Paper](https://arxiv.org/abs/2405.02842)

The official implementation of _**IceFormer: Accelerated Inference with Long-Sequence Transformers on CPUs**_ (ICLR 2024).

<img width="825" alt="Screenshot 2024-03-05 at 1 40 34 PM" src="https://github.com/yuzhenmao/IceFormer/assets/57878927/5f658c13-16ba-4435-a488-3e733870ed10">

This repository contains the reference implementation of Multi-level Dynamic Continuous Indexing (Multi-level DCI), which was written in C to take advantage of compile-time optimizations and multi-threading. It comes with a C interface, and a Python 2 & 3 interface. Currently, the code only runs on the CPU. GPU support will be added in the future. 

Dynamic Continuous Indexing (DCI) is a family of randomized algorithms for exact _k_-nearest neighbour search that overcomes the curse of dimensionality. Its query time complexity is linear in ambient dimensionality and sublinear in intrinsic dimensionality. Details of the algorithm and analysis of time complexity can be found in the following papers:

"[Fast _k_-Nearest Neighbour Search via Dynamic Continuous Indexing](https://arxiv.org/abs/1512.00442)", _International Conference on Machine Learning (ICML)_, 2016\
"[Fast _k_-Nearest Neighbour Search via Prioritized DCI](https://arxiv.org/abs/1703.00440)", _International Conference on Machine Learning (ICML)_, 2017


# Getting Started

## 1. Build the `dciknn` Package

To get started, follow the setup instructions located in the [`./IceFormer`](https://github.com/yuzhenmao/IceFormer/tree/main/IceFormer#prerequisites) folder.

## 2. Implement IceFormer Attention

Replace the default self-attention mechanism in the target model with the IceFormer attention module. Here is an example:

default self-attention

```python
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = 0.1)
        self.head_dim = 128

    def forward(self, Q, K, V, mask):
        # Q.shape [batch_size, num_heads, seq_len, head_dim]
        # K.shape [batch_size, num_heads, seq_len, head_dim]
        # V.shape [batch_size, num_heads, seq_len, head_dim]
        # mask.shape [batch_size, seq_len]
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot.masked_fill(mask, float("-inf"))

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        X = torch.matmul(attn, V)

        # output [batch_size, num_heads, seq_len, head_dim]
        return X
```

IceFormer attention

```python
try:
    from dciknn import DCI
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dciknn import DCI

class IceFormerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = 0.1)
        self.head_dim = 128
        self.top_k = 20
        self.dci_layers = 3

    def forward(self, Q, K, V, mask):
        # Q.shape [batch_size, num_heads, seq_len, head_dim]
        # K.shape [batch_size, num_heads, seq_len, head_dim]
        # V.shape [batch_size, num_heads, seq_len, head_dim]
        # mask.shape [batch_size, seq_len]

        batch_size, num_heads, seq_len, head_dim = Q.size()
        key = np.array(K.detach().numpy().reshape(-1, head_dim), order='C').astype(np.float32)
        query = np.array(Q.detach().numpy().reshape(-1, head_dim), order='C').astype(np.float32)
        value = np.array(V.detach().numpy().reshape(-1, head_dim), order='C').astype(np.float32)
        mask = mask.numpy().astype(np.bool_).repeat(num_heads, axis=0).reshape(-1)

        num_neighbours = self.top_k
        num_comp_indices = 2
        num_simp_indices = 4
        num_levels = self.dci_layers
        construction_field_of_view = 5
        construction_prop_to_retrieve = 0.002
        query_field_of_view = num_neighbours
        query_prop_to_retrieve = 0.8
        blind = False
        num_to_visit = seq_len
        num_to_retrieve = -1
        prop_to_visit = 1.0
        prop_to_retrieve = construction_prop_to_retrieve
        field_of_view = construction_field_of_view

        dci_db = DCI(head_dim, num_comp_indices, num_simp_indices, max_num_points=seq_len)
        new_v = dci_db.add_query(key, query, value, mask,
                                num_levels=num_levels,
                                num_inst = num_heads*batch_size,
                                num_points=seq_len,
                                num_neighbours=num_neighbours,
                                c_num_to_visit=num_to_visit, 
                                c_num_to_retrieve=num_to_retrieve,
                                c_prop_to_visit=prop_to_visit,
                                c_prop_to_retrieve=prop_to_retrieve,
                                c_field_of_view=field_of_view, 
                                q_num_to_visit=num_to_visit,
                                q_field_of_view=query_field_of_view,
                                q_num_to_retrieve=num_to_retrieve,
                                q_prop_to_visit=prop_to_visit,
                                q_prop_to_retrieve=query_prop_to_retrieve,
                                parallel_level=1,
                                causal=False,
                              )

        # output [batch_size, num_heads, seq_len, head_dim]
        
        return torch.from_numpy(new_v).float().reshape(V.shape)
```

# Reference

Please cite the following paper if you found this library useful in your research:

### IceFormer: Accelerated Inference with Long-Sequence Transformers on CPUs
[Yuzhen Mao](https://scholar.google.com/citations?user=9wKn1A0AAAAJ&hl=en), [Martin Ester](https://sites.google.com/view/esterlab), [Ke Li](https://www.sfu.ca/~keli/)\
*International Conference on Learning Representations (ICLR)*, 2024

```
@inproceedings{
  mao2024iceformer,
  title={IceFormer: Accelerated Inference with Long-Sequence Transformers on {CPU}s},
  author={Yuzhen Mao and Martin Ester and Ke Li},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
}
```
