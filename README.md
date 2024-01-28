# IceFormer: Accelerated Inference with Long-Sequence Transformers on CPUs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yuzhenmao/Iceformer/blob/main/LICENSE)

The official implementation of IceFormer: Accelerated Inference with Long-Sequence Transformers on CPUs (ICLR 2024).

This repository contains the reference implementation of Multi-level DCI, which was written in C to take advantage of compile-time optimizations and multi-threading. It comes with a C interface, and a Python 3 interface. Currently, the code only runs on the CPU. GPU support will be added in the future. 

# Prerequisites

1. A C compiler with support for OpenMP, e.g.: gcc
2. Python 3.1+
3. A BLAS library (supported implementations include the reference implementation from Netlib, ATLAS, OpenBLAS and MKL)
4. Python development headers
5. (If Python interface is desired) NumPy

# Setup

The library can be compiled in one of two ways: using Python distutils or the good old Makefile. The former requires less manual configuration, but *cannot* be used if your code uses the C interface. 

**Note:** If your Python interpreter is named differently, e.g.: "python3", you will need to replace all occurrences of "python" with "python3" in the commands below.

## Option 1: Python distutils

If you have sudo access, run the following command from the root directory of the code base to compile and install as a Python package:
```bash
sudo python setup.py install
```

If you do not have sudo access, run the following command instead:
```bash
python setup.py install --user
```

## Option 2: Makefile 

In the root directory of the code base, follow the instructions in the Makefile to specify the paths to BLAS, and optionally, Python, NumPy and TensorFlow. 

### Python Interface

If you would like to build the Python interface, run the following from the root directory of the code base:
```bash
make py
```

If you would like to use DCI in a script outside of the root directory of the code base, either add a symbolic link to the "dciknn" subdirectory within the directory containing your script, or add the root directory of the code base to your PYTHONPATH environment variable. 

### C Interface

If you would like to build a binary executable from code that uses the C interface, run the following from the root directory of the code base:
```bash
make c
```

# Getting Started

In the root directory of the code base, execute the following commands, depending on which interface you would like to use:

### Python Interface

```bash
python examples/example.py
```


### C Interface

```bash
examples/example
```

See the source code for example usage. The source code of the binary executable that uses the C interface is in "src/example.c".

# Reference

Please cite the following paper if you found this library useful in your research:

### IceFormer: Accelerated Inference with Long-Sequence Transformers on CPUs
[Yuzhen Mao](https://scholar.google.com/citations?user=9wKn1A0AAAAJ&hl=en), [Martin Ester](https://sites.google.com/view/esterlab), [Ke Li](https://www.sfu.ca/~keli/)\
*International Conference on Learning Representations (ICLR)*, 2024
