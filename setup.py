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

from setuptools import setup, Extension, find_packages
import numpy as np
import os
import platform

# Get the directory containing this setup.py
here = os.path.abspath(os.path.dirname(__file__))

# Change to the IceFormer subdirectory for building
iceformer_dir = os.path.join(here, 'IceFormer')

# Define the extension module
dci_sources = [
    'IceFormer/src/dci.c',
    'IceFormer/src/py_dci.c', 
    'IceFormer/src/util.c',
    'IceFormer/src/hashtable_i.c',
    'IceFormer/src/hashtable_d.c',
    'IceFormer/src/btree_i.c',
    'IceFormer/src/btree_p.c',
    'IceFormer/src/hashtable_p.c',
    'IceFormer/src/hashtable_pp.c'
]

# Platform-specific compiler flags
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':  # macOS
    if platform.machine() == 'arm64':
        extra_compile_args = ['-O3', '-DNO_SIMD']
    else:
        extra_compile_args = ['-O3']
    extra_link_args = []
else:  # Linux and others
    extra_compile_args = ['-fopenmp', '-DUSE_OPENMP', '-march=native', '-O3']
    extra_link_args = ['-lgomp']

dci_extension = Extension(
    'dciknn._dci',
    sources=dci_sources,
    include_dirs=[
        'IceFormer/include',
        np.get_include()
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=['blas', 'lapack']
)

setup(
    name="dciknn",
    version="0.1.0",
    description="(Modified) Dynamic Continuous Indexing reference implementation.",
    author="Yuzhen Mao, Ke Li",
    author_email="yuzhenm@sfu.ca",
    url="https://yuzhenmao.github.io/IceFormer/",
    license="Mozilla Public License 2.0",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=['dciknn'],
    package_dir={'dciknn': 'IceFormer/dciknn'},
    install_requires=[
        'numpy',
    ],
    ext_modules=[dci_extension],
    long_description="""
    Dynamic Continuous Indexing (DCI) is a family of randomized algorithms for
    exact k-nearest neighbour search that overcomes the curse of dimensionality.
    Its query time complexity is linear in ambient dimensionality and sublinear
    in intrinsic dimensionality. ``dciknn`` is a python package that contains
    the reference implementation of a modified version of DCI 
    and a convenient Python interface which can be used to accelerate Transformers. 

    ``dciknn`` requires ``NumPy``. 
    """,
    python_requires='>=3.6',
)