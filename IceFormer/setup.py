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

import sys
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
from numpy.distutils.core import setup

def build_ext(config, dist):
    
    lapack_info = get_info('lapack_opt', 1)
    dci_sources = ['src/dci.c', 'src/py_dci.c', 'src/util.c', 'src/hashtable_i.c', 'src/hashtable_d.c', 'src/btree_i.c', 'src/btree_p.c', 'src/hashtable_p.c', 'src/hashtable_pp.c']
    dci_headers = ['include/dci.h', 'include/util.h', 'include/hashtable_i.h', 'include/hashtable_d.h', 'include/btree_i.h', 'include/btree_p.h', 'include/hashtable_p.h', 'include/hashtable_pp.h']
    if lapack_info:
        answer = input("Enable Multithreading? (Y/N)\n")
        if answer.lower().startswith("y"):
            config.add_extension(name='_dci',sources=dci_sources, depends=dci_headers, include_dirs=['include'], extra_info=lapack_info, extra_compile_args=['-fopenmp', '-DUSE_OPENMP', '-march=core-avx2'], extra_link_args=['-lgomp'])
        else:
            config.add_extension(name='_dci',sources=dci_sources, depends=dci_headers, include_dirs=['include'], extra_info=lapack_info)

    if not lapack_info:
        raise ImportError("No BLAS library found.")

    config_dict = config.todict()
    try:
        config_dict.pop('packages')
    except:
        pass

    return config_dict

def setup_dci(dist):

    config_dict = build_ext(Configuration('dciknn', parent_package=None, top_path=None), dist)
    
    setup(  version="0.1.0",
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
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3.4',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',
                'Programming Language :: Python :: 3.8',
                'Programming Language :: Python :: 3.9',
                'Programming Language :: Python :: 3.10',
                'Programming Language :: Python :: 3.11',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Software Development :: Libraries :: Python Modules',
                 ],
            requires=['NumPy',],
            long_description="""
            Dynamic Continuous Indexing (DCI) is a family of randomized algorithms for
            exact k-nearest neighbour search that overcomes the curse of dimensionality.
            Its query time complexity is linear in ambient dimensionality and sublinear
            in intrinsic dimensionality. ``dciknn`` is a python package that contains
            the reference implementation of a modified version of DCI 
            and a convenient Python interface which can be used to accelerate Transformers. 

            ``dciknn`` requires ``NumPy``. 
            """,
            packages=["dciknn"],
            **(config_dict))
            
if __name__ == '__main__':
    dist = sys.argv[1]
    setup_dci(dist)
