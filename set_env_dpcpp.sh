#!/bin/bash

CLANG_PATH=/home/chenglong/intel/oneapi/compiler/2022.0.2/linux/bin-llvm
SYCL_LIB_PATH=/home/chenglong/intel/oneapi/compiler/2022.0.2/linux/include/sycl

export PATH=$CLANG_PATH:$PATH
export LD_LIBRARY_PATH=$SYCL_LIB_PATH:$LD_LIBRARY_PATH
